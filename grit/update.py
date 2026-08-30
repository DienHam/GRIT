"""Total GRIT gradient assembly for Phase 5.

This module writes the optimizer-facing gradient:

    final_grad = projected_task_grad - lambda_pres * v

or, with curvature enabled:

    final_grad = projected_task_grad - lambda_pres * (v + alpha * H P v)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from grit.curvature import (
    CurvatureCorrectionResult,
    project_vector_with_module_projectors,
    trainable_named_parameters,
)
from grit.predictor import PredictorStepInfo, temporary_predictor_step
from grit.projection import ProjectorBuildResult, default_module_filter
from grit.preservation_loss import PreservationLossResult


ModuleFilter = Callable[[str, nn.Module], bool]
ParameterFilter = Callable[[str, nn.Parameter], bool]
NamedParameter = tuple[str, nn.Parameter]
GradientMap = Mapping[str, torch.Tensor | None]
PreservationLossFn = Callable[[], torch.Tensor | PreservationLossResult]


@dataclass(frozen=True)
class GritUpdateConfig:
    """Configuration for one GRIT optimizer-gradient assembly."""

    alpha: float = 1.0
    lambda_pres: float = 1.0
    use_curvature: bool = False
    missing_projector: str = "identity"
    zero_grad_before_write: bool = True


@dataclass(frozen=True)
class GritUpdateResult:
    """Named gradients and scalar diagnostics from one GRIT update."""

    final_gradients: dict[str, torch.Tensor]
    projected_task_gradients: dict[str, torch.Tensor]
    preservation_gradients: dict[str, torch.Tensor]
    preservation_correction: dict[str, torch.Tensor]
    preservation_loss: torch.Tensor
    predictor: PredictorStepInfo
    curvature: CurvatureCorrectionResult | None
    metrics: dict[str, float]


def _zero_like_parameter(parameter: nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(parameter, memory_format=torch.preserve_format)


def _squared_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(tensor.detach().float().square().sum().item())
    return total


def _coerce_config(config: GritUpdateConfig | None, **overrides) -> GritUpdateConfig:
    if config is None:
        config = GritUpdateConfig()
    values = {
        "alpha": config.alpha,
        "lambda_pres": config.lambda_pres,
        "use_curvature": config.use_curvature,
        "missing_projector": config.missing_projector,
        "zero_grad_before_write": config.zero_grad_before_write,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return GritUpdateConfig(**values)


def _autograd_gradient_map(
    loss: torch.Tensor,
    parameters: Sequence[NamedParameter],
    *,
    retain_graph: bool,
    create_graph: bool = False,
    allow_unused: bool = True,
) -> dict[str, torch.Tensor]:
    tensors = [parameter for _, parameter in parameters]
    gradients = torch.autograd.grad(
        loss,
        tensors,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )
    return {
        name: _zero_like_parameter(parameter) if grad is None else grad.detach().clone()
        for (name, parameter), grad in zip(parameters, gradients, strict=True)
    }


def _write_gradients(
    parameters: Sequence[NamedParameter],
    gradients: Mapping[str, torch.Tensor],
    *,
    zero_grad_before_write: bool,
) -> None:
    for name, parameter in parameters:
        grad = gradients[name].to(device=parameter.device, dtype=parameter.dtype)
        if zero_grad_before_write or parameter.grad is None:
            parameter.grad = grad.detach().clone()
        else:
            parameter.grad.copy_(grad)


def _projector_metrics(
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
) -> dict[str, float]:
    ranks: list[float] = []
    nullities: list[float] = []
    metrics: dict[str, float] = {}
    for name, raw_projector in projectors.items():
        if isinstance(raw_projector, ProjectorBuildResult):
            rank = float(raw_projector.rank)
            nullity = float(raw_projector.nullity)
        else:
            projector = raw_projector.detach().float()
            rank = float(torch.trace(projector).round().item())
            nullity = float(projector.shape[-1] - rank)
        metrics[f"grit/projector/{name}/rank"] = rank
        metrics[f"grit/projector/{name}/nullity"] = nullity
        ranks.append(rank)
        nullities.append(nullity)

    if ranks:
        metrics["grit/projector/rank_mean"] = sum(ranks) / len(ranks)
        metrics["grit/projector/nullity_mean"] = sum(nullities) / len(nullities)
    else:
        metrics["grit/projector/rank_mean"] = 0.0
        metrics["grit/projector/nullity_mean"] = 0.0
    return metrics


def _preservation_loss_and_metrics(
    result: torch.Tensor | PreservationLossResult,
) -> tuple[torch.Tensor, dict[str, float]]:
    if torch.is_tensor(result):
        return result, {}

    projection = result.projection
    active_mask = projection.active_mask
    violation_mask = projection.violation_mask
    active_count = active_mask.sum().clamp_min(1)
    metrics = {
        "grit/preservation_loss": float(result.loss.detach().float().item()),
        "grit/kl_violation_fraction": float(
            (violation_mask & active_mask).to(torch.float32).sum().div(active_count).item()
        ),
        "grit/kl_raw_mean": float(projection.token_kl.detach().float().mean().item()),
        "grit/kl_projected_mean": float(projection.projected_kl.detach().float().mean().item()),
    }
    return result.loss, metrics


def assemble_grit_update(
    model: nn.Module,
    task_loss: torch.Tensor,
    preservation_loss_fn: PreservationLossFn | None,
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    *,
    config: GritUpdateConfig | None = None,
    alpha: float | None = None,
    lambda_pres: float | None = None,
    use_curvature: bool | None = None,
    parameters: Sequence[NamedParameter] | None = None,
    parameter_filter: ParameterFilter | None = None,
    module_filter: ModuleFilter | None = None,
    missing_projector: str | None = None,
    zero_grad_before_write: bool | None = None,
) -> GritUpdateResult:
    """Assemble and write the final GRIT gradient to ``parameter.grad``.

    ``task_loss`` is the scalar minimization loss for the main RL/objective
    batch. When ``lambda_pres > 0``, ``preservation_loss_fn`` is called inside a
    temporary ``theta_tilde = theta - alpha * projected_task_grad`` context and
    should return either a scalar tensor or ``PreservationLossResult``.
    """

    resolved = _coerce_config(
        config,
        alpha=alpha,
        lambda_pres=lambda_pres,
        use_curvature=use_curvature,
        missing_projector=missing_projector,
        zero_grad_before_write=zero_grad_before_write,
    )
    if resolved.alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {resolved.alpha}")
    if resolved.lambda_pres < 0:
        raise ValueError(f"lambda_pres must be non-negative, got {resolved.lambda_pres}")
    if resolved.missing_projector not in {"identity", "zero"}:
        raise ValueError(
            f"missing_projector must be 'identity' or 'zero', got {resolved.missing_projector!r}"
        )
    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)
    if parameters is None:
        parameters = trainable_named_parameters(model, parameter_filter)

    task_gradients = _autograd_gradient_map(
        task_loss,
        parameters,
        retain_graph=resolved.use_curvature,
    )
    projected_task_gradients = project_vector_with_module_projectors(
        model,
        task_gradients,
        projectors,
        parameters=parameters,
        module_filter=module_filter,
        missing=resolved.missing_projector,
    )

    preservation_metrics: dict[str, float] = {}
    if resolved.lambda_pres == 0.0:
        preservation_loss = task_loss.detach().new_zeros(())
        preservation_gradients = {
            name: _zero_like_parameter(parameter) for name, parameter in parameters
        }
        predictor_info = PredictorStepInfo(
            updated_parameters=0,
            update_norm=0.0,
            max_update_abs=0.0,
        )
    else:
        if preservation_loss_fn is None:
            raise ValueError("preservation_loss_fn is required when lambda_pres > 0")
        with temporary_predictor_step(
            model,
            alpha=resolved.alpha,
            gradients=projected_task_gradients,
            parameter_filter=parameter_filter,
            module_filter=module_filter,
        ) as predictor_info:
            preservation_result = preservation_loss_fn()
            preservation_loss, preservation_metrics = _preservation_loss_and_metrics(preservation_result)
            preservation_gradients = _autograd_gradient_map(
                preservation_loss,
                parameters,
                retain_graph=False,
            )

    curvature_result: CurvatureCorrectionResult | None = None
    if resolved.use_curvature and resolved.lambda_pres > 0.0:
        from grit.curvature import curvature_corrected_preservation_gradients

        curvature_result = curvature_corrected_preservation_gradients(
            model,
            task_loss,
            preservation_gradients,
            projectors,
            alpha=resolved.alpha,
            parameters=parameters,
            module_filter=module_filter,
            missing_projector=resolved.missing_projector,
        )
        preservation_correction = curvature_result.gradients
    else:
        preservation_correction = preservation_gradients

    final_gradients = {
        name: projected_task_gradients[name]
        - preservation_correction[name].to(projected_task_gradients[name]).mul(resolved.lambda_pres)
        for name, _parameter in parameters
    }
    _write_gradients(
        parameters,
        final_gradients,
        zero_grad_before_write=resolved.zero_grad_before_write,
    )

    skipped_hvp = True if curvature_result is None else curvature_result.skipped_hvp
    metrics = {
        "grit/alpha": float(resolved.alpha),
        "grit/lambda_pres": float(resolved.lambda_pres),
        "grit/use_curvature": float(resolved.use_curvature),
        "grit/hvp_skipped": float(skipped_hvp),
        "grit/task_grad_norm": _squared_norm(list(task_gradients.values())) ** 0.5,
        "grit/projected_task_grad_norm": _squared_norm(list(projected_task_gradients.values())) ** 0.5,
        "grit/preservation_grad_norm": _squared_norm(list(preservation_gradients.values())) ** 0.5,
        "grit/preservation_correction_norm": _squared_norm(list(preservation_correction.values())) ** 0.5,
        "grit/final_grad_norm": _squared_norm(list(final_gradients.values())) ** 0.5,
        "grit/predictor_updated_parameters": float(predictor_info.updated_parameters),
        "grit/predictor_update_norm": float(predictor_info.update_norm),
        "grit/predictor_max_update_abs": float(predictor_info.max_update_abs),
    }
    metrics.update(_projector_metrics(projectors))
    metrics.update(preservation_metrics)
    if "grit/preservation_loss" not in metrics:
        metrics["grit/preservation_loss"] = float(preservation_loss.detach().float().item())
    if curvature_result is not None:
        metrics["grit/hvp_norm"] = float(curvature_result.hvp_norm)
        metrics["grit/hvp_projected_vector_norm"] = float(curvature_result.projected_vector_norm)
    else:
        metrics["grit/hvp_norm"] = 0.0
        metrics["grit/hvp_projected_vector_norm"] = 0.0

    return GritUpdateResult(
        final_gradients=final_gradients,
        projected_task_gradients=projected_task_gradients,
        preservation_gradients=preservation_gradients,
        preservation_correction=preservation_correction,
        preservation_loss=preservation_loss.detach(),
        predictor=predictor_info,
        curvature=curvature_result,
        metrics=metrics,
    )
