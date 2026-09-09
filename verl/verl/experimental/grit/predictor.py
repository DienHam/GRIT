"""GRIT Phase 2 predictor helpers for verl actor workers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


GradientMap = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class PredictorStepInfo:
    """Diagnostics for one temporary predictor update."""

    updated_parameters: int
    update_norm: float
    max_update_abs: float

    def metrics(self) -> dict[str, float]:
        return {
            "grit/predictor_updated_parameters": float(self.updated_parameters),
            "grit/predictor_update_norm": float(self.update_norm),
            "grit/predictor_max_update_abs": float(self.max_update_abs),
        }


def _named_trainable_parameters(model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def clone_current_gradients(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone all currently materialized trainable parameter gradients."""

    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in _named_trainable_parameters(model):
        if parameter.grad is not None:
            gradients[name] = parameter.grad.detach().clone()
    return gradients


def _squared_norm(gradients: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for gradient in gradients.values():
        total += float(gradient.detach().float().square().sum().item())
    return total


@contextmanager
def temporary_predictor_step(
    model: nn.Module,
    *,
    learning_rate: float,
    gradients: GradientMap,
) -> Iterator[PredictorStepInfo]:
    """Temporarily apply ``theta_tilde = theta - learning_rate * gradients``.

    The ``gradients`` mapping should contain the post-Phase-1 task gradients:
    projected MLP Linear weight gradients and identity/unprojected gradients for
    trainable parameters outside the projector surface.
    """

    if learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")

    deltas: list[tuple[nn.Parameter, torch.Tensor]] = []
    update_sq = 0.0
    max_update_abs = 0.0
    named_parameters = dict(_named_trainable_parameters(model))

    with torch.no_grad():
        for name, gradient in gradients.items():
            if name not in named_parameters:
                continue
            parameter = named_parameters[name]
            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"gradient for {name} has shape {tuple(gradient.shape)}, "
                    f"expected {tuple(parameter.shape)}"
                )

            delta = gradient.detach().to(device=parameter.device, dtype=parameter.dtype).mul(
                -learning_rate
            )
            parameter.add_(delta)
            deltas.append((parameter, delta))

            delta_float = delta.float()
            update_sq += float(delta_float.square().sum().item())
            max_update_abs = max(max_update_abs, float(delta_float.abs().max().item()))

    info = PredictorStepInfo(
        updated_parameters=len(deltas),
        update_norm=update_sq**0.5,
        max_update_abs=max_update_abs,
    )
    try:
        yield info
    finally:
        with torch.no_grad():
            for parameter, delta in reversed(deltas):
                parameter.sub_(delta)


def write_final_grit_gradients(
    model: nn.Module,
    projected_task_gradients: GradientMap,
    preservation_gradients: GradientMap,
    *,
    lambda_pres: float,
) -> dict[str, float]:
    """Write ``g_projected + lambda_pres * v_preservation`` to ``.grad``."""

    if lambda_pres < 0:
        raise ValueError(f"lambda_pres must be non-negative, got {lambda_pres}")

    final_gradients: dict[str, torch.Tensor] = {}

    for name, parameter in _named_trainable_parameters(model):
        projected = projected_task_gradients.get(name)
        preservation = preservation_gradients.get(name)
        if projected is None and preservation is None:
            continue
        if projected is None:
            projected = torch.zeros_like(preservation)
        if preservation is None:
            preservation = torch.zeros_like(projected)

        final = projected.to(device=parameter.device, dtype=parameter.dtype).add(
            preservation.to(device=parameter.device, dtype=parameter.dtype).mul(lambda_pres)
        )
        parameter.grad = final.detach().clone()
        final_gradients[name] = final.detach().clone()

    return {
        "grit/projected_task_grad_norm": _squared_norm(projected_task_gradients) ** 0.5,
        "grit/preservation_grad_norm": _squared_norm(preservation_gradients) ** 0.5,
        "grit/final_grad_norm": _squared_norm(final_gradients) ** 0.5,
        "grit/lambda_pres": float(lambda_pres),
    }


def combine_projected_task_and_preservation_gradients(
    model: nn.Module,
    projected_task_gradients: GradientMap,
    preservation_gradients: GradientMap | None = None,
    *,
    lambda_pres: float,
) -> dict[str, float]:
    """Backward-compatible wrapper for writing final GRIT gradients.

    New Phase 5 integration should pass ``preservation_gradients`` explicitly
    after zeroing task gradients and backwarding ``L_pres``. If omitted, this
    falls back to the older accumulated-gradient interpretation.
    """

    if preservation_gradients is None:
        inferred_preservation_gradients: dict[str, torch.Tensor] = {}
        for name, parameter in _named_trainable_parameters(model):
            projected = projected_task_gradients.get(name)
            if parameter.grad is None:
                continue
            if projected is None:
                inferred_preservation_gradients[name] = parameter.grad.detach().clone()
            else:
                projected = projected.to(device=parameter.grad.device, dtype=parameter.grad.dtype)
                inferred_preservation_gradients[name] = parameter.grad.detach().sub(projected).clone()
        preservation_gradients = inferred_preservation_gradients

    return write_final_grit_gradients(
        model,
        projected_task_gradients,
        preservation_gradients,
        lambda_pres=lambda_pres,
    )
