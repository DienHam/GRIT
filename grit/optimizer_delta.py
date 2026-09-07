"""AdamW-style delta helpers for split GRIT update experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from grit.curvature import project_vector_with_module_projectors
from grit.projection import ProjectorBuildResult, default_module_filter


NamedParameter = tuple[str, nn.Parameter]
ModuleFilter = object


def _squared_norm(tensors) -> float:
    total = 0.0
    for tensor in tensors:
        norm = torch.linalg.vector_norm(tensor.detach())
        total += float(norm.item()) ** 2
    return total


@dataclass
class AdamWDeltaPreconditioner:
    """Compute the parameter delta that AdamW would apply for a gradient map.

    This object intentionally does not mutate parameters. It only maintains
    AdamW moments and returns additive deltas, so callers can combine multiple
    optimizer-shaped directions before applying one manual weight update.
    """

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        self.state: dict[str, dict[str, torch.Tensor | int]] = {}

    def state_dict(self) -> dict:
        packed: dict[str, dict[str, torch.Tensor | int]] = {}
        for name, state in self.state.items():
            packed[name] = {
                "step": int(state["step"]),
                "exp_avg": state["exp_avg"].detach().cpu(),
                "exp_avg_sq": state["exp_avg_sq"].detach().cpu(),
            }
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": packed,
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        self.lr = float(state_dict.get("lr", self.lr))
        self.betas = tuple(state_dict.get("betas", self.betas))  # type: ignore[assignment]
        self.eps = float(state_dict.get("eps", self.eps))
        self.weight_decay = float(state_dict.get("weight_decay", self.weight_decay))
        self.state = {}
        for name, state in state_dict.get("state", {}).items():
            self.state[name] = {
                "step": int(state["step"]),
                "exp_avg": state["exp_avg"].detach().clone(),
                "exp_avg_sq": state["exp_avg_sq"].detach().clone(),
            }

    @torch.no_grad()
    def deltas(
        self,
        parameters: Sequence[NamedParameter],
        gradients: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        beta1, beta2 = self.betas
        updates: dict[str, torch.Tensor] = {}
        for name, parameter in parameters:
            grad = gradients[name].detach().to(device=parameter.device, dtype=torch.float32)
            state = self.state.get(name)
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(parameter.detach(), dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(parameter.detach(), dtype=torch.float32),
                }
                self.state[name] = state

            exp_avg = state["exp_avg"].to(device=parameter.device, dtype=torch.float32)
            exp_avg_sq = state["exp_avg_sq"].to(device=parameter.device, dtype=torch.float32)
            step = int(state["step"]) + 1

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(self.eps)
            adam_direction = exp_avg.div(bias_correction1).div(denom)
            delta = adam_direction.mul(-self.lr)
            if self.weight_decay != 0.0:
                delta = delta.add(parameter.detach().to(torch.float32), alpha=-self.lr * self.weight_decay)

            updates[name] = delta.to(device=parameter.device, dtype=parameter.dtype)
            state["step"] = step
            state["exp_avg"] = exp_avg.detach()
            state["exp_avg_sq"] = exp_avg_sq.detach()
        return updates


@torch.no_grad()
def apply_split_adamw_delta_update(
    *,
    model: nn.Module,
    parameters: Sequence[NamedParameter],
    task_gradients: Mapping[str, torch.Tensor],
    preservation_gradients: Mapping[str, torch.Tensor],
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    task_preconditioner: AdamWDeltaPreconditioner,
    preservation_preconditioner: AdamWDeltaPreconditioner,
    alpha: float,
    lambda_pres: float,
    module_filter=None,
    missing_projector: str = "identity",
) -> dict[str, float]:
    """Apply ``alpha * AdamW(g_task)P - lambda * AdamW(g_pres)`` to weights."""

    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)

    task_deltas = task_preconditioner.deltas(parameters, task_gradients)
    projected_task_deltas = project_vector_with_module_projectors(
        model,
        task_deltas,
        projectors,
        parameters=parameters,
        module_filter=module_filter,
        missing=missing_projector,
    )
    preservation_deltas = preservation_preconditioner.deltas(parameters, preservation_gradients)

    final_deltas: dict[str, torch.Tensor] = {}
    for name, parameter in parameters:
        delta = projected_task_deltas[name].to(device=parameter.device, dtype=parameter.dtype).mul(alpha)
        delta = delta.add(
            preservation_deltas[name].to(device=parameter.device, dtype=parameter.dtype),
            alpha=-lambda_pres,
        )
        parameter.add_(delta)
        final_deltas[name] = delta.detach()

    task_before_norm = _squared_norm(task_deltas.values()) ** 0.5
    task_after_norm = _squared_norm(projected_task_deltas.values()) ** 0.5
    task_removed = max(task_before_norm**2 - task_after_norm**2, 0.0) ** 0.5
    final_norm = _squared_norm(final_deltas.values()) ** 0.5
    return {
        "split_adamw_task_delta_norm": task_before_norm,
        "split_adamw_task_projected_delta_norm": task_after_norm,
        "split_adamw_task_removed_delta_norm": task_removed,
        "split_adamw_task_removed_fraction": task_removed / task_before_norm if task_before_norm > 0 else 0.0,
        "split_adamw_preservation_delta_norm": _squared_norm(preservation_deltas.values()) ** 0.5,
        "split_adamw_final_delta_norm": final_norm,
    }
