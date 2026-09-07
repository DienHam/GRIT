"""GRIT Phase 1 projector helpers for verl actor workers.

This module implements only the optimizer-facing gradient projection:

    grad_W <- grad_W @ P

It deliberately does not implement NSPO's periodic weight repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class ProjectorAttachSummary:
    """Result from attaching projectors to actor modules."""

    attached: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[str, ...]


def load_projectors(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    projector_key: str = "projectors",
) -> dict[str, torch.Tensor]:
    """Load projectors from a torch artifact.

    The normal artifact is produced by ``scripts/build_projectors.py`` and has
    shape ``{"projectors": {module_name: P}, "metadata": ...}``. A raw
    ``{module_name: P}`` mapping is also accepted for tests.
    """

    payload = torch.load(Path(path), map_location=map_location)
    if isinstance(payload, Mapping) and projector_key in payload:
        payload = payload[projector_key]
    if not isinstance(payload, Mapping):
        raise TypeError(f"GRIT projector artifact must be a mapping, got {type(payload).__name__}")

    projectors: dict[str, torch.Tensor] = {}
    for name, projector in payload.items():
        if not isinstance(name, str):
            raise TypeError(f"GRIT projector name must be str, got {type(name).__name__}")
        if not torch.is_tensor(projector):
            raise TypeError(f"GRIT projector for {name!r} must be a Tensor, got {type(projector).__name__}")
        if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
            raise ValueError(f"GRIT projector for {name!r} must be square, got shape {tuple(projector.shape)}")
        projectors[name] = projector.detach().cpu()
    return projectors


def attach_projectors_to_mlp_linears(
    model: nn.Module,
    projectors: Mapping[str, torch.Tensor],
    *,
    module_pattern: str = "mlp",
    attribute_name: str = "proj_w",
    strict: bool = True,
) -> ProjectorAttachSummary:
    """Attach projector tensors to matching MLP Linear modules."""

    modules = dict(model.named_modules())
    target_names = [
        name for name, module in modules.items() if isinstance(module, nn.Linear) and module_pattern in name
    ]
    attached: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[str] = []

    for name in target_names:
        module = modules[name]
        if name not in projectors:
            missing.append(name)
            continue
        projector = projectors[name].detach().cpu()
        expected_shape = (module.in_features, module.in_features)
        if tuple(projector.shape) != expected_shape:
            shape_mismatch.append(f"{name}: got {tuple(projector.shape)}, expected {expected_shape}")
            continue
        setattr(module, attribute_name, projector)
        attached.append(name)

    target_name_set = set(target_names)
    unexpected = tuple(sorted(name for name in projectors.keys() if name not in target_name_set))
    summary = ProjectorAttachSummary(
        attached=tuple(attached),
        missing=tuple(missing),
        unexpected=unexpected,
        shape_mismatch=tuple(shape_mismatch),
    )
    if strict and (summary.missing or summary.shape_mismatch):
        raise ValueError(
            "failed to attach all GRIT projectors: "
            f"missing={list(summary.missing)}, shape_mismatch={list(summary.shape_mismatch)}"
        )
    return summary


def project_actor_mlp_gradients(
    model: nn.Module,
    *,
    module_pattern: str = "mlp",
    attribute_name: str = "proj_w",
    require_projected: bool = True,
) -> dict[str, float]:
    """Project attached actor MLP Linear gradients and return logging metrics."""

    before_sq = 0.0
    after_sq = 0.0
    projected_count = 0
    attached_count = 0
    grad_missing_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or module_pattern not in name:
            continue
        projector = getattr(module, attribute_name, None)
        if projector is None:
            continue
        attached_count += 1
        if module.weight.grad is None:
            grad_missing_count += 1
            continue
        if not torch.is_tensor(projector):
            raise TypeError(f"attached GRIT projector on {name!r} must be a Tensor")

        grad = module.weight.grad
        projector = projector.to(device=grad.device, dtype=grad.dtype)
        expected_shape = (grad.shape[1], grad.shape[1])
        if tuple(projector.shape) != expected_shape:
            raise ValueError(
                f"attached GRIT projector on {name!r} has shape {tuple(projector.shape)}, "
                f"expected {expected_shape}"
            )

        before_sq += float(grad.detach().float().square().sum().item())
        grad.copy_(grad.matmul(projector))
        after_sq += float(grad.detach().float().square().sum().item())
        projected_count += 1

    if require_projected and attached_count > 0 and projected_count == 0:
        raise RuntimeError(
            "GRIT projectors are attached, but no MLP Linear weight gradients were projected. "
            "Check FSDP use_orig_params / parameter flattening and that projection runs after task backward."
        )

    return {
        "grit/task_grad_norm": before_sq**0.5,
        "grit/task_grad_norm_before_projection": before_sq**0.5,
        "grit/task_grad_norm_after_projection": after_sq**0.5,
        "grit/projected_module_count": float(projected_count),
        "grit/attached_projector_count": float(attached_count),
        "grit/projector_grad_missing_count": float(grad_missing_count),
    }
