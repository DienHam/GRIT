"""Null-space projector utilities for GRIT Phase 1.

The core update implemented here is the NSPO-theory form:

    grad_W <- grad_W @ P

for protected Linear weights W with shape [out_features, in_features].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn


ModuleFilter = Callable[[str, nn.Module], bool]


@dataclass
class ProjectorBuildResult:
    """A projector and its spectrum metadata for one module."""

    projector: torch.Tensor
    eigenvalues: torch.Tensor
    threshold_value: float
    nullity: int
    rank: int


def default_module_filter(name: str, module: nn.Module, pattern: str = "mlp") -> bool:
    """Select protected modules.

    The default follows NSPO's public code: protect Linear layers whose module
    path contains "mlp".
    """

    return isinstance(module, nn.Linear) and pattern in name


def collect_activation_covariances(
    model: nn.Module,
    input_batches: Iterable[dict[str, torch.Tensor]],
    *,
    module_filter: ModuleFilter | None = None,
    device: torch.device | str | None = None,
    progress_desc: str | None = None,
    progress_total: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collect non-central activation covariances X^T X per protected Linear.

    Forward hooks capture each Linear layer's input activations. For a layer with
    input dimension d, this accumulates a [d, d] matrix over all tokens/samples.
    """

    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    covariances: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(module_name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if not inputs:
                return
            hidden = inputs[0].detach()
            if hidden.numel() == 0:
                return
            hidden = hidden.reshape(-1, hidden.shape[-1]).to(device=device, dtype=torch.float32)
            cov = hidden.transpose(0, 1).matmul(hidden)
            if module_name in covariances:
                covariances[module_name] = covariances[module_name].to(device) + cov
            else:
                covariances[module_name] = cov

        return hook

    for name, module in model.named_modules():
        if module_filter(name, module):
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            batches = input_batches
            if progress_desc is not None:
                from tqdm.auto import tqdm

                batches = tqdm(input_batches, total=progress_total, desc=progress_desc)
            for batch in batches:
                batch_on_device = {
                    key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
                }
                model(**batch_on_device)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {name: cov.cpu() for name, cov in covariances.items()}


def build_projectors_from_covariances(
    covariances: dict[str, torch.Tensor],
    *,
    relative_threshold: float = 5e-4,
    absolute_threshold: float | None = None,
    progress_desc: str | None = None,
) -> dict[str, ProjectorBuildResult]:
    """Build `P = U_null U_null^T` from activation covariance matrices.

    A direction is treated as null if its eigenvalue is below
    `relative_threshold * max_eigenvalue`, unless `absolute_threshold` is given.
    """

    results: dict[str, ProjectorBuildResult] = {}
    items = covariances.items()
    if progress_desc is not None:
        from tqdm.auto import tqdm

        items = tqdm(list(items), desc=progress_desc)
    for name, covariance in items:
        matrix = covariance.to(dtype=torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        max_eval = torch.clamp(eigenvalues.max(), min=0.0)
        if absolute_threshold is None:
            threshold_value = float(relative_threshold * max_eval.item())
        else:
            threshold_value = float(absolute_threshold)

        if max_eval.item() == 0.0 and absolute_threshold is None:
            null_mask = torch.ones_like(eigenvalues, dtype=torch.bool)
        else:
            null_mask = eigenvalues < threshold_value
        null_vectors = eigenvectors[:, null_mask]
        if null_vectors.numel() == 0:
            projector = torch.zeros(
                (matrix.shape[0], matrix.shape[0]),
                dtype=torch.float32,
                device=matrix.device,
            )
        else:
            projector = null_vectors.matmul(null_vectors.transpose(0, 1)).contiguous()

        nullity = int(null_mask.sum().item())
        results[name] = ProjectorBuildResult(
            projector=projector.cpu(),
            eigenvalues=eigenvalues.cpu(),
            threshold_value=threshold_value,
            nullity=nullity,
            rank=matrix.shape[0] - nullity,
        )

    return results


def apply_gradient_projection(
    model: nn.Module,
    projectors: dict[str, torch.Tensor | ProjectorBuildResult],
    *,
    module_filter: ModuleFilter | None = None,
) -> dict[str, float]:
    """Apply `grad_W <- grad_W @ P` to protected Linear weights.

    Returns per-layer before/after gradient norms for logging.
    """

    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)

    metrics: dict[str, float] = {}
    for name, module in model.named_modules():
        if not module_filter(name, module):
            continue
        if name not in projectors:
            continue
        if not isinstance(module, nn.Linear) or module.weight.grad is None:
            continue

        raw_projector = projectors[name]
        projector = raw_projector.projector if isinstance(raw_projector, ProjectorBuildResult) else raw_projector
        projector = projector.to(device=module.weight.grad.device, dtype=module.weight.grad.dtype)

        before = module.weight.grad.detach().float().norm()
        module.weight.grad.copy_(module.weight.grad.matmul(projector))
        after = module.weight.grad.detach().float().norm()

        metrics[f"{name}.grad_norm_before"] = float(before.item())
        metrics[f"{name}.grad_norm_after"] = float(after.item())

    return metrics


def projector_diagnostics(projector: torch.Tensor) -> dict[str, float]:
    """Return symmetry/idempotence diagnostics for an orthogonal projector."""

    p = projector.float()
    symmetry_error = (p - p.transpose(0, 1)).norm()
    idempotence_error = (p.matmul(p) - p).norm()
    spectral_norm = torch.linalg.matrix_norm(p, ord=2)
    return {
        "symmetry_error": float(symmetry_error.item()),
        "idempotence_error": float(idempotence_error.item()),
        "spectral_norm": float(spectral_norm.item()),
    }
