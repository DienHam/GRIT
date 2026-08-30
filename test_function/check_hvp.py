#!/usr/bin/env python3
"""Toy sanity check for GRIT Phase 4 curvature HVP."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.curvature import curvature_corrected_preservation_gradients


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(3, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def task_loss(model: ToyModel, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return 0.5 * (logits - target).square().mean()


def finite_difference_hvp(
    model: ToyModel,
    x: torch.Tensor,
    target: torch.Tensor,
    weight_direction: torch.Tensor,
    bias_direction: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    with torch.no_grad():
        model.mlp.weight.add_(weight_direction, alpha=eps)
        model.mlp.bias.add_(bias_direction, alpha=eps)
    loss_plus = task_loss(model, x, target)
    grad_plus = torch.autograd.grad(loss_plus, (model.mlp.weight, model.mlp.bias))

    with torch.no_grad():
        model.mlp.weight.add_(weight_direction, alpha=-2.0 * eps)
        model.mlp.bias.add_(bias_direction, alpha=-2.0 * eps)
    loss_minus = task_loss(model, x, target)
    grad_minus = torch.autograd.grad(loss_minus, (model.mlp.weight, model.mlp.bias))

    with torch.no_grad():
        model.mlp.weight.add_(weight_direction, alpha=eps)
        model.mlp.bias.add_(bias_direction, alpha=eps)

    return tuple((plus - minus) / (2.0 * eps) for plus, minus in zip(grad_plus, grad_minus, strict=True))


def main() -> None:
    torch.manual_seed(41)
    torch.set_default_dtype(torch.float64)

    model = ToyModel()
    x = torch.randn(7, 3)
    target = torch.randn(7, 2)
    loss = task_loss(model, x, target)
    parameters = [("mlp.weight", model.mlp.weight), ("mlp.bias", model.mlp.bias)]

    projector = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    v = torch.randn_like(model.mlp.weight)
    bias_v = torch.randn_like(model.mlp.bias)
    preservation_grads = {"mlp.weight": v, "mlp.bias": bias_v}

    result = curvature_corrected_preservation_gradients(
        model,
        loss,
        preservation_grads,
        {"mlp": projector},
        alpha=0.25,
        parameters=parameters,
    )
    manual_direction = v.matmul(projector)
    manual_bias_direction = bias_v
    finite_difference = finite_difference_hvp(
        model,
        x,
        target,
        manual_direction,
        manual_bias_direction,
        eps=1e-5,
    )

    torch.testing.assert_close(result.projected_vector["mlp.weight"], manual_direction, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.projected_vector["mlp.bias"], manual_bias_direction, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.hvp["mlp.weight"], finite_difference[0], atol=5e-9, rtol=5e-7)
    torch.testing.assert_close(result.hvp["mlp.bias"], finite_difference[1], atol=5e-9, rtol=5e-7)
    torch.testing.assert_close(
        result.gradients["mlp.weight"],
        v + 0.25 * result.hvp["mlp.weight"],
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        result.gradients["mlp.bias"],
        bias_v + 0.25 * result.hvp["mlp.bias"],
        atol=1e-12,
        rtol=1e-12,
    )
    assert not result.skipped_hvp

    zero_result = curvature_corrected_preservation_gradients(
        model,
        loss,
        {"mlp.weight": torch.zeros_like(v), "mlp.bias": torch.zeros_like(bias_v)},
        {"mlp": projector},
        alpha=0.25,
        parameters=parameters,
    )
    assert zero_result.skipped_hvp
    assert zero_result.projected_vector_norm == 0.0
    assert zero_result.hvp_norm == 0.0
    torch.testing.assert_close(zero_result.gradients["mlp.weight"], torch.zeros_like(v), atol=0.0, rtol=0.0)
    torch.testing.assert_close(zero_result.gradients["mlp.bias"], torch.zeros_like(bias_v), atol=0.0, rtol=0.0)

    print(f"projected_vector_norm={result.projected_vector_norm:.8f}")
    print(f"hvp_norm={result.hvp_norm:.8f}")
    finite_difference_error = (
        (result.hvp["mlp.weight"] - finite_difference[0]).float().square().sum()
        + (result.hvp["mlp.bias"] - finite_difference[1]).float().square().sum()
    ).sqrt()
    print(f"finite_difference_error={finite_difference_error.item():.8e}")
    print(f"corrected_grad_norm={result.gradients['mlp.weight'].norm().item():.8f}")
    print("missing_projector_identity=True")
    print(f"zero_vector_skipped={zero_result.skipped_hvp}")


if __name__ == "__main__":
    main()
