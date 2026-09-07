#!/usr/bin/env python3
"""Toy sanity check for GRIT Phase 5 total update assembly."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.preservation_loss import preservation_kl_loss
from grit.update import GritUpdateConfig, assemble_grit_update


class ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(3, 5, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def clone_named_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def assert_parameters_restored(model: nn.Module, reference: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), reference[name], atol=1e-8, rtol=1e-8)


def make_losses(
    model: ToyPolicy,
    base_model: ToyPolicy,
    task_x: torch.Tensor,
    task_target: torch.Tensor,
    pres_x: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
):
    task_logits = model(task_x)
    task_loss = 0.5 * (task_logits - task_target).square().mean()

    def preservation_loss_fn():
        policy_logits = model(pres_x)
        with torch.no_grad():
            base_logits = base_model(pres_x)
        return preservation_kl_loss(
            policy_logits,
            base_logits,
            epsilon_pres=0.001,
            response_mask=response_mask,
            selected_token_ids=responses,
            top_k=2,
            default_probability=1e-6,
        )

    return task_loss, preservation_loss_fn


def run_update(
    model: ToyPolicy,
    base_model: ToyPolicy,
    projectors: dict[str, torch.Tensor],
    *,
    lambda_pres: float,
    use_curvature: bool,
):
    task_x = torch.randn(4, 2, 3)
    task_target = torch.randn(4, 2, 5)
    pres_x = torch.randn(4, 2, 3)
    responses = torch.tensor([[0, 1], [2, 3], [4, 0], [1, 2]])
    response_mask = torch.tensor(
        [[True, True], [True, False], [True, True], [False, True]]
    )
    task_loss, preservation_loss_fn = make_losses(
        model,
        base_model,
        task_x,
        task_target,
        pres_x,
        responses,
        response_mask,
    )
    return assemble_grit_update(
        model,
        task_loss,
        preservation_loss_fn,
        projectors,
        config=GritUpdateConfig(
            alpha=0.2,
            lambda_pres=lambda_pres,
            use_curvature=use_curvature,
        ),
    )


def main() -> None:
    torch.manual_seed(53)
    torch.set_default_dtype(torch.float64)

    model = ToyPolicy()
    base_model = copy.deepcopy(model)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        model.mlp.weight.add_(0.3 * torch.randn_like(model.mlp.weight))
        model.mlp.bias.add_(0.2 * torch.randn_like(model.mlp.bias))

    baseline_parameters = clone_named_parameters(model)
    projector = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    projectors = {"mlp": projector}

    task_x = torch.randn(4, 2, 3)
    task_target = torch.randn(4, 2, 5)
    task_loss = 0.5 * (model(task_x) - task_target).square().mean()
    projection_only = assemble_grit_update(
        model,
        task_loss,
        None,
        projectors,
        config=GritUpdateConfig(
            alpha=0.2,
            lambda_pres=0.0,
            use_curvature=False,
        ),
    )
    torch.testing.assert_close(
        projection_only.final_gradients["mlp.weight"],
        projection_only.projected_task_gradients["mlp.weight"],
        atol=1e-12,
        rtol=1e-12,
    )
    assert projection_only.curvature is None
    assert projection_only.metrics["grit/hvp_skipped"] == 1.0
    assert_parameters_restored(model, baseline_parameters)

    first_order = run_update(
        model,
        base_model,
        projectors,
        lambda_pres=0.7,
        use_curvature=False,
    )
    manual_first_order = (
        first_order.projected_task_gradients["mlp.weight"]
        - 0.7 * first_order.preservation_gradients["mlp.weight"]
    )
    torch.testing.assert_close(
        first_order.final_gradients["mlp.weight"],
        manual_first_order,
        atol=1e-12,
        rtol=1e-12,
    )
    assert first_order.curvature is None
    assert first_order.metrics["grit/preservation_loss"] > 0.0
    assert first_order.metrics["grit/kl_violation_fraction"] > 0.0
    assert first_order.metrics["grit/projected_module_count"] == 1.0
    assert_parameters_restored(model, baseline_parameters)

    curvature = run_update(
        model,
        base_model,
        projectors,
        lambda_pres=0.7,
        use_curvature=True,
    )
    assert curvature.curvature is not None
    assert not curvature.curvature.skipped_hvp
    assert curvature.metrics["grit/hvp_skipped"] == 0.0
    assert curvature.metrics["grit/projected_vector_norm"] > 0.0
    assert curvature.metrics["grit/hvp_norm"] > 0.0
    assert curvature.metrics["grit/corrected_preservation_grad_norm"] > 0.0
    assert "grit/projector/mlp/rank" in curvature.metrics
    assert "grit/projector/mlp/nullity" in curvature.metrics
    assert_parameters_restored(model, baseline_parameters)

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, curvature.final_gradients[name])

    print("gradient_projection_only=True")
    print("first_order_grit=True")
    print("curvature_toggle=True")
    print(f"projection_rank={curvature.metrics['grit/projector/mlp/rank']:.0f}")
    print(f"projection_nullity={curvature.metrics['grit/projector/mlp/nullity']:.0f}")
    print(f"projected_module_count={curvature.metrics['grit/projected_module_count']:.0f}")
    print(f"kl_violation_fraction={first_order.metrics['grit/kl_violation_fraction']:.6f}")
    print(f"preservation_loss={first_order.metrics['grit/preservation_loss']:.8f}")
    print(f"hvp_skipped_first_order={bool(first_order.metrics['grit/hvp_skipped'])}")
    print(f"hvp_skipped_curvature={bool(curvature.metrics['grit/hvp_skipped'])}")
    print(f"projected_vector_norm={curvature.metrics['grit/projected_vector_norm']:.8f}")
    print(f"hvp_norm={curvature.metrics['grit/hvp_norm']:.8f}")
    print(
        "corrected_preservation_grad_norm="
        f"{curvature.metrics['grit/corrected_preservation_grad_norm']:.8f}"
    )
    print(f"final_grad_norm={curvature.metrics['grit/final_grad_norm']:.8f}")


if __name__ == "__main__":
    main()
