#!/usr/bin/env python3
"""Check one-learning-rate GRIT update semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.optimizer_delta import (
    AdamWDirectionPreconditioner,
    adamw_task_directions,
    apply_grit_update,
)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(4, 2, bias=False)


def main() -> None:
    torch.manual_seed(23)
    model = ToyModel()
    before = model.mlp.weight.detach().clone()
    parameters = list(model.named_parameters())

    projector = torch.diag(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    task_gradients = {
        "mlp.weight": torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, -2.0, -3.0, -4.0],
            ]
        )
    }
    preservation_gradients = {
        "mlp.weight": torch.tensor(
            [
                [0.0, 5.0, 0.0, 6.0],
                [0.0, -5.0, 0.0, -6.0],
            ]
        )
    }

    task_adamw = AdamWDirectionPreconditioner(eps=1e-8, weight_decay=0.0)
    metrics = apply_grit_update(
        model=model,
        parameters=parameters,
        task_gradients=task_gradients,
        preservation_gradients=preservation_gradients,
        projectors={"mlp": projector},
        task_preconditioner=task_adamw,
        learning_rate=0.01,
        lambda_pres=1.0,
        module_filter=lambda _name, module: isinstance(module, nn.Linear),
    )

    task_direction = -torch.sign(task_gradients["mlp.weight"])
    expected = before + 0.01 * (
        task_direction.matmul(projector) - preservation_gradients["mlp.weight"]
    )

    torch.testing.assert_close(model.mlp.weight, expected)
    assert metrics["adamw_task_removed_fraction"] > 0.0
    assert metrics["raw_preservation_direction_norm"] > 0.0
    assert model.mlp.weight[:, 1].sub(before[:, 1]).abs().sum().item() > 0.0
    pres_only_step = model.mlp.weight[:, 1].sub(before[:, 1])
    pres_only_grad = preservation_gradients["mlp.weight"][:, 1]
    assert torch.dot(pres_only_step, pres_only_grad).item() < 0.0

    print("single_lr_grit_update=True")
    print(f"task_removed_fraction={metrics['adamw_task_removed_fraction']:.6f}")
    print(f"final_delta_norm={metrics['grit_final_delta_norm']:.6f}")

    model = ToyModel()
    parameters = list(model.named_parameters())
    task_adamw = AdamWDirectionPreconditioner(eps=1e-8, weight_decay=0.0)
    task_directions, projected_task_directions, _task_metrics = adamw_task_directions(
        model=model,
        parameters=parameters,
        task_gradients=task_gradients,
        projectors={"mlp": projector},
        task_preconditioner=task_adamw,
        module_filter=lambda _name, module: isinstance(module, nn.Linear),
    )
    assert task_adamw.state["mlp.weight"]["step"] == 1
    apply_grit_update(
        model=model,
        parameters=parameters,
        task_gradients=task_gradients,
        preservation_gradients=preservation_gradients,
        projectors={"mlp": projector},
        task_preconditioner=task_adamw,
        learning_rate=0.01,
        lambda_pres=1.0,
        task_directions=task_directions,
        projected_task_directions=projected_task_directions,
        module_filter=lambda _name, module: isinstance(module, nn.Linear),
    )
    assert task_adamw.state["mlp.weight"]["step"] == 1
    print("precomputed_task_direction_reused=True")


if __name__ == "__main__":
    main()
