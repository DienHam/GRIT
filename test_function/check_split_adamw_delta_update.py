#!/usr/bin/env python3
"""Check split AdamW-delta GRIT update semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.optimizer_delta import AdamWDeltaPreconditioner, apply_split_adamw_delta_update


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

    task_adamw = AdamWDeltaPreconditioner(lr=0.01, eps=1e-8, weight_decay=0.0)
    pres_adamw = AdamWDeltaPreconditioner(lr=0.01, eps=1e-8, weight_decay=0.0)
    metrics = apply_split_adamw_delta_update(
        model=model,
        parameters=parameters,
        task_gradients=task_gradients,
        preservation_gradients=preservation_gradients,
        projectors={"mlp": projector},
        task_preconditioner=task_adamw,
        preservation_preconditioner=pres_adamw,
        alpha=1.0,
        lambda_pres=1.0,
        module_filter=lambda _name, module: isinstance(module, nn.Linear),
    )

    task_delta = torch.full_like(before, -0.01).copysign(-task_gradients["mlp.weight"])
    pres_delta = torch.full_like(before, -0.01).copysign(-preservation_gradients["mlp.weight"])
    pres_delta = torch.where(preservation_gradients["mlp.weight"].eq(0), torch.zeros_like(pres_delta), pres_delta)
    expected = before + task_delta.matmul(projector) - pres_delta

    torch.testing.assert_close(model.mlp.weight, expected)
    assert metrics["split_adamw_task_removed_fraction"] > 0.0
    assert metrics["split_adamw_preservation_delta_norm"] > 0.0
    assert model.mlp.weight[:, 1].sub(before[:, 1]).abs().sum().item() > 0.0

    print("split_adamw_delta_update=True")
    print(f"task_removed_fraction={metrics['split_adamw_task_removed_fraction']:.6f}")
    print(f"final_delta_norm={metrics['split_adamw_final_delta_norm']:.6f}")


if __name__ == "__main__":
    main()
