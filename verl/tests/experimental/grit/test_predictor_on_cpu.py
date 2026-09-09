import sys
from pathlib import Path

import torch
from torch import nn

VERL_ROOT = Path(__file__).resolve().parents[3]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))
if "verl" in sys.modules and not hasattr(sys.modules["verl"], "DataProto"):
    del sys.modules["verl"]

from verl.experimental.grit.predictor import (
    clone_current_gradients,
    combine_projected_task_and_preservation_gradients,
    temporary_predictor_step,
    write_final_grit_gradients,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)


def test_temporary_predictor_restores_and_combines_gradients():
    torch.manual_seed(3)
    model = ToyModel()
    x = torch.randn(5, 3)
    target = torch.randn(5, 2)

    task_loss = model(x).square().mean()
    task_loss.backward()
    projected_task_gradients = clone_current_gradients(model)
    original_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    with temporary_predictor_step(
        model, learning_rate=0.2, gradients=projected_task_gradients
    ) as predictor_info:
        assert predictor_info.updated_parameters == len(projected_task_gradients)
        assert predictor_info.update_norm > 0.0
        predictor_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        assert any(
            not torch.allclose(predictor_parameters[name], original_parameters[name])
            for name in original_parameters.keys()
        )
        preservation_loss = (model(x) - target).square().mean()
        preservation_loss.backward()
        grad_after_preservation = clone_current_gradients(model)

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), original_parameters[name])

    lambda_pres = 0.7
    metrics = combine_projected_task_and_preservation_gradients(
        model,
        projected_task_gradients,
        lambda_pres=lambda_pres,
    )

    for name, parameter in model.named_parameters():
        preservation_gradient = grad_after_preservation[name] - projected_task_gradients[name]
        expected = projected_task_gradients[name] + lambda_pres * preservation_gradient
        torch.testing.assert_close(parameter.grad, expected)

    assert metrics["grit/projected_task_grad_norm"] > 0.0
    assert metrics["grit/preservation_grad_norm"] > 0.0
    assert metrics["grit/final_grad_norm"] > 0.0


def test_phase5_writes_explicit_final_gradient_after_separate_preservation_backward():
    torch.manual_seed(5)
    model = ToyModel()
    x = torch.randn(5, 3)
    target = torch.randn(5, 2)

    task_loss = model(x).square().mean()
    task_loss.backward()
    projected_task_gradients = clone_current_gradients(model)
    original_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    model.zero_grad(set_to_none=True)
    with temporary_predictor_step(
        model, learning_rate=0.2, gradients=projected_task_gradients
    ):
        preservation_loss = (model(x) - target).square().mean()
        preservation_loss.backward()
        preservation_gradients = clone_current_gradients(model)

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), original_parameters[name])
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, preservation_gradients[name])

    lambda_pres = 0.7
    metrics = write_final_grit_gradients(
        model,
        projected_task_gradients,
        preservation_gradients,
        lambda_pres=lambda_pres,
    )

    for name, parameter in model.named_parameters():
        expected = projected_task_gradients[name] + lambda_pres * preservation_gradients[name]
        torch.testing.assert_close(parameter.grad, expected)

    assert metrics["grit/projected_task_grad_norm"] > 0.0
    assert metrics["grit/preservation_grad_norm"] > 0.0
    assert metrics["grit/final_grad_norm"] > 0.0
