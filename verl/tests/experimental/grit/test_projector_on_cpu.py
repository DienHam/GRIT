from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

try:
    from verl.experimental.grit.projector import (
        attach_projectors_to_mlp_linears,
        load_projectors,
        project_actor_mlp_gradients,
    )
except ModuleNotFoundError:
    import importlib.util
    import sys

    projector_path = Path(__file__).resolve().parents[3] / "verl" / "experimental" / "grit" / "projector.py"
    spec = importlib.util.spec_from_file_location("grit_projector", projector_path)
    projector_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = projector_module
    spec.loader.exec_module(projector_module)
    attach_projectors_to_mlp_linears = projector_module.attach_projectors_to_mlp_linears
    load_projectors = projector_module.load_projectors
    project_actor_mlp_gradients = projector_module.project_actor_mlp_gradients


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(4, 3, bias=False)
        self.other = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.mlp(x).square().mean() + self.other(x).square().mean()


def test_project_actor_mlp_gradients_from_attached_projector():
    basis = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [2.0, -1.0, 0.0, 0.0],
        ]
    )
    covariance = basis.T @ basis
    projector = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0]))

    with TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "projectors.pt"
        torch.save({"projectors": {"mlp": projector}}, artifact_path)
        projectors = load_projectors(artifact_path)

    actor = ToyActor()
    attach_summary = attach_projectors_to_mlp_linears(actor, projectors)
    assert attach_summary.attached == ("mlp",)
    assert attach_summary.missing == ()
    assert attach_summary.shape_mismatch == ()

    loss = actor(torch.randn(8, 4))
    loss.backward()
    mlp_before = actor.mlp.weight.grad.detach().clone()
    other_before = actor.other.weight.grad.detach().clone()

    metrics = project_actor_mlp_gradients(actor)

    assert metrics["grit/projected_module_count"] == 1.0
    assert metrics["grit/task_grad_norm_after_projection"] <= metrics["grit/task_grad_norm_before_projection"]
    assert actor.mlp.weight.grad.matmul(covariance).norm().item() < 1e-6
    torch.testing.assert_close(actor.mlp.weight.grad, mlp_before @ projector)
    torch.testing.assert_close(actor.other.weight.grad, other_before)
