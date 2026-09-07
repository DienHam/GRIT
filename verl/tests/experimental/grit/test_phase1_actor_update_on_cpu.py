import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

VERL_ROOT = Path(__file__).resolve().parents[3]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))
if "verl" in sys.modules and not hasattr(sys.modules["verl"], "DataProto"):
    del sys.modules["verl"]

from verl import DataProto
from verl.experimental.grit.projector import attach_projectors_to_mlp_linears
from verl.workers.actor.dp_actor import DataParallelPPOActor


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size=11, hidden_size=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        hidden = self.embed(input_ids)
        hidden = torch.tanh(self.mlp(hidden))
        logits = self.lm_head(hidden)
        return type("Output", (), {"logits": logits})()


def test_phase1_projection_runs_inside_dp_actor_update(monkeypatch):
    torch.manual_seed(7)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_id", lambda: "cpu")

    model = ToyCausalLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    projector = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    attach_summary = attach_projectors_to_mlp_linears(model, {"mlp": projector})
    assert attach_summary.attached == ("mlp",)

    config = OmegaConf.create(
        {
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "ulysses_sequence_parallel_size": 1,
            "entropy_from_logits_with_chunking": False,
            "use_torch_compile": False,
            "entropy_checkpointing": False,
            "ppo_mini_batch_size": 2,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_epochs": 1,
            "use_dynamic_bsz": False,
            "tis_imp_ratio_cap": -1,
            "use_kl_loss": False,
            "entropy_coeff": 0.0,
            "loss_agg_mode": "token-mean",
            "policy_loss": {"loss_mode": "vanilla"},
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "grad_clip": 10.0,
            "grit": {
                "enable": True,
                "lambda_pres": 0.0,
                "module_pattern": "mlp",
                "require_projected_modules": True,
            },
        }
    )
    actor = DataParallelPPOActor(config=config, actor_module=model, actor_optimizer=optimizer)

    input_ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=torch.long)
    responses = input_ids[:, -2:]
    batch = DataProto.from_dict(
        tensors={
            "responses": responses,
            "response_mask": torch.ones_like(responses, dtype=torch.float32),
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1),
            "old_log_probs": torch.zeros_like(responses, dtype=torch.float32),
            "advantages": torch.ones_like(responses, dtype=torch.float32),
        },
        meta_info={"temperature": 1.0},
    )

    weight_before = model.mlp.weight.detach().clone()
    metrics = actor.update_policy(batch)

    assert "grit/projected_module_count" in metrics
    assert "grit/task_grad_norm_before_projection" in metrics
    assert "grit/task_grad_norm_after_projection" in metrics
    assert metrics["grit/projected_module_count"][-1] == 1.0
    assert metrics["grit/task_grad_norm_before_projection"][-1] > 0.0
    assert metrics["grit/task_grad_norm_after_projection"][-1] <= metrics["grit/task_grad_norm_before_projection"][-1]
    assert metrics["actor/grad_norm"][-1] > 0.0
    assert not torch.allclose(model.mlp.weight.detach(), weight_before)
