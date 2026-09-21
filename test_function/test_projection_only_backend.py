"""CPU regression checks for the projection-only Kaggle backend."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "verl")]

from verl import DataProto
from verl.experimental.grit.optimizer import ProjectedAdamW
from verl.workers.actor.dp_actor import DataParallelPPOActor
from scripts import nspo_vllm_reward as reward


def linear_model():
    model = nn.Sequential()
    model.add_module("mlp", nn.Linear(2, 2, bias=False))
    model.add_module("head", nn.Linear(2, 1, bias=False))
    # Rotated projector exposes the difference between P Adam(g) and Adam(P g).
    u = torch.tensor([1.0, 2.0]) / (5**0.5)
    model.mlp.proj_w = u[:, None] @ u[None, :]
    return model


@pytest.mark.parametrize("weight_decay", [0.0, 0.13])
def test_adamw_direction_projection_and_raw_moments(weight_decay):
    torch.manual_seed(8)
    model = linear_model()
    reference = copy.deepcopy(model)
    lr = 0.003
    opt = ProjectedAdamW(model, lr=lr, weight_decay=weight_decay)
    adam = torch.optim.AdamW(reference.parameters(), lr=lr, weight_decay=weight_decay)
    for step in range(3):
        before = model.mlp.weight.detach().clone()
        before_head = model.head.weight.detach().clone()
        reference.load_state_dict(model.state_dict())
        for p, ref in zip(model.parameters(), reference.parameters()):
            g = torch.randn_like(p) * (step + 1)
            p.grad, ref.grad = g.clone(), g.clone()
        raw = model.mlp.weight.grad.clone()
        adam.step()
        expected = before + (reference.mlp.weight.detach() - before) @ model.mlp.proj_w
        opt.step()
        torch.testing.assert_close(model.mlp.weight, expected, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(model.head.weight, reference.head.weight)
        torch.testing.assert_close(model.mlp.weight.grad, raw)
        torch.testing.assert_close(opt.state[model.mlp.weight]["exp_avg"], adam.state[reference.mlp.weight]["exp_avg"])
        torch.testing.assert_close(opt.state[model.mlp.weight]["exp_avg_sq"], adam.state[reference.mlp.weight]["exp_avg_sq"])
        leakage = (model.mlp.weight - before) @ (torch.eye(2) - model.mlp.proj_w)
        assert leakage.norm() < 1e-7
        assert not torch.equal(model.head.weight, before_head)
    assert opt.param_groups[0]["grit_policy_version"] == 3


def test_checkpoint_roundtrip_and_nonfinite_is_atomic():
    model = linear_model()
    opt = ProjectedAdamW(model, lr=1e-4)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    clone = copy.deepcopy(model)
    resumed = ProjectedAdamW(clone, lr=0.9)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p, q in zip(model.parameters(), clone.parameters()):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    opt.step()
    resumed.step()
    for p, q in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    before = copy.deepcopy(model.state_dict())
    moments = copy.deepcopy(opt.state_dict())
    model.head.weight.grad[0, 0] = float("inf")
    opt.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert opt.state_dict()["param_groups"] == moments["param_groups"]
    for key, state in opt.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, moments["state"][key][name])
    assert opt.last_metrics["grit/nonfinite_skipped"] == 1


def test_missing_projector_and_half_master_rejected():
    model = linear_model()
    del model.mlp.proj_w
    with pytest.raises(ValueError, match="Missing projector"):
        ProjectedAdamW(model)
    model = linear_model().half()
    opt = ProjectedAdamW(model)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="FP32 master"):
        opt.step()
    assert not opt.state


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 2)
        self.mlp = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 8, bias=False)
        self.mlp.proj_w = torch.tensor([[0.2, 0.4], [0.4, 0.8]])

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(torch.tanh(self.mlp(self.embed(input_ids)))))


def make_actor(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_id", lambda: "cpu")
    config = OmegaConf.create(dict(use_remove_padding=False, use_fused_kernels=False,
        ulysses_sequence_parallel_size=1, entropy_from_logits_with_chunking=False,
        use_torch_compile=False, entropy_checkpointing=False, ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1, ppo_epochs=1, use_dynamic_bsz=False,
        tis_imp_ratio_cap=-1, use_kl_loss=False, entropy_coeff=0.0,
        loss_agg_mode="seq-mean-token-mean", policy_loss={"loss_mode": "vanilla"},
        clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0,
        grad_clip=1.0, fsdp_config={"mixed_precision": {"param_dtype": "fp32"}},
        grit={"enable": True, "lambda_pres": 0, "preservation": {"enable": False}}))
    model = TinyLM()
    optimizer = ProjectedAdamW(model, lr=1e-4)
    actor = DataParallelPPOActor(config, model, optimizer)
    ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    batch = DataProto.from_dict(tensors=dict(input_ids=ids, responses=ids[:, -2:],
        response_mask=torch.ones(2, 2), attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(4).repeat(2, 1), old_log_probs=torch.full((2, 2), -2.),
        advantages=torch.tensor([[1., 1.], [-1., -1.]])), meta_info={"temperature": 1.0})
    return actor, model, optimizer, batch


def test_actor_uses_frozen_old_logprobs_and_skips_zero_advantage(monkeypatch):
    actor, model, optimizer, batch = make_actor(monkeypatch)
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    loss_fn = get_policy_loss_fn("vanilla")
    seen = []
    def capture(**kwargs):
        seen.append(kwargs["old_log_prob"].clone())
        return loss_fn(**kwargs)
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_policy_loss_fn", lambda _: capture)
    before = model.mlp.weight.detach().clone()
    metrics = actor.update_policy(batch)
    assert all(torch.equal(x, torch.full_like(x, -2.)) for x in seen)
    assert metrics["grit/optimizer_updates"] == [1.0]
    assert metrics["grit/projected_module_count"] == [1.0]
    assert ((model.mlp.weight - before) @ (torch.eye(2) - model.mlp.proj_w)).norm() < 1e-7
    state = copy.deepcopy(optimizer.state_dict())
    weights = copy.deepcopy(model.state_dict())
    batch.batch["advantages"].zero_()
    metrics = actor.update_policy(batch)
    assert metrics["grit/zero_advantage_batch"] == [1.0]
    assert len(seen) == 2  # No extra forward/loss on the zero-signal batch.
    assert optimizer.state_dict()["param_groups"] == state["param_groups"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key])


def test_actor_nonfinite_gradient_skips_step(monkeypatch):
    actor, model, optimizer, batch = make_actor(monkeypatch)
    model.head.weight.register_hook(lambda gradient: gradient * float("inf"))
    before = copy.deepcopy(model.state_dict())
    metrics = actor.update_policy(batch)
    assert metrics["grit/nonfinite_skipped"] == [1.0]
    assert not optimizer.state
    assert optimizer.param_groups[0]["grit_policy_version"] == 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize("label,score", [("Safe", 0), ("Unsafe", -1), ("Controversial", -1)])
def test_guard_uses_prompt_response_template_and_structured_labels(monkeypatch, label, score):
    seen = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.append(messages)
            assert kwargs == dict(tokenize=False, add_generation_prompt=False)
            return "rendered official template"
    def completion(**kwargs):
        assert kwargs["prompt"] == "rendered official template"
        return SimpleNamespace(choices=[SimpleNamespace(text=f"Safety: {label}\nCategories: None\nRefusal: Yes")])
    monkeypatch.setattr(reward, "_tokenizer", lambda: Tokenizer())
    monkeypatch.setattr(reward, "_client", lambda: SimpleNamespace(completions=SimpleNamespace(create=completion)))
    result = reward.compute_score_batched(solution_strs=["response"], extra_infos=[{"raw_prompt": "prompt"}])[0]
    assert result["score"] == score and result["guard/refusal_rate"] == 1
    assert seen == [[{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "response"}]]


def test_guard_parse_retry_fails_closed(monkeypatch):
    calls = []
    monkeypatch.setattr(reward, "_tokenizer", lambda: SimpleNamespace(apply_chat_template=lambda *a, **k: "text"))
    def completion(**kwargs):
        calls.append(1)
        return SimpleNamespace(choices=[SimpleNamespace(text="probably safe")])
    monkeypatch.setattr(reward, "_client", lambda: SimpleNamespace(completions=SimpleNamespace(create=completion)))
    monkeypatch.setattr(reward.time, "sleep", lambda _: None)
    monkeypatch.setenv("NSPO_GUARD_ATTEMPTS", "2")
    with pytest.raises(RuntimeError, match="parse_errors=2"):
        reward.compute_score_batched(solution_strs=["response"], extra_infos=[{"raw_prompt": "prompt"}])
    assert len(calls) == 2
    with pytest.raises(ValueError, match="raw_prompt"):
        reward.compute_score_batched(solution_strs=["response"], extra_infos=[{}])
    with pytest.raises(ValueError):
        reward.parse_guard_response("Safety: Safe\nSafety: Unsafe\nRefusal: No")


def test_verl_adapter_preserves_guard_prompt_and_reward_schema(tmp_path):
    from datasets import Dataset
    source, target = tmp_path / "source.parquet", tmp_path / "target.parquet"
    Dataset.from_list([{"raw_prompt": "Question one"}, {"raw_prompt": "Question two"}]).to_parquet(str(source))
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_verl_nspo_data.py"),
                    "--input", str(source), "--output", str(target)], check=True, capture_output=True)
    rows = Dataset.from_parquet(str(target))
    assert rows[0]["extra_info"]["raw_prompt"] == "Question one"
    assert rows[0]["reward_model"] == {"style": "rule", "ground_truth": ""}
    assert rows[1]["prompt"] == [{"role": "user", "content": "Question two"}]


def test_runner_overrides_compose_with_t4_config(tmp_path):
    """Execute the actual shell runner with process boundaries stubbed, then compose Hydra config."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "args.json"
    interpreter = fake_bin / "capture-python"
    interpreter.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                           "open(os.environ['CAPTURE_ARGS'], 'w').write(json.dumps(sys.argv[2:]))\n")
    interpreter.chmod(0o755)
    vllm = fake_bin / "vllm"
    vllm.write_text("#!/bin/sh\nexit 0\n")
    vllm.chmod(0o755)
    for name in ["projectors.pt", "train.parquet", "val.parquet"]:
        (tmp_path / name).touch()
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
               PYTHON_BIN=str(interpreter), CAPTURE_ARGS=str(capture), START_GUARD_SERVER="0",
               TASK_FILE=str(tmp_path / "train.parquet"), VAL_FILE=str(tmp_path / "val.parquet"),
               PROJECTORS_PATH=str(tmp_path / "projectors.pt"), VERL_DATA_DIR=str(tmp_path),
               OUTPUT_DIR=str(tmp_path / "checkpoints"), ACTOR_GPU="0", GUARD_GPU="1")
    subprocess.run(["bash", str(ROOT / "scripts/run_grit_vllm.sh"), "trainer.total_training_steps=3",
                    "trainer.resume_mode=resume_path", "trainer.resume_from_path=/tmp/global_step_2"],
                   env=env, check=True, capture_output=True)
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass
    with initialize_config_dir(config_dir=str(ROOT / "verl/verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=json.loads(capture.read_text()))
    assert cfg.trainer.total_training_steps == 3
    assert cfg.trainer.resume_from_path == "/tmp/global_step_2"
    assert cfg.actor_rollout_ref.actor.optim.weight_decay == 0
    assert cfg.actor_rollout_ref.model.override_config._attn_implementation == "sdpa"
    assert cfg.actor_rollout_ref.rollout.calculate_log_probs
    actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    assert actor.fsdp_config.model_dtype == "fp32"
    assert actor.fsdp_config.mixed_precision["param_dtype"] == "fp16"
    assert rollout.dtype == "float16" and rollout.seed == 66
