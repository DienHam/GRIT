#!/usr/bin/env python3
"""Train Qwen with GRIT on preference pairs, with resumable DDP-style updates."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import AdamW
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.preservation_loss import preservation_kl_loss
from grit.update import GritUpdateConfig, assemble_grit_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task-file", default="data/grit_qwen2_5_0_5b/task_train.parquet")
    parser.add_argument("--preserve-file", default="data/grit_qwen2_5_0_5b/preserve_1000.parquet")
    parser.add_argument("--projectors-path", default="artifacts/qwen2_5_0_5b_projectors.pt")
    parser.add_argument("--output-dir", default="checkpoints/grit_qwen2_5_0_5b")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--task-batch-size", type=int, default=1, help="Per-process task batch size.")
    parser.add_argument("--preserve-batch-size", type=int, default=1, help="Per-process preservation batch size.")
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--max-preserve-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--lambda-pres", type=float, default=1.0)
    parser.add_argument("--epsilon-pres", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--default-probability", type=float, default=1e-6)
    parser.add_argument("--module-pattern", default="mlp")
    parser.add_argument("--missing-projector", choices=["identity", "zero"], default="identity")
    parser.add_argument("--use-curvature", action="store_true")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default=None)
    parser.add_argument("--hub-private", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def load_table(path: str):
    from datasets import load_dataset

    return load_dataset("parquet", data_files=path, split="train")


def shard_indices(length: int, rank: int, world_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    indices = list(range(length))
    rng.shuffle(indices)
    return indices[rank::world_size]


def take_rows(dataset, indices: list[int], start: int, batch_size: int) -> tuple[list[dict[str, Any]], int]:
    rows = []
    position = start
    for _ in range(batch_size):
        rows.append(dataset[indices[position % len(indices)]])
        position += 1
    return rows, position


def pad_tokenized(tokenizer, input_ids: list[list[int]], labels: list[list[int]]):
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    padded_labels = []
    attention = []
    for ids, label in zip(input_ids, labels, strict=True):
        pad = max_len - len(ids)
        padded_ids.append(ids + [tokenizer.pad_token_id] * pad)
        padded_labels.append(label + [-100] * pad)
        attention.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def tokenize_pair_batch(tokenizer, rows, *, max_prompt_length: int, max_response_length: int):
    prompts = [row["prompt"] for row in rows]
    chosen = [row["chosen"] for row in rows]
    rejected = [row["rejected"] for row in rows]
    prompt_tokens = tokenizer(
        prompts,
        max_length=max_prompt_length,
        padding=False,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]

    def build(responses):
        input_ids = []
        labels = []
        for prompt_ids, response in zip(prompt_tokens, responses, strict=True):
            response_ids = tokenizer(
                response,
                max_length=max_response_length,
                padding=False,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
            response_ids = response_ids + [tokenizer.eos_token_id]
            ids = prompt_ids + response_ids
            label = [-100] * len(prompt_ids) + response_ids
            input_ids.append(ids)
            labels.append(label)
        return pad_tokenized(tokenizer, input_ids, labels)

    return build(chosen), build(rejected)


def tokenize_preserve_batch(tokenizer, rows, *, max_length: int):
    texts = [row["text"] for row in rows]
    tokens = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    selected = input_ids[:, 1:].contiguous()
    response_mask = attention_mask[:, 1:].bool().contiguous()
    return input_ids, attention_mask, selected, response_mask


def move_batch(batch, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def sequence_log_probs(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch["labels"]
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp_min(1)


def dpo_like_task_loss(model, chosen_batch, rejected_batch, *, beta: float) -> torch.Tensor:
    chosen_lp = sequence_log_probs(model, chosen_batch)
    rejected_lp = sequence_log_probs(model, rejected_batch)
    return -torch.nn.functional.logsigmoid(beta * (chosen_lp - rejected_lp)).mean()


def all_reduce_parameter_grads(model: torch.nn.Module, world_size: int) -> None:
    if world_size == 1:
        return
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def reduce_metrics(metrics: dict[str, float], world_size: int, device: torch.device) -> dict[str, float]:
    if world_size == 1:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values.div_(world_size)
    return {key: float(value) for key, value in zip(keys, values.tolist(), strict=True)}


def save_checkpoint(
    *,
    model,
    tokenizer,
    optimizer,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> Path:
    checkpoint_dir = output_dir / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    state = {
        "step": step,
        "args": vars(args),
        "metrics": metrics,
    }
    (checkpoint_dir / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    latest_path = output_dir / "latest_checkpoint.txt"
    latest_path.write_text(str(checkpoint_dir), encoding="utf-8")
    return checkpoint_dir


def maybe_push_to_hub(args: argparse.Namespace, checkpoint_dir: Path, step: int) -> None:
    if not args.push_to_hub:
        return
    if not args.hub_repo_id:
        raise ValueError("--hub-repo-id is required when --push-to-hub is set")
    from huggingface_hub import HfApi, create_repo

    create_repo(args.hub_repo_id, private=args.hub_private, exist_ok=True)
    api = HfApi()
    api.upload_folder(
        repo_id=args.hub_repo_id,
        folder_path=str(checkpoint_dir),
        path_in_repo=f"checkpoints/step_{step:06d}",
        commit_message=f"GRIT checkpoint step {step}",
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(f"checkpoints/step_{step:06d}\n")
        latest_pointer = Path(handle.name)
    try:
        api.upload_file(
            repo_id=args.hub_repo_id,
            path_or_fileobj=str(latest_pointer),
            path_in_repo="checkpoints/latest_checkpoint.txt",
            commit_message=f"Update latest GRIT checkpoint to step {step}",
        )
    finally:
        latest_pointer.unlink(missing_ok=True)


def load_checkpoint_if_needed(model, optimizer, checkpoint_path: str | None, device: torch.device) -> int:
    if checkpoint_path is None:
        return 0
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_file():
        checkpoint_dir = Path(checkpoint_dir.read_text(encoding="utf-8").strip())
    from transformers import AutoModelForCausalLM

    loaded_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    model.load_state_dict(loaded_model.state_dict())
    model.to(device)
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    state_path = checkpoint_dir / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state.get("step", 0))
    return 0


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    main_process = is_main_process(rank)

    if args.device == "cuda" and world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if main_process:
        print(f"world_size={world_size} device={device} dtype={args.dtype}", flush=True)
        print(f"loading tokenizer/model: {args.model_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="right",
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = dtype_from_name(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.train()
    model.config.use_cache = False

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    base_model.eval()
    base_model.config.use_cache = False
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = load_checkpoint_if_needed(model, optimizer, args.resume_from_checkpoint, device)

    if main_process:
        print(f"loading projectors: {args.projectors_path}", flush=True)
    payload = torch.load(args.projectors_path, map_location="cpu")
    projectors = payload["projectors"] if "projectors" in payload else payload
    module_filter = (
        lambda name, module: isinstance(module, torch.nn.Linear)
        and args.module_pattern in name
    )

    if main_process:
        print(f"loading data: {args.task_file} / {args.preserve_file}", flush=True)
    task_ds = load_table(args.task_file)
    preserve_ds = load_table(args.preserve_file)
    task_indices = shard_indices(len(task_ds), rank, world_size, args.seed)
    preserve_indices = shard_indices(len(preserve_ds), rank, world_size, args.seed + 1)
    if not task_indices or not preserve_indices:
        raise ValueError("Dataset shard is empty. Use fewer workers or more data.")

    task_pos = start_step * args.task_batch_size
    preserve_pos = start_step * args.preserve_batch_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(
        range(start_step + 1, args.max_steps + 1),
        desc="GRIT train",
        disable=not main_process,
        initial=start_step,
        total=args.max_steps,
    )
    last_metrics: dict[str, float] = {}

    for step in progress:
        task_rows, task_pos = take_rows(task_ds, task_indices, task_pos, args.task_batch_size)
        preserve_rows, preserve_pos = take_rows(
            preserve_ds,
            preserve_indices,
            preserve_pos,
            args.preserve_batch_size,
        )

        chosen_batch, rejected_batch = tokenize_pair_batch(
            tokenizer,
            task_rows,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
        )
        chosen_batch = move_batch(chosen_batch, device)
        rejected_batch = move_batch(rejected_batch, device)
        pres_input_ids, pres_attention_mask, selected_ids, response_mask = tokenize_preserve_batch(
            tokenizer,
            preserve_rows,
            max_length=args.max_preserve_length,
        )
        pres_input_ids = pres_input_ids.to(device)
        pres_attention_mask = pres_attention_mask.to(device)
        selected_ids = selected_ids.to(device)
        response_mask = response_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        task_loss = dpo_like_task_loss(model, chosen_batch, rejected_batch, beta=args.dpo_beta)

        def preservation_loss_fn():
            policy_logits = model(
                input_ids=pres_input_ids,
                attention_mask=pres_attention_mask,
            ).logits[:, :-1, :]
            with torch.no_grad():
                base_logits = base_model(
                    input_ids=pres_input_ids,
                    attention_mask=pres_attention_mask,
                ).logits[:, :-1, :]
            return preservation_kl_loss(
                policy_logits,
                base_logits,
                epsilon_pres=args.epsilon_pres,
                response_mask=response_mask,
                selected_token_ids=selected_ids,
                top_k=args.top_k,
                default_probability=args.default_probability,
            )

        result = assemble_grit_update(
            model,
            task_loss,
            preservation_loss_fn,
            projectors,
            config=GritUpdateConfig(
                alpha=args.alpha,
                lambda_pres=args.lambda_pres,
                use_curvature=args.use_curvature,
                missing_projector=args.missing_projector,
            ),
            module_filter=module_filter,
        )
        all_reduce_parameter_grads(model, world_size)
        optimizer.step()

        local_metrics = {
            "task_loss": float(task_loss.detach().float().item()),
            "pres_loss": result.metrics["grit/preservation_loss"],
            "kl_viol": result.metrics["grit/kl_violation_fraction"],
            "grad": result.metrics["grit/final_grad_norm"],
            "proj_grad": result.metrics["grit/projected_task_grad_norm"],
        }
        last_metrics = reduce_metrics(local_metrics, world_size, device)

        if main_process and step % args.log_steps == 0:
            progress.set_postfix(
                task=f"{last_metrics['task_loss']:.4f}",
                pres=f"{last_metrics['pres_loss']:.2e}",
                kl=f"{last_metrics['kl_viol']:.3f}",
                grad=f"{last_metrics['grad']:.3f}",
            )

        if main_process and args.save_steps > 0 and step % args.save_steps == 0:
            checkpoint_dir = save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                output_dir=output_dir,
                step=step,
                args=args,
                metrics=last_metrics,
            )
            print(f"\nsaved checkpoint: {checkpoint_dir}", flush=True)
            maybe_push_to_hub(args, checkpoint_dir, step)

    if main_process:
        checkpoint_dir = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            output_dir=output_dir,
            step=args.max_steps,
            args=args,
            metrics=last_metrics,
        )
        print(f"\nfinal checkpoint: {checkpoint_dir}", flush=True)
        maybe_push_to_hub(args, checkpoint_dir, args.max_steps)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
