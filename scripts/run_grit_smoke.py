#!/usr/bin/env python3
"""Run one end-to-end GRIT optimizer-gradient assembly on real HF artifacts."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

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
    parser.add_argument("--task-batch-size", type=int, default=1)
    parser.add_argument("--preserve-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--max-preserve-length", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--lambda-pres", type=float, default=1.0)
    parser.add_argument("--epsilon-pres", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--default-probability", type=float, default=1e-12)
    parser.add_argument("--module-pattern", default="mlp")
    parser.add_argument("--missing-projector", choices=["identity", "zero"], default="identity")
    parser.add_argument("--use-curvature", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_table(path: str):
    from datasets import load_dataset

    return load_dataset("parquet", data_files=path, split="train")


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


def sequence_log_probs(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch["labels"]
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp_min(1)


def dpo_like_task_loss(model, chosen_batch, rejected_batch, *, beta: float = 0.1) -> torch.Tensor:
    chosen_lp = sequence_log_probs(model, chosen_batch)
    rejected_lp = sequence_log_probs(model, rejected_batch)
    return -torch.nn.functional.logsigmoid(beta * (chosen_lp - rejected_lp)).mean()


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


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="right",
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.train()

    base_model = copy.deepcopy(model).to(device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    payload = torch.load(args.projectors_path, map_location="cpu")
    projectors = payload["projectors"] if "projectors" in payload else payload
    module_filter = (
        lambda name, module: isinstance(module, torch.nn.Linear)
        and args.module_pattern in name
    )

    task_ds = load_table(args.task_file)
    preserve_ds = load_table(args.preserve_file)
    task_rows = [task_ds[index] for index in range(args.task_batch_size)]
    preserve_rows = [preserve_ds[index] for index in range(args.preserve_batch_size)]

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

    task_loss = dpo_like_task_loss(model, chosen_batch, rejected_batch)

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

    print(f"task_loss={task_loss.detach().float().item():.8f}")
    for key in (
        "grit/projected_task_grad_norm",
        "grit/preservation_loss",
        "grit/kl_violation_fraction",
        "grit/final_grad_norm",
        "grit/hvp_skipped",
        "grit/projector/rank_mean",
        "grit/projector/nullity_mean",
    ):
        print(f"{key}={result.metrics[key]:.8f}")


if __name__ == "__main__":
    main()
