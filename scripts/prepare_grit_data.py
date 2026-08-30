#!/usr/bin/env python3
"""Prepare task and preservation parquet files for a first GRIT run.

The task set is built from PKU-SafeRLHF preference/safety labels. The
preservation set is a 1,000-sample prompt/text pool used by Phase 1 projector
building and Phase 3 trust-region anchoring.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROMPT_BEGIN = "BEGINNING OF CONVERSATION: "
PROMPT_USER = "USER: {input} "
PROMPT_ASSISTANT = "ASSISTANT:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dataset", default="PKU-Alignment/PKU-SafeRLHF")
    parser.add_argument("--task-split", default="train")
    parser.add_argument("--task-max-samples", type=int, default=11000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--preserve-dataset", default=None)
    parser.add_argument("--preserve-split", default="train")
    parser.add_argument("--preserve-text-column", default=None)
    parser.add_argument("--preserve-max-samples", type=int, default=1000)
    parser.add_argument("--prefer-safe", action="store_true", default=True)
    parser.add_argument("--prefer-helpful", dest="prefer_safe", action="store_false")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--output-dir", default="data/grit_qwen2_5_0_5b")
    return parser.parse_args()


def format_prompt(prompt: str) -> str:
    return PROMPT_BEGIN + PROMPT_USER.format(input=prompt) + PROMPT_ASSISTANT


def choose_response_ids(row: dict[str, Any], *, prefer_safe: bool) -> tuple[int, int]:
    preferred_key = "safer_response_id" if prefer_safe else "better_response_id"
    if preferred_key in row and row[preferred_key] is not None:
        chosen = int(row[preferred_key])
        return chosen, 1 - chosen

    if "better_response_id" in row and row["better_response_id"] is not None:
        chosen = int(row["better_response_id"])
        return chosen, 1 - chosen

    safety_0 = row.get("is_response_0_safe")
    safety_1 = row.get("is_response_1_safe")
    if safety_0 is not None and safety_1 is not None and safety_0 != safety_1:
        chosen = 0 if bool(safety_0) else 1
        return chosen, 1 - chosen

    raise ValueError("Cannot infer chosen/rejected response ids from PKU row.")


def row_to_task(row: dict[str, Any], *, prefer_safe: bool) -> dict[str, Any]:
    chosen_id, rejected_id = choose_response_ids(row, prefer_safe=prefer_safe)
    return {
        "data_source": "PKU-SafeRLHF",
        "prompt": format_prompt(str(row["prompt"])),
        "raw_prompt": str(row["prompt"]),
        "chosen": str(row[f"response_{chosen_id}"]),
        "rejected": str(row[f"response_{rejected_id}"]),
        "chosen_response_id": chosen_id,
        "rejected_response_id": rejected_id,
    }


def infer_text_column(dataset, requested: str | None) -> str:
    if requested is not None:
        if requested not in dataset.column_names:
            raise ValueError(
                f"Column {requested!r} not found. Available columns: {dataset.column_names}"
            )
        return requested
    for candidate in ("prompt", "question", "instruction", "problem", "text"):
        if candidate in dataset.column_names:
            return candidate
    raise ValueError(
        "Could not infer preservation text column. Pass --preserve-text-column. "
        f"Available columns: {dataset.column_names}"
    )


def load_any_dataset(dataset_path: str, split: str):
    from datasets import load_dataset

    path = Path(dataset_path)
    if path.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split=split)
    if path.suffix == ".json" or path.suffix == ".jsonl":
        return load_dataset("json", data_files=str(path), split=split)
    if path.suffix == ".csv":
        return load_dataset("csv", data_files=str(path), split=split)
    return load_dataset(dataset_path, split=split)


def main() -> None:
    args = parse_args()

    from datasets import Dataset

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading task dataset: {args.task_dataset}", flush=True)
    task_raw = load_any_dataset(args.task_dataset, args.task_split)
    task_indices = list(range(len(task_raw)))
    random.shuffle(task_indices)
    total_task = min(args.task_max_samples + args.val_size, len(task_indices))
    task_rows = [
        row_to_task(task_raw[index], prefer_safe=args.prefer_safe)
        for index in tqdm(task_indices[:total_task], desc="[2/4] Formatting task rows")
    ]
    train_rows = task_rows[: args.task_max_samples]
    val_rows = task_rows[args.task_max_samples :]

    train_path = output_dir / "task_train.parquet"
    val_path = output_dir / "task_val.parquet"
    print(f"[3/4] Writing task parquet files to {output_dir}", flush=True)
    Dataset.from_list(train_rows).to_parquet(str(train_path))
    Dataset.from_list(val_rows).to_parquet(str(val_path))

    preserve_dataset_name = args.preserve_dataset or args.task_dataset
    print(f"[4/4] Loading preservation dataset: {preserve_dataset_name}", flush=True)
    preserve_raw = load_any_dataset(preserve_dataset_name, args.preserve_split)
    text_column = infer_text_column(preserve_raw, args.preserve_text_column)
    preserve_indices = list(range(len(preserve_raw)))
    random.shuffle(preserve_indices)
    preserve_rows = []
    with tqdm(total=args.preserve_max_samples, desc="[4/4] Sampling preserve rows") as progress:
        for index in preserve_indices:
            text = str(preserve_raw[index][text_column]).strip()
            if not text:
                continue
            preserve_rows.append(
                {
                    "data_source": preserve_dataset_name,
                    "text": format_prompt(text) if text_column == "prompt" else text,
                    "raw_text": text,
                }
            )
            progress.update(1)
            if len(preserve_rows) >= args.preserve_max_samples:
                break

    preserve_path = output_dir / "preserve_1000.parquet"
    Dataset.from_list(preserve_rows).to_parquet(str(preserve_path))

    print(f"task_train={train_path} rows={len(train_rows)}")
    print(f"task_val={val_path} rows={len(val_rows)}")
    print(
        f"preserve={preserve_path} rows={len(preserve_rows)} "
        f"source={preserve_dataset_name} text_column={text_column}"
    )


if __name__ == "__main__":
    main()
