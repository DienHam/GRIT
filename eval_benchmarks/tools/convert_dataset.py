from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json


def load_hf_dataset(dataset: str, config: str | None, split: str) -> list[dict[str, Any]]:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("Hugging Face conversion requires the 'datasets' package") from exc
    if config:
        return datasets.load_dataset(dataset, config, split=split).to_list()
    return datasets.load_dataset(dataset, split=split).to_list()


def pick_value(record: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in record and record[key] not in (None, ""):
            return record[key]
    raise KeyError(f"None of the candidate keys exist: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a HF dataset split to prompt JSONL.")
    parser.add_argument("--dataset", required=True, help="HF dataset id, e.g. declare-lab/HarmfulQA")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-keys", default="prompt,question,goal,instruction")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    prompt_keys = [key.strip() for key in args.prompt_keys.split(",") if key.strip()]
    records = load_hf_dataset(args.dataset, args.config, args.split)
    if args.limit:
        records = records[: args.limit]

    converted = []
    for index, record in enumerate(records):
        prompt = pick_value(record, prompt_keys)
        converted.append({"id": index, "prompt": str(prompt), "raw": record})

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(converted)} prompts to {output_path}")


if __name__ == "__main__":
    main()

