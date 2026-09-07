from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json


def read_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path)
    suffix = data_path.suffix.lower()

    if suffix == ".jsonl":
        with data_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if suffix == ".json":
        with data_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "examples", "records", "test"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Cannot find a record list in {data_path}")

    if suffix == ".csv":
        with data_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported dataset format: {data_path}")


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

