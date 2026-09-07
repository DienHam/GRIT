from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class Benchmark:
    name: str
    group: str
    metric: str
    higher_is_better: bool
    runner: str
    dataset_path: str | None = None
    prompt_key: str | None = None
    judge: str | None = None
    command: list[str] | None = None
    notes: str = ""


def load_benchmarks(path: str | Path) -> list[Benchmark]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    benchmarks: list[Benchmark] = []
    for group in ("safety", "general"):
        for item in payload.get(group, []):
            benchmarks.append(Benchmark(group=group, **item))
    return benchmarks


def select_benchmarks(
    benchmarks: list[Benchmark],
    groups: set[str] | None = None,
    names: set[str] | None = None,
) -> list[Benchmark]:
    selected: list[Benchmark] = []
    for benchmark in benchmarks:
        if groups and benchmark.group not in groups:
            continue
        if names and benchmark.name not in names:
            continue
        selected.append(benchmark)
    return selected


def load_model_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    models = payload.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError(f"{config_path} must contain a non-empty 'models' list")
    for model in models:
        if "name" not in model or "path" not in model:
            raise ValueError("Each model entry must contain 'name' and 'path'")
    return payload

