from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json


def collect_metrics(output_root: str | Path) -> list[dict[str, Any]]:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/*/*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(payload)
    return rows


def write_summary(output_root: str | Path) -> tuple[Path, Path]:
    root = Path(output_root)
    rows = collect_metrics(root)
    summary_json = root / "summary.json"
    summary_csv = root / "summary.csv"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "model",
        "benchmark",
        "group",
        "metric",
        "asr_percent",
        "safe_percent",
        "returncode",
        "status",
        "log_path",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary_json, summary_csv

