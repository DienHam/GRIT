#!/usr/bin/env python3
"""Focused checks for the NSPO three-domain preservation mixture."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "prepare_grit_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_grit_data", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_grit_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_grit_data)


def main() -> None:
    common = [
        {"instruction": f"instruction-{index}", "input": f"context-{index}", "output": "unused"}
        for index in range(400)
    ]
    code = [
        {"query": f"code-query-{index}", "completion": "unused"}
        for index in range(400)
    ]
    math = [
        {"question": f"math-question-{index}", "answer": "unused"}
        for index in range(400)
    ]
    rows = prepare_grit_data.build_nspo_preserve_rows(
        [
            ("common_sense", "tatsu-lab/alpaca_farm", common),
            ("code", "newfacade/LeetCodeDataset", code),
            ("math", "openai/gsm8k", math),
        ],
        max_samples=1000,
        seed=66,
    )

    counts = Counter(row["domain"] for row in rows)
    assert len(rows) == 1000
    assert counts == {"common_sense": 334, "code": 333, "math": 333}
    assert all(row["text"].startswith(prepare_grit_data.PROMPT_BEGIN) for row in rows)
    assert any("instruction-" in row["raw_text"] and "context-" in row["raw_text"] for row in rows)
    assert any(row["raw_text"].startswith("code-query-") for row in rows)
    assert any(row["raw_text"].startswith("math-question-") for row in rows)
    assert not any("unused" in row["raw_text"] for row in rows)
    print(f"nspo_preserve_rows={len(rows)} domains={dict(counts)}")


if __name__ == "__main__":
    main()
