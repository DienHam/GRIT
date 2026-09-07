#!/usr/bin/env bash
set -euo pipefail

MODELS_CONFIG=${MODELS_CONFIG:-eval_benchmarks/configs/models.example.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/eval_benchmarks}

python eval_benchmarks/run.py \
  --models "$MODELS_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --groups general

