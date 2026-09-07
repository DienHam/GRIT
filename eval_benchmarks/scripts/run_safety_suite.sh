#!/usr/bin/env bash
set -euo pipefail

MODELS_CONFIG=${MODELS_CONFIG:-eval_benchmarks/configs/models.example.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/eval_benchmarks}
BATCH_SIZE=${BATCH_SIZE:-64}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
SAFETY_JUDGE=${SAFETY_JUDGE:-heuristic}
GUARD_MODEL_PATH=${GUARD_MODEL_PATH:-}

ARGS=(
  --models "$MODELS_CONFIG"
  --output-root "$OUTPUT_ROOT"
  --groups safety
  --batch-size "$BATCH_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --safety-judge "$SAFETY_JUDGE"
)

if [[ "$SAFETY_JUDGE" == "llama_guard" ]]; then
  ARGS+=(--guard-model-path "$GUARD_MODEL_PATH")
fi

python eval_benchmarks/run.py "${ARGS[@]}"

