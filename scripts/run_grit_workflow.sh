#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
TASK_DATASET="${TASK_DATASET:-PKU-Alignment/PKU-SafeRLHF}"
PRESERVE_DATASET="${PRESERVE_DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-data/grit_qwen2_5_0_5b}"
PROJECTORS_PATH="${PROJECTORS_PATH:-artifacts/qwen2_5_0_5b_projectors.pt}"
PYTHON_BIN="${PYTHON_BIN:-/Users/apple/miniconda3/envs/grit-qwen3/bin/python}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

log_step() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

log_step "GRIT workflow start"
log_step "Model: ${MODEL_PATH}"
log_step "Output dir: ${OUTPUT_DIR}"
log_step "Projectors: ${PROJECTORS_PATH}"

prepare_args=(
  scripts/prepare_grit_data.py
  --task-dataset "${TASK_DATASET}"
  --task-max-samples 11000
  --val-size 1000
  --preserve-max-samples 1000
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${PRESERVE_DATASET}" ]]; then
  prepare_args+=(--preserve-dataset "${PRESERVE_DATASET}")
fi

log_step "Step 1/3: prepare task/preservation parquet data"
"${PYTHON_BIN}" "${prepare_args[@]}"

log_step "Step 2/3: build null-space projectors"
"${PYTHON_BIN}" scripts/build_projectors.py \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${OUTPUT_DIR}/preserve_1000.parquet" \
  --dataset-split train \
  --text-column text \
  --module-pattern mlp \
  --relative-threshold 5e-4 \
  --max-samples 1000 \
  --batch-size 2 \
  --max-length 256 \
  --dtype float32 \
  --output-path "${PROJECTORS_PATH}" \
  --trust-remote-code

log_step "Step 3/3: run one GRIT smoke update"
"${PYTHON_BIN}" scripts/run_grit_smoke.py \
  --model-path "${MODEL_PATH}" \
  --task-file "${OUTPUT_DIR}/task_train.parquet" \
  --preserve-file "${OUTPUT_DIR}/preserve_1000.parquet" \
  --projectors-path "${PROJECTORS_PATH}" \
  --task-batch-size 1 \
  --preserve-batch-size 1 \
  --alpha 1e-5 \
  --lambda-pres 1.0 \
  --epsilon-pres 0.05 \
  --top-k 64 \
  --dtype float32 \
  --trust-remote-code

log_step "GRIT workflow complete"
