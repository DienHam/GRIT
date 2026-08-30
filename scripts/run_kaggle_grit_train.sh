#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
if [[ -z "${TASK_FILE:-}" ]]; then
  if [[ -f data/grit_qwen2_5_0_5b/task_train.parquet ]]; then
    TASK_FILE="data/grit_qwen2_5_0_5b/task_train.parquet"
  else
    TASK_FILE="data/grit_qwen3_0_6b/task_train.parquet"
  fi
fi
if [[ -z "${PRESERVE_FILE:-}" ]]; then
  if [[ -f data/grit_qwen2_5_0_5b/preserve_1000.parquet ]]; then
    PRESERVE_FILE="data/grit_qwen2_5_0_5b/preserve_1000.parquet"
  else
    PRESERVE_FILE="data/grit_qwen3_0_6b/preserve_1000.parquet"
  fi
fi
PROJECTORS_PATH="${PROJECTORS_PATH:-artifacts/qwen2_5_0_5b_projectors.pt}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ -d /kaggle/working ]]; then
    OUTPUT_DIR="/kaggle/working/grit_qwen2_5_0_5b"
  else
    OUTPUT_DIR="checkpoints/grit_qwen2_5_0_5b"
  fi
fi
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK_FILE=${TASK_FILE}"
echo "PRESERVE_FILE=${PRESERVE_FILE}"
echo "PROJECTORS_PATH=${PROJECTORS_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} NPROC_PER_NODE=${NPROC_PER_NODE}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/train_grit_dpo.py \
  --model-path "${MODEL_PATH}" \
  --task-file "${TASK_FILE}" \
  --preserve-file "${PRESERVE_FILE}" \
  --projectors-path "${PROJECTORS_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --task-batch-size 1 \
  --preserve-batch-size 1 \
  --max-prompt-length 256 \
  --max-response-length 128 \
  --max-preserve-length 256 \
  --lr 1e-6 \
  --alpha 1e-2 \
  --lambda-pres 1.0 \
  --epsilon-pres 1e-4 \
  --top-k 64 \
  --default-probability 1e-6 \
  --dtype float16 \
  --trust-remote-code \
  "$@"
