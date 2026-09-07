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
METRIC_WINDOW="${METRIC_WINDOW:-20}"
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-0}"
EVAL_GENERATIONS="${EVAL_GENERATIONS:-1}"
if [[ -z "${EVAL_FILE:-}" ]]; then
  TASK_DIR="$(dirname "${TASK_FILE}")"
  if [[ -f "${TASK_DIR}/task_val.parquet" ]]; then
    EVAL_FILE="${TASK_DIR}/task_val.parquet"
  else
    EVAL_FILE=""
  fi
fi
EVAL_OUTPUT_FILE="${EVAL_OUTPUT_FILE:-}"
HVP_LAST_LINEAR_LAYERS="${HVP_LAST_LINEAR_LAYERS:-0}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-0}"
BEST_MIN_DELTA="${BEST_MIN_DELTA:-0.0}"
VISIBLE_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ "${VISIBLE_GPU_COUNT}" -gt 0 ]]; then
    NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
  else
    NPROC_PER_NODE="1"
  fi
elif [[ "${VISIBLE_GPU_COUNT}" -gt 0 && "${NPROC_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} but only ${VISIBLE_GPU_COUNT} CUDA device(s) are visible; using ${VISIBLE_GPU_COUNT}." >&2
  NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
fi
TASK_OBJECTIVE="${TASK_OBJECTIVE:-dpo_pair}"
GRPO_GENERATIONS="${GRPO_GENERATIONS:-4}"
GRPO_CLIP_RATIO="${GRPO_CLIP_RATIO:-0.2}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
SAFETY_MODEL_PATH="${SAFETY_MODEL_PATH:-}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK_FILE=${TASK_FILE}"
echo "PRESERVE_FILE=${PRESERVE_FILE}"
echo "PROJECTORS_PATH=${PROJECTORS_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} METRIC_WINDOW=${METRIC_WINDOW} NPROC_PER_NODE=${NPROC_PER_NODE} VISIBLE_GPU_COUNT=${VISIBLE_GPU_COUNT}"
echo "EVAL_STEPS=${EVAL_STEPS} EVAL_SAMPLES=${EVAL_SAMPLES} EVAL_GENERATIONS=${EVAL_GENERATIONS} EVAL_FILE=${EVAL_FILE} EVAL_OUTPUT_FILE=${EVAL_OUTPUT_FILE}"
echo "SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT} BEST_MIN_DELTA=${BEST_MIN_DELTA}"
echo "UPDATE_RULE=split_adamw_delta"
echo "TASK_OBJECTIVE=${TASK_OBJECTIVE}"
if [[ "${TASK_OBJECTIVE}" == "grpo_safety" ]]; then
  echo "SAFETY_MODEL_PATH=${SAFETY_MODEL_PATH}"
  if [[ -z "${SAFETY_MODEL_PATH}" ]]; then
    echo "SAFETY_MODEL_PATH is required when TASK_OBJECTIVE=grpo_safety" >&2
    exit 1
  fi
fi

BEST_CHECKPOINT_ARGS=()
if [[ "${SAVE_BEST_CHECKPOINT}" == "1" || "${SAVE_BEST_CHECKPOINT}" == "true" ]]; then
  BEST_CHECKPOINT_ARGS=(--save-best-checkpoint --best-min-delta "${BEST_MIN_DELTA}")
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/train_grit_dpo.py \
  --model-path "${MODEL_PATH}" \
  --task-file "${TASK_FILE}" \
  --preserve-file "${PRESERVE_FILE}" \
  --projectors-path "${PROJECTORS_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --metric-window "${METRIC_WINDOW}" \
  --eval-steps "${EVAL_STEPS}" \
  --eval-samples "${EVAL_SAMPLES}" \
  --eval-generations "${EVAL_GENERATIONS}" \
  --eval-file "${EVAL_FILE}" \
  --eval-output-file "${EVAL_OUTPUT_FILE}" \
  "${BEST_CHECKPOINT_ARGS[@]}" \
  --task-objective "${TASK_OBJECTIVE}" \
  --grpo-generations "${GRPO_GENERATIONS}" \
  --grpo-clip-ratio "${GRPO_CLIP_RATIO}" \
  --rollout-temperature "${ROLLOUT_TEMPERATURE}" \
  --rollout-top-p "${ROLLOUT_TOP_P}" \
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
  --hvp-last-linear-layers "${HVP_LAST_LINEAR_LAYERS}" \
  --dtype float16 \
  --trust-remote-code \
  --safety-model-path "${SAFETY_MODEL_PATH}" \
  "$@"
