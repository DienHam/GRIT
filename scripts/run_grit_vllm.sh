#!/usr/bin/env bash
set -euo pipefail

# Run projection-only GRIT with verl's FSDP actor and vLLM rollout.
# GPU 0 is reserved for the actor/rollout worker; GPU 1 serves Qwen3Guard.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
SAFETY_MODEL_PATH="${SAFETY_MODEL_PATH:-Qwen/Qwen3Guard-Gen-0.6B}"
TASK_FILE="${TASK_FILE:-data/grit_qwen2_5_0_5b/task_train.parquet}"
VAL_FILE="${VAL_FILE:-data/grit_qwen2_5_0_5b/task_val.parquet}"
PROJECTORS_PATH="${PROJECTORS_PATH:-artifacts/qwen2_5_0_5b_projectors.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/grit_vllm_qwen2_5_0_5b}"
VERL_DATA_DIR="${VERL_DATA_DIR:-${OUTPUT_DIR}/verl_data}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GRPO_GENERATIONS="${GRPO_GENERATIONS:-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ACTOR_GPU="${ACTOR_GPU:-0}"
if [[ -n "${GUARD_GPU:-}" ]]; then
  GUARD_GPU="${GUARD_GPU}"
else
  GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${GPU_COUNT:-0}" -ge 2 ]]; then
    GUARD_GPU="1"
  else
    # A single A100-40/80GB can colocate the tiny guard and actor rollout.
    GUARD_GPU="${ACTOR_GPU}"
  fi
fi
GUARD_PORT="${GUARD_PORT:-52001}"
GUARD_LOG="${GUARD_LOG:-${OUTPUT_DIR}/qwen3guard_vllm.log}"
GUARD_BASE_URL="${GUARD_BASE_URL:-http://127.0.0.1:${GUARD_PORT}/v1}"
START_GUARD_SERVER="${START_GUARD_SERVER:-1}"
GUARD_GPU_MEMORY_UTILIZATION="${GUARD_GPU_MEMORY_UTILIZATION:-0.20}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}"

mkdir -p "${OUTPUT_DIR}" "${VERL_DATA_DIR}"

if [[ ! -f "${PROJECTORS_PATH}" ]]; then
  echo "Missing projector artifact: ${PROJECTORS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TASK_FILE}" || ! -f "${VAL_FILE}" ]]; then
  echo "TASK_FILE and VAL_FILE must exist before starting vLLM training." >&2
  exit 1
fi

VERL_TRAIN_FILE="${VERL_DATA_DIR}/train.parquet"
VERL_VAL_FILE="${VERL_DATA_DIR}/val.parquet"
if [[ ! -f "${VERL_TRAIN_FILE}" ]]; then
  "${PYTHON_BIN}" scripts/prepare_verl_nspo_data.py --input "${TASK_FILE}" --output "${VERL_TRAIN_FILE}"
fi
if [[ ! -f "${VERL_VAL_FILE}" ]]; then
  "${PYTHON_BIN}" scripts/prepare_verl_nspo_data.py --input "${VAL_FILE}" --output "${VERL_VAL_FILE}"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed. Install it with: python -m pip install -e 'verl[vllm]'" >&2
  exit 1
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export TOKENIZERS_PARALLELISM="false"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}"

guard_pid=""
cleanup() {
  if [[ -n "${guard_pid}" ]] && kill -0 "${guard_pid}" 2>/dev/null; then
    kill "${guard_pid}" 2>/dev/null || true
    wait "${guard_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${START_GUARD_SERVER}" == "1" || "${START_GUARD_SERVER}" == "true" ]]; then
  echo "Starting Qwen3Guard vLLM on GPU ${GUARD_GPU}, ${GUARD_BASE_URL}"
  mkdir -p "$(dirname "${GUARD_LOG}")"
  CUDA_VISIBLE_DEVICES="${GUARD_GPU}" vllm serve "${SAFETY_MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${GUARD_PORT}" \
    --dtype half \
    --max-model-len "${GUARD_MAX_MODEL_LEN:-2048}" \
    --gpu-memory-utilization "${GUARD_GPU_MEMORY_UTILIZATION}" \
    --enforce-eager \
    >"${GUARD_LOG}" 2>&1 &
  guard_pid=$!

  "${PYTHON_BIN}" - <<PY
import time
import urllib.request

url = "${GUARD_BASE_URL}/models"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=2):
            print("Guard server is ready", flush=True)
            break
    except Exception:
        time.sleep(2)
else:
    from pathlib import Path
    log = Path("${GUARD_LOG}")
    raise SystemExit(f"Guard server did not become ready. Log:\n{log.read_text(errors='replace')[-4000:]}")
PY
fi

echo "Starting GRIT projection-only training with vLLM rollout"
echo "actor_gpu=${ACTOR_GPU} guard_gpu=${GUARD_GPU} steps=${MAX_STEPS} output=${OUTPUT_DIR}"
if [[ "${ACTOR_GPU}" == "${GUARD_GPU}" ]]; then
  echo "Using one-GPU colocated mode; use a 40/80GB GPU for this configuration."
fi

CUDA_VISIBLE_DEVICES="${ACTOR_GPU}" \
NSPO_GUARD_BASE_URL="${GUARD_BASE_URL}" \
NSPO_GUARD_MODEL="${SAFETY_MODEL_PATH}" \
"${PYTHON_BIN}" "${ROOT_DIR}/verl/verl/trainer/main_ppo.py" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="${VERL_TRAIN_FILE}" \
  data.val_files="${VERL_VAL_FILE}" \
  data.prompt_key=prompt \
  data.reward_fn_key=data_source \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  +actor_rollout_ref.model.override_config._attn_implementation=sdpa \
  +actor_rollout_ref.model.override_config.use_cache=false \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  ++actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  ++actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  ++actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp16 \
  ++actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
  ++actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
  ++data.seed=66 \
  ++actor_rollout_ref.actor.fsdp_config.seed=66 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  ++actor_rollout_ref.rollout.seed=66 \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.n="${GRPO_GENERATIONS}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  custom_reward_function.path="${ROOT_DIR}/scripts/nspo_vllm_reward.py" \
  custom_reward_function.name=compute_score_batched \
  reward_model.reward_manager=batch \
  grit.enable=true \
  grit.projectors_path="${PROJECTORS_PATH}" \
  grit.lambda_pres=0.0 \
  grit.use_curvature=false \
  grit.preservation.enable=false \
  grit.module_pattern=mlp \
  grit.strict_projector_attach=true \
  grit.require_projected_modules=true \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${MAX_STEPS}" \
  trainer.save_freq="${SAVE_STEPS}" \
  trainer.test_freq="${EVAL_STEPS}" \
  trainer.val_before_train=false \
  trainer.logger='[console]' \
  trainer.project_name=grit_vllm \
  trainer.experiment_name=qwen2_5_0_5b_projection_only \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  "$@"
