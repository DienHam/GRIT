#!/usr/bin/env bash
# Prepare frozen-base contexts and projectors, with all outputs under DATA_ROOT.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts}"
PROMPTS="$DATA_ROOT/preservation/nspo_mix"
CONTEXTS="$DATA_ROOT/preservation/qwen2_5_0_5b"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
PROJECTORS_PATH="${PROJECTORS_PATH:-$ARTIFACT_ROOT/qwen2_5_0_5b_context_projectors.pt}"
STAGE="${1:-all}"
case "$STAGE" in sample|generate|build|all) ;; *) echo 'Usage: bash scripts/run_preservation_projectors.sh [sample|generate|build|all]' >&2; exit 2;; esac

# Reuse only complete, checksum-matching datasets. Keep incomplete attempts.
check_dataset() {
  "$PYTHON_BIN" - "$1" "$2" "$3" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, filename, stage = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['stage'] == stage, 'Unexpected preservation stage'
assert manifest['rows'] == 1000, 'Expected 1000 preservation rows'
assert manifest['domain_counts'] == {'general': 334, 'math': 333, 'code': 333}
assert manifest['parquet'] == filename
assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == manifest['parquet_sha256'], 'Dataset checksum mismatch'
print('Verified:', root / filename)
PY
}
keep_partial() {
  if [[ -e "$1" ]]; then
    local backup="$1.partial.$(date +%Y%m%dT%H%M%S).$$"
    mv -- "$1" "$backup"
    echo "Kept unfinished attempt: $backup"
  fi
}

if [[ "$STAGE" == sample || "$STAGE" == all ]]; then
  if [[ -s "$PROMPTS/manifest.json" ]]; then
    check_dataset "$PROMPTS" preserve_prompts.parquet prompts_only
  else
    keep_partial "$PROMPTS"
    "$PYTHON_BIN" -u scripts/prepare_preservation_data.py sample --output-dir "$PROMPTS"
    check_dataset "$PROMPTS" preserve_prompts.parquet prompts_only
  fi
fi

if [[ "$STAGE" == generate || "$STAGE" == all ]]; then
  check_dataset "$PROMPTS" preserve_prompts.parquet prompts_only
  if [[ -s "$CONTEXTS/manifest.json" ]]; then
    check_dataset "$CONTEXTS" preserve_contexts.parquet base_contexts
  else
    keep_partial "$CONTEXTS"
    "$PYTHON_BIN" -u scripts/prepare_preservation_data.py generate \
      --prompts "$PROMPTS/preserve_prompts.parquet" --output-dir "$CONTEXTS" \
      --model-path "$MODEL_PATH" --max-prompt-length 2048 --max-new-tokens 256 --device cuda
    check_dataset "$CONTEXTS" preserve_contexts.parquet base_contexts
  fi
fi

if [[ "$STAGE" == build || "$STAGE" == all ]]; then
  check_dataset "$CONTEXTS" preserve_contexts.parquet base_contexts
  if [[ -e "$PROJECTORS_PATH" ]]; then
    echo "Projector already exists: $PROJECTORS_PATH"
    echo 'Not overwriting. Set PROJECTORS_PATH to a new filename if rebuilding.'
    exit 0
  fi
  MODEL_SNAPSHOT=$("$PYTHON_BIN" - "$CONTEXTS/manifest.json" "$MODEL_PATH" <<'PY'
import json, sys
from huggingface_hub import snapshot_download
manifest = json.load(open(sys.argv[1]))
assert manifest['base_model'] == sys.argv[2], 'Base model mismatch'
print(snapshot_download(manifest['base_model'], revision=manifest['base_revision']))
PY
  )
  mkdir -p "$(dirname "$PROJECTORS_PATH")"
  PARTIAL_PATH="$PROJECTORS_PATH.partial.$$"
  "$PYTHON_BIN" -u scripts/build_projectors.py \
    --model-path "$MODEL_SNAPSHOT" \
    --dataset-path "$CONTEXTS/preserve_contexts.parquet" \
    --dataset-split train --text-column text --module-pattern mlp \
    --relative-threshold 5e-4 --max-samples 1000 --seed 66 \
    --batch-size 1 --max-length 2304 --dtype float16 --device cuda \
    --output-path "$PARTIAL_PATH"
  test -s "$PARTIAL_PATH"
  mv -- "$PARTIAL_PATH" "$PROJECTORS_PATH"
  echo "Saved: $PROJECTORS_PATH"
  du -h "$PROJECTORS_PATH"
fi
