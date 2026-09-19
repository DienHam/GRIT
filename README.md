# GRIT
Official repository for the paper "GRIT: Gradient Projection Meets Trust-Region Anchoring for Forgetting-Resistant Reinforcement Learning"

Start from `WORKFLOW.md` when running or modifying the repo. It is the compact
map of the current artifact flow, phase contracts, training commands, metrics,
and Google Drive sync expectations.

For the AlpacaFarm/GSM8K/LeetCodeDataset preservation mixture, see
[preservation data preparation](docs/preservation_data.md).

## Phase 1: Null-Space Gradient Projection

Phase 1 implements the NSPO-theory gradient projection primitive:

```text
grad_W <- grad_W @ P
```

where `P = U_null U_null^T` is computed from preservation-set activation covariances.

Run the toy sanity check:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python test_function/check_projection.py
```

Build projectors from a Hugging Face model and preservation dataset:

```bash
scripts/run_build_projectors.sh \
  --model-path /path/to/base-model \
  --dataset-path /path/to/preservation-dataset \
  --dataset-split train \
  --text-column prompt \
  --module-pattern mlp \
  --relative-threshold 5e-4 \
  --max-samples 1024 \
  --batch-size 8 \
  --max-length 256 \
  --output-path artifacts/projectors.pt
```

When wiring Phase 1 into `verl`, load and attach the saved projectors once in
the actor worker, then project gradients after the RL/task backward:

```python
from verl.experimental.grit.projector import (
    attach_projectors_to_mlp_linears,
    load_projectors,
    project_actor_mlp_gradients,
)

projectors = load_projectors("artifacts/projectors.pt", map_location="cpu")
attach_projectors_to_mlp_linears(actor_module, projectors)

task_loss.backward()
projection_metrics = project_actor_mlp_gradients(actor_module)
```

Call this before gradient clipping and `optimizer.step()`. This is gradient
projection only; do not use NSPO's periodic weight repair as the GRIT Phase 1
mechanism. With FSDP1, enable `actor_rollout_ref.actor.fsdp_config.use_orig_params=True`
so `Linear.weight.grad` exists for projection.

## Phase 2: Predictor Weights

Phase 2 implements the temporary predictor point used for preservation checks:

```text
theta_tilde = theta + lr * projected_task_direction
```

Use `temporary_predictor_step(...)` to forward a preservation batch at
`theta_tilde`; by default it updates only Linear weight parameters, matching
the Phase 1 projector surface. The original `theta` is restored when the
context exits.

Run the toy restore check:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python test_function/check_predictor_restore.py
```

## Phase 3: Trust-Region Preservation

Phase 3 implements TROLL-style sparse token-level preservation on `D_pres`
anchored to the frozen base policy:

```text
KL(pi_tilde(. | q, o_<t) || pi_base(. | q, o_<t)) <= epsilon_pres
```

Use `preservation_kl_loss(...)` to compute:

```text
L_pres = KL(pi_tilde || stopgrad(pi_proj))
```

Accepted tokens keep `pi_tilde` as the target and produce zero preservation
loss. Violating tokens are projected toward `pi_base` with geometric
interpolation in log-prob space, solving `eta*` by bracketing/bisection until
the target satisfies the KL bound. Preservation gradients flow only through
`pi_tilde`; `pi_proj`/`pi_interpolation` is used as a detached target.
The default reduction is `seq-mean-token-mean`, matching the proposal's
`1/|o| sum_t` objective and TROLL/verl's length-normalized aggregation.
Sparse-default KL keeps policy/base top-k logits, always keeps the selected
response token, and assigns dropped tokens a positive default probability
`p_d > 0`.

Run the toy preservation check:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python test_function/check_trust_region_preservation.py
```

`verl` integration evaluates Phase 3 at the Phase 2 predictor point. Enable
top-level `grit.preservation.enable` (passed into the actor config), provide a
tensorized `D_preserve` file with `input_ids`, `attention_mask`,
`position_ids`, `responses`, and `response_mask`, and set `base_model_path`
for the frozen `pi_base`. The actor computes the PPO/task backward, projects
the task gradients, temporarily applies
`theta_tilde = theta + lr * projected_task_direction`, evaluates
`KL(pi_tilde || pi_base)` through the same trust-region projection loss,
restores `theta`, and combines the final gradient with
`lambda_pres * grad_{theta_tilde} L_pres`. It logs:

```text
grit/preservation_loss
grit/kl_violation_fraction
```

Run the actor integration check:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python -m pytest \
  verl/tests/experimental/grit/test_phase3_preservation_actor_on_cpu.py -q
```

## Phase 4: SAM-FD Curvature

Phase 4 implements the optional curvature correction from the unrolled GRIT
objective using SAM-style finite difference in training:

```text
v - lr * H_task(theta) P v
```

where `v = grad_{theta_tilde} L_pres(theta_tilde)` and `P` is a full-theta
block operator: protected Linear weights use the per-layer null-space
projectors from Phase 1, while parameters without a projector use identity.
The SAM-FD approximation skips the curvature path when the projected vector is
zero. If the training code minimizes `task_loss = -J_task`, the approximation
follows that minimization-loss sign. The exact autograd HVP helper remains only
for toy correctness checks.

Run the toy HVP check:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python test_function/check_hvp.py
```

## Phase 5: Total GRIT Update

Phase 5 assembles the optimizer-facing gradient in one explicit step:

```text
final_grad = projected_task_grad - lambda_pres * v
```

with the optional curvature form:

```text
update_direction = projected_task_direction - lambda_pres * (v - lr * H P v)
```

Use `assemble_grit_update(...)` after computing the task minimization loss and
before `optimizer.step()`. The preservation loss callback is evaluated at
`theta_tilde = theta + lr * projected_task_direction`, then the original model
weights are restored and `parameter.grad` is replaced with the final GRIT
gradient. Set `lambda_pres: 0` for the gradient-projection-only ablation,
`use_curvature: false` for first-order GRIT, and `use_curvature: true` to add
the Phase 4 HVP correction.

Default method settings live in:

```bash
config/method/grit.yaml
```

Run the total-update check:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python test_function/check_total_update.py
```

## NSPO-Style GRPO Safety Task Objective

For the real NSPO-style task gradient, `D_task` supplies prompts only. The
current policy rolls out `G` responses per prompt, a Llama/Llama-Guard style
safety model scores each `(prompt, response)` pair, and the reward is:

```text
r_{i,g} =  0   if response is safe
r_{i,g} = -1   if response is unsafe
```

The group advantage is normalized within each prompt group:

```text
A_{i,g} = (r_{i,g} - mean_g r_{i,g}) / (std_g r_{i,g} + eps)
```

The task loss is clipped GRPO/PPO without an extra task KL penalty:

```text
L_task = - mean_{i,g} min(
    rho_{i,g} A_{i,g},
    clip(rho_{i,g}, 1-epsilon, 1+epsilon) A_{i,g}
)

rho_{i,g} = pi_theta(o_{i,g} | q_i) / pi_old(o_{i,g} | q_i)
```

GRIT then treats this as the task gradient. Protected Linear weights use:

```text
grad_W = grad_W_GRPO_clipped @ P
```

Unprotected parameters keep their normal clipped-GRPO gradient. Preservation
KL remains separate in Phase 3 and is anchored to the frozen base policy.

Run the objective sanity check:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python test_function/check_grpo_safety_objective.py
```

## Real-Model Smoke Workflow: Qwen2.5-0.5B + PKU-SafeRLHF

The pieces above are connected by `scripts/run_grit_workflow.sh` for a
first real-artifact run:

1. Prepare PKU-SafeRLHF task parquet files with 11K train samples.
2. Prepare a 1,000-sample preservation parquet file.
3. Build Phase 1 projectors from the preservation set.
4. Run one end-to-end DPO-style GRIT smoke step through Phase 2-5.

```bash
PYTHON_BIN=/Users/apple/miniconda3/envs/grit-qwen3/bin/python \
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
TASK_DATASET=PKU-Alignment/PKU-SafeRLHF \
scripts/run_grit_workflow.sh
```

Qwen2.5 works with the pinned Transformers release in `requirements.txt`. With
older environments, data preparation still works, but model loading may fail
before projector building.
The workflow disables Hugging Face Xet downloads by default because they can
stall on some local networks while fetching model weights.
Long steps print timestamps and progress bars. During projector building,
`[3/5] Forward batches` tracks model forward over preservation batches, and
`[4/5] Eigendecomposition` tracks projector construction per protected layer.

By default the preservation file uses the three general-task sources cited by
the NSPO paper: AlpacaFarm instructions, LeetCodeDataset train, and GSM8K train.
The paper specifies 1,000 mixed prompts but not the mixing ratio, so this repo
uses a deterministic near-even split (334 common-sense/instruction, 333 code,
333 math). To use a custom preservation dataset instead, pass a Hugging Face
dataset name or local path:

```bash
PRESERVE_DATASET=/path/to/mixed_preserve_dataset \
scripts/run_grit_workflow.sh
```

Rebuild only the default NSPO preservation file without touching task data:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python scripts/prepare_grit_data.py \
  --preserve-only \
  --preserve-max-samples 1000 \
  --output-dir data/grit_qwen2_5_0_5b
```

Intermediate artifacts:

```text
data/grit_qwen2_5_0_5b/task_train.parquet
data/grit_qwen2_5_0_5b/task_val.parquet
data/grit_qwen2_5_0_5b/preserve_1000.parquet
artifacts/qwen2_5_0_5b_projectors.pt
```

Run only the smoke step after projectors already exist:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python scripts/run_grit_smoke.py \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --task-file data/grit_qwen2_5_0_5b/task_train.parquet \
  --preserve-file data/grit_qwen2_5_0_5b/preserve_1000.parquet \
  --projectors-path artifacts/qwen2_5_0_5b_projectors.pt \
  --trust-remote-code
```

## Kaggle Full Training

On Kaggle, keep the CUDA PyTorch that is already installed and install only the
GPU notebook dependencies:

```bash
pip install -r requirements-kaggle.txt
```

After `task_train.parquet`, `preserve_1000.parquet`, and
`qwen2_5_0_5b_projectors.pt` exist, launch 2-GPU manual data-parallel training:

```bash
MAX_STEPS=1000 SAVE_STEPS=100 NPROC_PER_NODE=2 scripts/run_kaggle_grit_train.sh
```

Rank 0 shows a `tqdm` progress bar with task loss, preservation loss, KL
violation fraction, and final gradient norm. Checkpoints are written under
`/kaggle/working/grit_qwen2_5_0_5b` by default and can be resumed:

```bash
scripts/run_kaggle_grit_train.sh \
  --resume-from-checkpoint /kaggle/working/grit_qwen2_5_0_5b/latest_checkpoint.txt
```

To push checkpoints to Hugging Face Hub, set `HF_TOKEN` in Kaggle Secrets and
add:

```bash
scripts/run_kaggle_grit_train.sh \
  --push-to-hub \
  --hub-repo-id your-username/grit-qwen2-5-0-5b
```

## Modal GPU Notebook: NSPO-style Small-Model Run

Install the vLLM-enabled `verl` extra before running the repository entrypoint:

```bash
python -m pip install -r requirements-kaggle.txt
python -m pip install -e 'verl[vllm]'
```

Use the separate Modal notebook artifact from your Downloads folder, select two
A10/L4 GPUs (or one A100-40/80GB; the runner
automatically colocates the tiny guard in that case), and attach the `grit-data`
Volume at `/mnt/grit-data`.
The notebook follows the original NSPO recipe from `ivanniu/NSPO`—GRPO on
PKU-SafeRLHF, five rollouts per prompt, `safe=0`/`unsafe=-1`, and MLP
activation null-space projectors—while using the small-model path already
implemented here (`Qwen2.5-0.5B-Instruct` plus `Qwen3Guard-Gen-0.6B`).
The default backend is `verl_vllm`: GPU 0 runs the FSDP actor plus vLLM
rollout, while GPU 1 serves the small guard through an OpenAI-compatible vLLM
endpoint. Set `TRAIN_BACKEND="standalone_hf"` when only one GPU is available.

`METHOD="nspo"` is the projection-only baseline (`lambda_pres=0`). The vLLM
runner intentionally stays on this path; predictor/preservation/HVP ablations
remain in the standalone Transformers trainer. Run `RUN_MODE="smoke"` first,
then switch to `RUN_MODE="full"` for the 1,000-context, 11K-task run.

The reusable repository entrypoint for the projection-only vLLM path is
`scripts/run_grit_vllm.sh`. GPU 0 runs the FSDP actor plus vLLM rollout and GPU 1
runs the Qwen3Guard vLLM reward server. Set `OUTPUT_DIR` and the dataset/projector
paths to directories on the attached Modal Volume:

```bash
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
SAFETY_MODEL_PATH=Qwen/Qwen3Guard-Gen-0.6B \
TASK_FILE=/mnt/grit-data/grit/data/task_train.parquet \
VAL_FILE=/mnt/grit-data/grit/data/task_val.parquet \
PROJECTORS_PATH=/mnt/grit-data/grit/artifacts/projectors.pt \
OUTPUT_DIR=/mnt/grit-data/grit/checkpoints/grit_vllm \
MAX_STEPS=32 \
bash scripts/run_grit_vllm.sh
```

This path uses vLLM for rollout and safety scoring, while the actor backward,
GRIT MLP gradient projection, and optimizer update remain in PyTorch/FSDP.
