# GRIT
Official repository for the paper "GRIT: Gradient Projection Meets Trust-Region Anchoring for Forgetting-Resistant Reinforcement Learning"

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

## Phase 2: Predictor Weights

Phase 2 implements the temporary predictor point used for preservation checks:

```text
theta_tilde = theta - alpha * projected_grad
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

## Phase 4: Curvature HVP

Phase 4 implements the optional exact curvature correction from the unrolled
GRIT objective:

```text
v + alpha * H_task(theta) P v
```

where `v = grad_{theta_tilde} L_pres(theta_tilde)` and `P` is a full-theta
block operator: protected Linear weights use the per-layer null-space
projectors from Phase 1, while parameters without a projector use identity.
The HVP uses the Pearlmutter identity
`H u = grad_theta <grad_theta task_loss, u>` and skips the curvature path when
the projected vector is zero. If the training code minimizes
`task_loss = -J_task`, the HVP follows that minimization-loss sign.

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
final_grad = projected_task_grad - lambda_pres * (v + alpha * H P v)
```

Use `assemble_grit_update(...)` after computing the task minimization loss and
before `optimizer.step()`. The preservation loss callback is evaluated at
`theta_tilde = theta - alpha * projected_task_grad`, then the original model
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

By default the preservation file is sampled from the task dataset prompt
column, which is enough to validate the GRIT wiring. If you have the NSPO-style
mixed preservation dataset from common-sense/math/code prompts, pass it as a
Hugging Face dataset name or local dataset path:

```bash
PRESERVE_DATASET=/path/to/mixed_preserve_dataset \
scripts/run_grit_workflow.sh
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
