# Phase 1 Skill: Gradient Projection

## Goal

Implement NSPO-theory null-space gradient projection:

```text
grad_W <- grad_W @ P
P = U_null U_null^T
```

This phase should not implement `theta_tilde`, KL preservation, or curvature.

## Source To Compare

- NSPO paper: projected gradient is `(grad_W J) @ P`.
- NSPO public code: useful for collecting activations and computing `P`, but its `amend_perturbation()` repair is not the GRIT main mechanism.

## Current Code Surface

- `grit/projection.py`
- `scripts/build_projectors.py`
- `test_function/check_projection.py`
- `scripts/run_build_projectors.sh`
- `verl/verl/experimental/grit/projector.py`
- `verl/verl/workers/fsdp_workers.py`
- `verl/verl/workers/actor/dp_actor.py`
- `verl/verl/trainer/config/ppo_trainer.yaml`

## VERL Integration Contract

Load and attach projectors once when the actor worker builds the actor model
in `verl/verl/workers/fsdp_workers.py`:

```python
from verl.experimental.grit.projector import attach_projectors_to_mlp_linears, load_projectors

projectors = load_projectors(projector_path, map_location="cpu")
attach_projectors_to_mlp_linears(actor_module, projectors)
```

Then project task gradients immediately after the RL/task loss backward and
before gradient clipping, optimizer step, or scheduler step in
`verl/verl/workers/actor/dp_actor.py`:

```python
from verl.experimental.grit.projector import project_actor_mlp_gradients

task_loss.backward()
projection_metrics = project_actor_mlp_gradients(actor_module)
```

This is the only Phase 1 optimizer-facing operation:

```text
grad_W <- grad_W @ P
```

Do not call NSPO-style periodic weight repair:

```text
W <- W_base + (W - W_base) @ P
```

For FSDP1 smoke tests, set:

```text
grit.enable=True
grit.projectors_path=/path/to/projectors.pt
actor_rollout_ref.actor.fsdp_config.use_orig_params=True
```

Expected metrics:

```text
grit/task_grad_norm_before_projection
grit/task_grad_norm_after_projection
grit/projected_module_count
grit/attached_projector_count
grit/projector_grad_missing_count
```

## Required Checks

Run:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python test_function/check_projection.py
```

Expected properties:

```text
P^T ~= P
P^2 ~= P
||grad_W @ P|| <= ||grad_W||
leakage into protected covariance is small
attached projector path produces the same no-leak property
```

## Debug Notes

- Linear layer weights are shaped `[out_features, in_features]`; right-multiply by `P`.
- `P` must be `[in_features, in_features]`.
- If a covariance matrix is all zeros, the whole input space is null and `P` should be identity.
- Avoid `.data` for gradient projection; use `module.weight.grad.copy_(...)` before optimizer step.
