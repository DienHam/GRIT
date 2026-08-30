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
```

## Debug Notes

- Linear layer weights are shaped `[out_features, in_features]`; right-multiply by `P`.
- `P` must be `[in_features, in_features]`.
- If a covariance matrix is all zeros, the whole input space is null and `P` should be identity.
- Avoid `.data` for gradient projection; use `module.weight.grad.copy_(...)` before optimizer step.
