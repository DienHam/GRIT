# Phase 4 Skill: Curvature HVP / SAM-FD

## Goal

Add optional curvature, using either exact HVP or SAM-style finite difference:

```text
v - lr * H P v
```

where:

```text
v = grad_{theta_tilde} L_pres(theta_tilde)
H = Hessian_theta L_task(theta)
```

SAM-FD approximation:

```text
u = P v
H u ~= (grad_task(theta + rho * u / ||u||) - grad_task(theta)) * ||u|| / rho
```

## Source To Compare

- GRIT proposal: the curvature term appears from differentiating through `theta_tilde(theta)`.
- Neither NSPO nor TROLL public code provides this full term.

## Expected Code Surface

Likely files:

- `grit/curvature.py`
- `test_function/check_hvp.py`
- later integration in actor update

## Required Checks

Before moving on:

```text
manual HVP matches finite difference on a toy model
SAM-FD approximates exact HVP on a toy model
curvature path is skipped when v == 0
create_graph=True is only used for exact HVP, not SAM-FD
```

## Debug Notes

Manual HVP identity:

```text
H u = grad_theta <grad_theta J_task(theta), u>
u = P v
```

Implementation sketch:

```python
task_grads = torch.autograd.grad(task_loss, params, create_graph=True)
u = project_with_P(v).detach()
dot = sum((g * ui).sum() for g, ui in zip(task_grads, u))
hvp = torch.autograd.grad(dot, params)
```

- Do not build the full Hessian matrix.
- Prefer `curvature_mode=sam_fd` for memory-constrained real-model tests.
- Watch signs: if code minimizes `task_loss = -J_task`, document whether `H` belongs to `J_task` or `task_loss`.
