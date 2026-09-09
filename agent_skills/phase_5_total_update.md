# Phase 5 Skill: Total Update

## Goal

Assemble the final GRIT update and expose configs/metrics.

First-order form:

```text
direction_task = AdamW_direction(grad_task)
term_task = project(direction_task)
delta_final = lr * (term_task - lambda_pres * v)
```

Full form:

```text
direction_task = AdamW_direction(grad_task)
term_task = project(direction_task)
correction = v - lr * H P v
delta_final = lr * (term_task - lambda_pres * correction)
```

`H P v` may come from exact autograd HVP or SAM-FD approximation.

## Source To Compare

- GRIT proposal: predictor-corrector preservation is one training step.
- NSPO baseline: gradient projection only.
- TROLL baseline: trust-region output projection only.

## Expected Code Surface

Likely files:

- `grit/update.py`
- `config/method/grit.yaml`
- `test_function/check_total_update.py`
- integration in `verl/verl/workers/actor/dp_actor.py` and trainer code

## Required Checks

Before calling the implementation complete:

```text
can run gradient-projection-only ablation
can run first-order GRIT
can toggle SAM-FD curvature
logs projection rank/nullity and KL violation fraction
logs preservation loss and whether HVP was skipped
logs task AdamW delta projection removal
```

## Debug Notes

- Prefer assembling final gradients explicitly over pretending everything is one scalar loss if manual HVP is used.
- Keep NSPO-style weight repair as an ablation only, if implemented.
- Only the task direction uses AdamW state; preservation is raw SGD descent.
- Project only the task AdamW delta; do not project the preservation correction.
- Build `theta_tilde` from the projected task AdamW delta, not from raw `P grad_task`.
- Do not call a final optimizer step after manually applying the split deltas.
- The frozen base policy used by preservation must not be updated.
