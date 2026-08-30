# Phase 5 Skill: Total Update

## Goal

Assemble the final GRIT update and expose configs/metrics/ablations.

First-order form:

```text
final_grad = projected_task_grad - lambda_pres * v
```

Full form:

```text
final_grad = projected_task_grad - lambda_pres * (v + alpha * H P v)
```

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
can toggle curvature
logs projection rank/nullity and KL violation fraction
logs preservation loss and whether HVP was skipped
```

## Debug Notes

- Prefer assembling final gradients explicitly over pretending everything is one scalar loss if manual HVP is used.
- Keep NSPO-style weight repair as an ablation only, if implemented.
- Optimizer state should receive the final GRIT gradient, not the unprojected task gradient.
- The frozen base policy used by preservation must not be updated.
