# Phase 5 Skill: Total Update

## Goal

Assemble the final GRIT update and expose configs/metrics.

First-order form:

```text
delta_task = AdamW_task(grad_task)
term_task = project(delta_task)
delta_pres = AdamW_pres(v)
delta_final = alpha * term_task - lambda_pres * delta_pres
```

Full form:

```text
delta_task = AdamW_task(grad_task)
term_task = project(delta_task)
delta_pres = AdamW_pres(v + alpha * H P v)
delta_final = alpha * term_task - lambda_pres * delta_pres
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
logs task AdamW delta projection removal
```

## Debug Notes

- Prefer assembling final gradients explicitly over pretending everything is one scalar loss if manual HVP is used.
- Keep NSPO-style weight repair as an ablation only, if implemented.
- The task and preservation AdamW delta states should be separate.
- Project only the task AdamW delta; do not project the preservation correction.
- Do not call a final optimizer step after manually applying the split deltas.
- The frozen base policy used by preservation must not be updated.
