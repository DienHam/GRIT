# Phase 2 Skill: Predictor Theta Tilde

## Goal

Create temporary predictor weights:

```text
theta_tilde = theta - alpha * projected_grad
```

then forward a preservation batch at `theta_tilde`, and restore the original `theta`.

## Source To Compare

- GRIT proposal: predictor is the point training is about to occupy after the projected task step.
- This is not present in vanilla NSPO or TROLL code.

## Expected Code Surface

Likely files:

- `grit/predictor.py`
- `test_function/check_predictor_restore.py`
- later integration in `verl/verl/workers/actor/dp_actor.py`

## Required Checks

Before moving on:

```text
theta -> theta_tilde changes logits on a toy/preservation batch
restore returns every parameter exactly or within dtype tolerance
no optimizer state is stepped during predictor creation
```

## Debug Notes

- The first implementation can use non-differentiable temporary updates under `torch.no_grad()`.
- Store deltas, not a full model copy, where possible:

```text
delta = -alpha * projected_grad
p.add_(delta)
p.sub_(delta)
```

- In FSDP, temporary parameter updates need careful shard/full-param handling. Prototype on single GPU first.
