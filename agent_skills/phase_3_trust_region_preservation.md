# Phase 3 Skill: Trust-Region Preservation

## Goal

Implement token-level preservation on `D_pres`:

```text
KL(pi_tilde(. | q, o_<t) || pi_base(. | q, o_<t)) <= epsilon_pres
```

If the predictor violates the bound, project `pi_tilde` toward `pi_base` using TROLL-style sparse trust-region distribution projection and train with a preservation loss.

## Source To Compare

- TROLL paper/code: sparse token-level KL projection (`SdtrplLayer` behavior).
- Important difference: TROLL anchors to `pi_old`; GRIT anchors to frozen `pi_base`.

## Expected Code Surface

Likely files:

- `grit/trust_region.py`
- `grit/preservation_loss.py`
- `test_function/check_trust_region_preservation.py`
- later integration with TROLL's `SdtrplLayer` or equivalent projection layer if vendored

## Required Checks

Before moving on:

```text
accepted tokens produce zero preservation loss
violating tokens produce nonzero preservation loss
projected target satisfies KL(projected || pi_base) <= epsilon_pres
sampled/target response tokens are always retained in sparse logits
dropped tokens have default probability p_d > 0 in sparse KL
violating tokens solve eta* > 0 by bracketing/bisection
projection uses geometric interpolation in log-prob space
preservation gradients flow only through pi_tilde; pi_proj/pi_interpolation is stop-gradient
default loss aggregation is seq-mean-token-mean: average token loss within each response, then average sequences
```

## Debug Notes

- Do not use `old_logits` as the preservation anchor.
- Base logits should come from frozen `pi_base` on the same preservation contexts.
- Use TROLL's geometric interpolation, replacing `pi_old` with frozen `pi_base`:

```text
pi_proj(o_t | q, o_<t) ∝ exp((log pi_tilde(o_t | q, o_<t) + eta* log pi_base(o_t | q, o_<t)) / (eta* + 1))
```

- Find `eta*` with bracketing plus bisection so the active constraint satisfies:

```text
KL(pi_proj || pi_base) <= epsilon_pres
```

- The preservation loss target should be stop-gradient. Do not backpropagate through `pi_proj`, `pi_interpolation`, or `eta*` in the main GRIT objective:

```text
L_pres = KL(pi_tilde || stopgrad(pi_proj))
```

- Sparse projection should preserve the selected response token even if it is not top-K.
- Sparse KL should assign dropped tokens a positive default probability `p_d > 0`; do not silently set dropped-token mass to zero.
- Match the proposal's `1/|o| sum_t` preservation objective and TROLL/verl's `seq-mean-token-mean` aggregation by default. `token-mean` may be kept only as an explicit ablation/debug mode because it overweights long responses.
