# GRIT Agent Notes

This repository implements GRIT: Gradient Projection Meets Trust-Region Anchoring for Forgetting-Resistant Reinforcement Learning.

Use the phase skill cards in `agent_skills/` when implementing or debugging. Each card names the goal, source papers/repos to compare against, expected code surface, and checks that should pass before moving on.

## Phase Routing

| Phase | Skill card | Purpose |
| --- | --- | --- |
| 1 | `agent_skills/phase_1_gradient_projection.md` | Build null-space projectors and apply `grad_W <- grad_W @ P`. |
| 2 | `agent_skills/phase_2_predictor_theta_tilde.md` | Create temporary predictor weights `theta_tilde` and restore `theta`. |
| 3 | `agent_skills/phase_3_trust_region_preservation.md` | Add TROLL-style token KL projection on `D_pres` anchored to `pi_base`. |
| 4 | `agent_skills/phase_4_curvature_hvp.md` | Add optional curvature term `H P v` with Hessian-vector products. |
| 5 | `agent_skills/phase_5_total_update.md` | Assemble final GRIT gradients, logging, configs, and ablations. |

## Non-Negotiables

- Keep algorithm pieces small and testable before wiring into `verl`.
- Prefer explicit CLI/config arguments over hard-coded model or dataset paths.
- Do not use NSPO's periodic weight repair as the primary GRIT mechanism:

```text
W <- W_base + (W - W_base) P
```

- For GRIT, the NSPO part should primarily be:

```text
grad_W <- grad_W @ P
```

- For GRIT, the TROLL-style preservation anchor should be the frozen base policy:

```text
KL(pi_tilde(. | q, o_<t) || pi_base(. | q, o_<t)) <= epsilon_pres
```

not the previous rollout policy `pi_old`.

## Useful Commands

Run the Phase 1 toy sanity check:

```bash
/Users/apple/miniconda3/envs/grit-qwen3/bin/python test_function/check_projection.py
```

Build projectors from a Hugging Face model and dataset:

```bash
scripts/run_build_projectors.sh \
  --model-path /path/to/base-model \
  --dataset-path /path/to/preservation-dataset \
  --text-column prompt \
  --output-path artifacts/projectors.pt
```
