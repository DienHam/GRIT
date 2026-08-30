#!/usr/bin/env python3
"""Toy sanity check for GRIT Phase 3 trust-region preservation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.preservation_loss import preservation_kl_loss
from grit.trust_region import project_to_kl_ball, token_kl_from_logits


def main() -> None:
    torch.manual_seed(23)

    base_logits = torch.tensor(
        [
            [[5.0, 1.0, 0.0, -1.0, -2.0], [4.0, 0.0, -1.0, -2.0, -3.0]],
            [[0.2, 0.1, 0.0, -0.1, -0.2], [3.0, 1.0, 0.0, -1.0, -2.0]],
        ],
        dtype=torch.float64,
    )
    policy_logits = base_logits.clone()
    policy_logits[0, 0] = torch.tensor([5.1, 0.9, 0.0, -1.0, -2.0], dtype=torch.float64)
    policy_logits[0, 1] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 5.0], dtype=torch.float64)
    policy_logits[1, 0] = torch.tensor([0.2, 0.1, 0.0, -0.1, -0.2], dtype=torch.float64)
    policy_logits[1, 1] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 4.0], dtype=torch.float64)
    policy_logits.requires_grad_(True)

    responses = torch.tensor([[0, 3], [4, 4]])
    response_mask = torch.tensor([[True, True], [False, True]])
    epsilon = 0.05

    result = preservation_kl_loss(
        policy_logits,
        base_logits,
        epsilon_pres=epsilon,
        response_mask=response_mask,
        selected_token_ids=responses,
        top_k=1,
        default_probability=1e-4,
    )
    projection = result.projection
    sparse_raw_kl = projection.token_kl
    dense_raw_kl = token_kl_from_logits(policy_logits, base_logits)

    accepted = (sparse_raw_kl <= epsilon + 1e-6) & response_mask
    violating = (sparse_raw_kl > epsilon + 1e-6) & response_mask
    active_projected_kl = projection.projected_kl[response_mask]

    assert torch.all(result.token_loss[accepted] < 1e-10)
    assert torch.all(result.token_loss[violating] > 1e-5)
    assert torch.all(active_projected_kl <= epsilon + 1e-5)
    assert projection.support_mask is not None
    retained = projection.support_mask.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    assert torch.all(retained[response_mask])
    dropped_probs = projection.policy_log_probs.exp()[~projection.support_mask]
    assert dropped_probs.numel() > 0
    assert torch.all(dropped_probs > 0)
    assert torch.all(projection.eta[violating] > 0)
    assert not projection.eta.requires_grad
    assert not projection.projected_log_probs.requires_grad
    assert projection.policy_log_probs.requires_grad

    result.loss.backward()
    assert policy_logits.grad is not None
    assert torch.isfinite(policy_logits.grad).all()
    assert policy_logits.grad.norm() > 0

    dense_projection = project_to_kl_ball(policy_logits.detach(), base_logits, epsilon=epsilon)

    token_mean_result = preservation_kl_loss(
        policy_logits.detach(),
        base_logits,
        epsilon_pres=epsilon,
        response_mask=response_mask,
        selected_token_ids=responses,
        top_k=1,
        default_probability=1e-4,
        reduction="token-mean",
    )
    seq_lengths = response_mask.to(dtype=result.token_loss.dtype).sum(dim=-1)
    manual_seq_loss = (
        (result.token_loss.detach() * response_mask).sum(dim=-1) / seq_lengths.clamp_min(1.0)
    )
    manual_seq_loss = manual_seq_loss[seq_lengths > 0].mean()
    torch.testing.assert_close(result.loss.detach(), manual_seq_loss, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(result.loss.detach(), token_mean_result.loss.detach())

    fp16_policy_logits = policy_logits.detach().to(torch.float16).requires_grad_(True)
    fp16_result = preservation_kl_loss(
        fp16_policy_logits,
        base_logits.to(torch.float16),
        epsilon_pres=epsilon,
        response_mask=response_mask,
        selected_token_ids=responses,
        top_k=1,
        default_probability=1e-12,
    )
    assert torch.isfinite(fp16_result.loss)
    assert torch.isfinite(fp16_result.token_loss).all()
    assert torch.isfinite(fp16_result.projection.policy_log_probs).all()
    fp16_result.loss.backward()
    assert fp16_policy_logits.grad is not None
    assert torch.isfinite(fp16_policy_logits.grad).all()

    print(f"accepted_tokens={int(accepted.sum().item())}")
    print(f"violating_tokens={int(violating.sum().item())}")
    print(f"loss={result.loss.item():.8f}")
    print(f"token_mean_loss={token_mean_result.loss.item():.8f}")
    print("loss_reduction=seq-mean-token-mean")
    print(f"max_dense_raw_kl={dense_raw_kl[response_mask].max().item():.8f}")
    print(f"max_sparse_raw_kl={sparse_raw_kl[response_mask].max().item():.8f}")
    print(f"max_projected_kl={active_projected_kl.max().item():.8f}")
    print(f"max_eta={projection.eta[response_mask].max().item():.8f}")
    print(f"projection_target_stopgrad=True")
    print(f"dropped_default_probability={projection.default_probability:.1e}")
    print(f"fp16_default_probability_finite={torch.isfinite(fp16_result.loss).item()}")
    print(f"selected_tokens_retained={bool(torch.all(retained[response_mask]).item())}")
    print(f"dense_projection_has_sparse_support={dense_projection.support_mask is not None}")


if __name__ == "__main__":
    main()
