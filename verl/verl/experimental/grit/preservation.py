"""GRIT Phase 3 preservation helpers for verl actor workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from grit.preservation_loss import PreservationLossResult, preservation_kl_loss
from verl import DataProto


PRESERVATION_TENSOR_KEYS = (
    "responses",
    "response_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
)


@dataclass
class PreservationBranchResult:
    """Loss and metrics produced by one preservation micro-batch."""

    loss: torch.Tensor
    metrics: dict[str, float]
    raw: PreservationLossResult


class TensorizedPreservationDataset:
    """Cyclic tensorized ``D_preserve`` provider for actor-local training."""

    def __init__(self, data: DataProto):
        missing = [key for key in PRESERVATION_TENSOR_KEYS if key not in data.batch.keys()]
        if missing:
            raise ValueError(f"D_preserve is missing required tensor keys: {missing}")
        if len(data) == 0:
            raise ValueError("D_preserve must contain at least one row")
        self.data = data
        self.cursor = 0

    @classmethod
    def from_path(cls, path: str | Path) -> "TensorizedPreservationDataset":
        path = Path(path)
        if path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, DataProto):
                return cls(payload)
            if isinstance(payload, Mapping) and "tensors" in payload:
                return cls(DataProto.from_dict(tensors=dict(payload["tensors"])))
            if isinstance(payload, Mapping):
                tensor_payload = {key: value for key, value in payload.items() if torch.is_tensor(value)}
                return cls(DataProto.from_dict(tensors=tensor_payload))
            raise TypeError(f"unsupported D_preserve .pt payload type: {type(payload).__name__}")
        if path.suffix == ".parquet":
            import pandas as pd

            frame = pd.read_parquet(path)
            tensors = {key: _series_to_tensor(frame[key], key) for key in PRESERVATION_TENSOR_KEYS}
            return cls(DataProto.from_dict(tensors=tensors))
        raise ValueError(f"unsupported D_preserve format: {path.suffix}; expected .pt or .parquet")

    def next_batch(self, batch_size: int) -> DataProto:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        remaining = batch_size
        chunks: list[DataProto] = []
        while remaining > 0:
            take = min(remaining, len(self.data) - self.cursor)
            chunks.append(self.data[self.cursor : self.cursor + take])
            self.cursor = (self.cursor + take) % len(self.data)
            remaining -= take
        return chunks[0] if len(chunks) == 1 else DataProto.concat(chunks)


def _series_to_tensor(series, key: str) -> torch.Tensor:
    values = series.to_list()
    tensor = torch.as_tensor(np.asarray(values))
    if key in {"responses", "input_ids", "attention_mask", "position_ids"}:
        return tensor.long()
    return tensor.float()


def build_frozen_base_model(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Load frozen ``pi_base`` from a Hugging Face causal LM checkpoint."""

    from transformers import AutoModelForCausalLM

    kwargs = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    base_model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()
    if device is not None:
        base_model.to(device)
    return base_model


def compute_preservation_branch_loss(
    policy_logits: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    epsilon_pres: float,
    top_k: int | None,
    default_probability: float,
    reduction: str,
) -> PreservationBranchResult:
    """Compute GRIT Phase 3 loss and standard verl metrics."""

    result = preservation_kl_loss(
        policy_logits,
        base_logits,
        epsilon_pres=epsilon_pres,
        response_mask=response_mask,
        selected_token_ids=responses,
        top_k=top_k,
        default_probability=default_probability,
        reduction=reduction,
    )
    active_mask = result.projection.active_mask
    violation_mask = result.projection.violation_mask
    active_count = active_mask.sum().clamp_min(1)
    violation_fraction = violation_mask.sum().to(dtype=torch.float32) / active_count.to(dtype=torch.float32)
    metrics = {
        "grit/preservation_loss": float(result.loss.detach().float().item()),
        "grit/kl_violation_fraction": float(violation_fraction.detach().item()),
    }
    return PreservationBranchResult(loss=result.loss, metrics=metrics, raw=result)
