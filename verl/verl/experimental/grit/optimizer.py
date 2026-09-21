"""Checkpoint-compatible AdamW with projection of the preconditioned task direction.

The initial backend supports one unsharded FSDP1 actor (GPU 0). Projectors
stay on CPU and are transferred one module at a time in FP32.
"""

import torch
from torch import nn


class ProjectedAdamW(torch.optim.AdamW):
    def __init__(self, model, *, module_pattern="mlp", **kwargs):
        super().__init__(model.parameters(), **kwargs)
        self.projectors = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module_pattern in name:
                projector = getattr(module, "proj_w", None)
                if projector is None:
                    raise ValueError(f"Missing projector on {name}")
                if projector.dtype != torch.float32 or not torch.isfinite(projector).all():
                    raise ValueError(f"Expected finite FP32 projector on {name}")
                if tuple(projector.shape) != (module.in_features, module.in_features):
                    raise ValueError(f"Projector shape mismatch on {name}")
                self.projectors[module.weight] = projector.detach().cpu()
        if not self.projectors:
            raise ValueError("Projection-only training requires attached MLP projectors")
        self.last_metrics = {}
        self.grad_scaler = None
        # Param-group metadata is included in the ordinary optimizer checkpoint.
        self.param_groups[0].setdefault("grit_policy_version", 0)

    def load_state_dict(self, state_dict):
        groups = state_dict.get("param_groups", [])
        if not groups or "grit_policy_version" not in groups[0]:
            raise ValueError("Resume requires a ProjectedAdamW checkpoint, not a legacy gradient-projection optimizer")
        return super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("ProjectedAdamW does not support closures")
        active = [(p, group) for group in self.param_groups for p in group["params"] if p.grad is not None]
        # Validate the whole step before mutating any moments or weights.
        for p, group in active:
            if p.dtype != torch.float32 or p.grad.is_sparse:
                raise ValueError("ProjectedAdamW requires dense gradients and FP32 master parameters")
            if group.get("amsgrad") or group.get("maximize") or group.get("capturable") or group.get("differentiable"):
                raise ValueError("Unsupported AdamW option in projection-only mode")
            if p in self.projectors and (p.ndim != 2 or p.shape[1] != self.projectors[p].shape[0]):
                raise ValueError("Projection requires full 2D weights; sharded/flattened parameters are unsupported")
        if not all(bool(torch.isfinite(p.grad).all()) for p, _ in active):
            self.last_metrics = {"grit/nonfinite_skipped": 1.0, "grit/optimizer_updates": 0.0}
            return
        count = sum(p in self.projectors for p, _ in active)
        if count == 0:
            raise RuntimeError("No protected module has a task gradient")
        before_sq = after_sq = 0.0
        for p, group in active:
            grad = p.grad.detach().float()
            state = self.state[p]
            if not state:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"].add_(1)
            step = int(state["step"].item())
            beta1, beta2 = group["betas"]
            m, v = state["exp_avg"], state["exp_avg_sq"]
            m.lerp_(grad, 1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denominator = v.sqrt().div_((1 - beta2**step) ** 0.5).add_(group["eps"])
            direction = m.div(1 - beta1**step).div_(denominator).neg_()
            if group["weight_decay"]:
                direction.add_(p, alpha=-group["weight_decay"])
            if p in self.projectors:
                before_sq += float(direction.square().sum())
                projector = self.projectors[p].to(device=p.device, dtype=torch.float32)
                direction = direction @ projector
                after_sq += float(direction.square().sum())
                del projector
            p.add_(direction, alpha=group["lr"])
        self.param_groups[0]["grit_policy_version"] += 1
        self.last_metrics = {
            "grit/attached_projector_count": float(len(self.projectors)),
            "grit/projected_module_count": float(count),
            "grit/adamw_task_direction_norm": before_sq**0.5,
            "grit/adamw_projected_direction_norm": after_sq**0.5,
            "grit/nonfinite_skipped": 0.0,
            "grit/optimizer_updates": 1.0,
            "grit/policy_version": float(self.param_groups[0]["grit_policy_version"]),
        }
