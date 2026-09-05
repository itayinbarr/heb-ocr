"""Sharpness-Aware Minimization.

SAM takes a step, looks at the loss at a nearby worst-case point, and uses the
gradient from *there*. It costs two forward/backward passes per step. HTR-VT
credits it, alongside span masking, for the model's behaviour on small
datasets, and this run is squarely in that regime: 150k synthetic lines with a
test set from a different visual domain entirely, where a sharp minimum on
synthetic data is exactly the failure to avoid.
"""

import torch


class SAM(torch.optim.Optimizer):
    """Wraps a base optimizer. Call `first_step`, recompute the loss, `second_step`."""

    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Climb to the worst-case point within an epsilon ball."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["e_w"] = e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                )
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Undo the climb, then apply the base optimizer's update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p].pop("e_w"))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        shared = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class ModelEMA:
    """An exponentially-moved average of the weights, evaluated alongside them.

    SGD with a cosine schedule still bounces around the minimum it is
    descending into; averaging recent weights lands nearer the centre of that
    basin than any single step does. It costs one extra copy of the model and no
    extra gradient computation, which makes it close to free here.

    The average is kept in float32 even when training runs in bf16, because the
    whole point is to accumulate small differences that bf16 would round away.
    """

    def __init__(self, model, decay: float = 0.9995, warmup_steps: int = 1000):
        import copy

        self.decay = decay
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.shadow = copy.deepcopy(model).eval().float()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _current_decay(self) -> float:
        """Ramp the decay in, so early averages are not dominated by noise."""
        if self.steps >= self.warmup_steps:
            return self.decay
        return min(self.decay, (1 + self.steps) / (10 + self.steps))

    @torch.no_grad()
    def update(self, model) -> None:
        decay = self._current_decay()
        self.steps += 1
        shadow_params = dict(self.shadow.named_parameters())
        for name, param in model.named_parameters():
            shadow_params[name].mul_(decay).add_(param.detach().float(), alpha=1 - decay)
        # Buffers (BatchNorm statistics) are copied rather than averaged: they
        # are already running averages, and averaging them twice lags the model.
        shadow_buffers = dict(self.shadow.named_buffers())
        for name, buffer in model.named_buffers():
            shadow_buffers[name].copy_(buffer.detach().float())

    def state_dict(self) -> dict:
        return self.shadow.state_dict()
