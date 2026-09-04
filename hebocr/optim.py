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
