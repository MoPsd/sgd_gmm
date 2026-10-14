"""Adaptive coordinate-wise clipping SGD (ACClip)."""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class ACClip(Optimizer):
    """Euclidean ACClip (Zhang et al.).

    Momentum ``m_t``, adaptive threshold
    ``tau^alpha = beta2 tau^alpha + (1-beta2)|g|^alpha``.
    With ``mod=1``, the update is also scaled by ``1/sqrt(second_moment)``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-5,
        weight_decay: float = 0.0,
        alpha: float = 2.0,
        mod: int = 1,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 1.0 <= alpha <= 2.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        if mod not in (0, 1):
            raise ValueError("mod must be 0 or 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            mod=mod,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            alpha = group["alpha"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            mod = group["mod"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("ACClip does not support sparse gradients")

                if weight_decay != 0.0:
                    grad = grad.add(parameter, alpha=weight_decay)

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(parameter)
                    state["clip"] = torch.zeros_like(parameter)
                    state["second_moment"] = torch.zeros_like(parameter)

                state["step"] += 1
                momentum = state["momentum"]
                clip = state["clip"]
                second_moment = state["second_moment"]

                momentum.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                clip.mul_(beta2).add_(grad.abs().pow(alpha), alpha=1.0 - beta2)
                second_moment.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # ACCLip: min{ tau / (|m| + eps), 1 } * m
                denom = clip.pow(1.0 / alpha).div(momentum.abs().add(eps)).clamp(0.0, 1.0)
                if mod == 1:
                    denom = denom.div(second_moment.mul(beta2).sqrt().add(eps))

                parameter.addcmul_(denom, momentum, value=-lr)

        return loss
