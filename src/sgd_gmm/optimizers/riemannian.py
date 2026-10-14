"""Riemannian ACClip on the product of SO(n) and Euclidean spaces."""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from sgd_gmm.manifold import project_tangent, retract


class RiemannianACClip(Optimizer):
    """ACClip with a Stiefel/orthogonal update for ``U``.

    Momentum and clipping are computed in ambient space; for parameters in the
    ``stiefel`` group the clipped direction is projected to the tangent space
    and applied with a retraction. Remaining parameters use a Euclidean step.
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
        retraction: str = "qr",
        projection: str = "half",
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
        if retraction not in ("qr", "polar", "svd", "cayley"):
            raise ValueError("retraction must be 'qr', 'polar'/'svd', or 'cayley'")
        if projection not in ("half", "simple"):
            raise ValueError("projection must be 'half' or 'simple'")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            mod=mod,
            retraction=retraction,
            projection=projection,
            manifold="euclidean",
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
            manifold = group.get("manifold", "euclidean")
            retraction = group["retraction"]
            projection = group["projection"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("RiemannianACClip does not support sparse gradients")

                if weight_decay != 0.0 and manifold != "stiefel":
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

                scale = clip.pow(1.0 / alpha).div(momentum.abs().add(eps)).clamp(0.0, 1.0)
                clipped = scale * momentum
                if mod == 1:
                    clipped = clipped.div(second_moment.mul(beta2).sqrt().add(eps))

                if manifold == "stiefel":
                    tangent = project_tangent(parameter, clipped, mode=projection)
                    parameter.copy_(
                        retract(parameter, tangent, lr, kind=retraction)
                    )
                else:
                    parameter.add_(clipped, alpha=-lr)

        return loss


def split_gmm_param_groups(
    model,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.99),
    **kwargs,
) -> list[dict]:
    """Build optimizer groups with ``factor`` marked as Stiefel."""
    stiefel = {"params": [model.factor], "manifold": "stiefel", "lr": lr, "betas": betas}
    stiefel.update(kwargs)
    euclidean_params = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "factor"
    ]
    euclidean = {
        "params": euclidean_params,
        "manifold": "euclidean",
        "lr": lr,
        "betas": betas,
    }
    euclidean.update(kwargs)
    return [stiefel, euclidean]
