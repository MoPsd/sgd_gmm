"""Mini-batch training for flexibly-tied GMMs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from sgd_gmm.model import FlexiblyTiedGMM
from sgd_gmm.optimizers.acclip import ACClip
from sgd_gmm.optimizers.riemannian import RiemannianACClip, split_gmm_param_groups

DIAG_RAW_LIMIT = 20.0
LOG_LAMBDA_LIMIT = 8.0


@dataclass
class TrainResult:
    """Mean training NLL after each epoch."""

    nll_per_epoch: list[float]

    @property
    def final_nll(self) -> float:
        return self.nll_per_epoch[-1]


def _build_optimizer(
    model: FlexiblyTiedGMM,
    method: str,
    learning_rate: float,
    momentum: float,
) -> Optimizer:
    if method == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    if method == "sgd":
        return SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if method == "acclip":
        return ACClip(model.parameters(), lr=learning_rate)
    if method == "riemannian_acclip":
        if not model.orthogonal:
            raise ValueError("riemannian_acclip requires model.orthogonal=True")
        return RiemannianACClip(split_gmm_param_groups(model, lr=learning_rate))
    raise ValueError(
        "method must be one of: adam, sgd, acclip, riemannian_acclip"
    )


def _build_scheduler(
    optimizer: Optimizer,
    *,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
    use_schedule: bool,
):
    if not use_schedule:
        return None
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(steps_per_epoch * warmup_epochs, 1)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=1e-7,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def train(
    model: FlexiblyTiedGMM,
    data: torch.Tensor,
    *,
    method: str = "adam",
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-2,
    momentum: float = 0.0,
    warmup_epochs: int = 1,
    use_schedule: bool = True,
    with_regularizer: bool = False,
    seed: int | None = None,
) -> TrainResult:
    """Minimize mini-batch negative log-likelihood.

    For Euclidean methods with ``model.orthogonal=True``, ``U`` is QR-retracted
    after each step. ``riemannian_acclip`` retracts inside the optimizer.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if data.ndim != 2:
        raise ValueError("data must have shape (N, D)")

    retract_after = method in ("adam", "sgd", "acclip") and model.orthogonal

    data = data.to(dtype=model.means.dtype, device=model.means.device)
    if model._prior is None:
        model.initialize(data)

    n_rows = data.shape[0]
    steps_per_epoch = max(math.ceil(n_rows / batch_size), 1)
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    optimizer = _build_optimizer(model, method, learning_rate, momentum)
    scheduler = _build_scheduler(
        optimizer,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        use_schedule=use_schedule,
    )
    history: list[float] = []

    model.train()
    for _epoch in range(epochs):
        order = torch.randperm(n_rows, generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, n_rows, batch_size):
            batch = data[order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = model.nll(batch, with_regularizer=with_regularizer)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                if retract_after and method != "riemannian_acclip":
                    model.retract(kind="qr")
                model.diag_raw.clamp_(-DIAG_RAW_LIMIT, DIAG_RAW_LIMIT)
                model.log_lambda.clamp_(-LOG_LAMBDA_LIMIT, LOG_LAMBDA_LIMIT)
            total += float(loss.detach().item()) * batch.shape[0]
            seen += batch.shape[0]
        history.append(total / seen)
    return TrainResult(nll_per_epoch=history)


def heldout_nll(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int = 256,
    *,
    with_regularizer: bool = False,
) -> float:
    """Mean negative log-likelihood without updating parameters."""
    data = data.to(dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    total = 0.0
    seen = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            batch = data[start : start + batch_size]
            if isinstance(model, FlexiblyTiedGMM):
                loss = model.nll(batch, with_regularizer=with_regularizer)
            else:
                loss = model.nll(batch)
            total += float(loss.item()) * batch.shape[0]
            seen += batch.shape[0]
    model.train(was_training)
    return total / seen
