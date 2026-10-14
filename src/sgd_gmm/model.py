"""Flexibly-tied Gaussian mixture from the paper (eqs. 2–11)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from sgd_gmm.manifold import orthogonality_error, qr_retract
from sgd_gmm.regularizers import PriorStats, total_regularizer


def softplus_inv(positive: torch.Tensor) -> torch.Tensor:
    """Inverse of SoftPlus for ``omega = 1``: ``log(exp(x) - 1)``."""
    return torch.log(positive.expm1().clamp_min(1e-12))


def _symmetric_eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigen-decomposition with a CPU fallback when the device lacks ``eigh``."""
    try:
        return torch.linalg.eigh(matrix)
    except NotImplementedError:
        values, vectors = torch.linalg.eigh(matrix.cpu())
        return values.to(device=matrix.device), vectors.to(device=matrix.device)


class FlexiblyTiedGMM(nn.Module):
    """Mixture with shared factor ``U`` and per-component diagonal ``D_k``.

    Precision of component ``k``:

        P_k = lambda_k * U * diag(d_k) * U^T

    ``d_k`` is SoftPlus of unconstrained parameters. ``lambda_k`` is stored as
    ``log_lambda``. Mixture weights are SoftMax of logits ``alpha``.

    When ``orthogonal=True``, training keeps ``U`` on ``SO(n)`` and the
    log-determinant reduces to ``n log lambda + sum log d``. When
    ``orthogonal=False``, ``U`` is free and ``log|det U|`` is included.
    """

    def __init__(
        self,
        dimension: int,
        n_components: int,
        *,
        orthogonal: bool = True,
        softplus_beta: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if n_components < 1:
            raise ValueError("n_components must be positive")

        self.dimension = dimension
        self.n_components = n_components
        self.orthogonal = orthogonal
        self.softplus_beta = softplus_beta

        self.alpha = nn.Parameter(torch.zeros(n_components, dtype=dtype, device=device))
        self.means = nn.Parameter(torch.zeros(n_components, dimension, dtype=dtype, device=device))
        self.diag_raw = nn.Parameter(torch.zeros(n_components, dimension, dtype=dtype, device=device))
        self.log_lambda = nn.Parameter(torch.zeros(n_components, dtype=dtype, device=device))
        self.factor = nn.Parameter(torch.eye(dimension, dtype=dtype, device=device))
        self.register_buffer("_n_train", torch.tensor(1.0, dtype=dtype, device=device))
        self._prior: PriorStats | None = None

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.alpha, dim=0)

    def diagonals(self) -> torch.Tensor:
        return F.softplus(self.diag_raw, beta=self.softplus_beta)

    def lambdas(self) -> torch.Tensor:
        return self.log_lambda.exp()

    def precisions(self) -> torch.Tensor:
        """Full precision matrices, shape ``(K, D, D)``."""
        scale = (self.lambdas().unsqueeze(-1) * self.diagonals()).sqrt()
        scaled = self.factor.unsqueeze(0) * scale.unsqueeze(-2)
        return scaled @ scaled.transpose(-1, -2)

    def covariances(self) -> torch.Tensor:
        return torch.linalg.inv(self.precisions())

    def set_prior(self, stats: PriorStats, n_train: int) -> None:
        self._prior = stats
        self._n_train = torch.as_tensor(
            float(n_train),
            dtype=self.means.dtype,
            device=self.means.device,
        )

    @torch.no_grad()
    def initialize(self, data: torch.Tensor) -> None:
        """Eigenbasis of the sample covariance plus a random partition."""
        if data.ndim != 2 or data.shape[1] != self.dimension:
            raise ValueError(
                f"expected data of shape (N, {self.dimension}), got {tuple(data.shape)}"
            )
        data = data.to(dtype=self.means.dtype, device=self.means.device)
        n_rows = data.shape[0]
        if n_rows < 2:
            raise ValueError("initialization needs at least two observations")

        stats = PriorStats.from_data(data, self.n_components)
        self.set_prior(stats, n_rows)

        labels = torch.randint(0, self.n_components, (n_rows,), device=data.device)
        centered = data - data.mean(dim=0, keepdim=True)
        covariance = (centered.T @ centered) / (n_rows - 1)
        covariance = 0.5 * (covariance + covariance.T)
        _eigenvalues, eigenvectors = _symmetric_eigh(covariance)
        if self.orthogonal:
            eigenvectors = qr_retract(eigenvectors)
        self.factor.copy_(eigenvectors)

        counts = torch.zeros(self.n_components, dtype=data.dtype, device=data.device)
        global_var = centered.square().mean(dim=0).clamp_min(1e-4)
        for index in range(self.n_components):
            mask = labels == index
            count = int(mask.sum().item())
            counts[index] = count
            if count == 0:
                self.means[index].copy_(data[torch.randint(0, n_rows, (1,), device=data.device)])
                projected_var = global_var
            else:
                chunk = data[mask]
                mean = chunk.mean(dim=0)
                self.means[index].copy_(mean)
                if count == 1:
                    projected_var = global_var
                else:
                    local = chunk - mean
                    projected = local @ eigenvectors
                    projected_var = projected.square().mean(dim=0).clamp_min(1e-4)
            # P = lambda * U diag(d) U^T with lambda=1 initially.
            self.diag_raw[index].copy_(softplus_inv(1.0 / projected_var))
            self.log_lambda[index].zero_()

        frequencies = counts.clamp_min(1.0)
        frequencies = frequencies / frequencies.sum()
        self.alpha.copy_(frequencies.log())

    def retract(self, kind: str = "qr") -> None:
        """Replace ``U`` by a retraction onto the orthogonal group."""
        if self.orthogonal:
            from sgd_gmm.manifold import retract as manifold_retract

            self.factor.copy_(manifold_retract(self.factor.data, kind=kind))

    def component_log_prob(self, data: torch.Tensor) -> torch.Tensor:
        """Per-component log density, shape ``(N, K)``."""
        if data.ndim != 2 or data.shape[1] != self.dimension:
            raise ValueError(
                f"expected data of shape (N, {self.dimension}), got {tuple(data.shape)}"
            )
        diagonals = self.diagonals()
        lambdas = self.lambdas()
        centered = data.unsqueeze(1) - self.means.unsqueeze(0)
        rotated = centered @ self.factor
        quadratic = 0.5 * lambdas * (rotated.square() * diagonals).sum(dim=-1)

        log_det = 0.5 * (
            self.dimension * self.log_lambda + diagonals.log().sum(dim=-1)
        )
        if not self.orthogonal:
            _sign, log_abs_det = torch.linalg.slogdet(self.factor)
            log_det = log_det + log_abs_det

        log_norm = -0.5 * self.dimension * math.log(2.0 * math.pi)
        return log_norm + log_det - quadratic

    def log_prob(self, data: torch.Tensor) -> torch.Tensor:
        responsibilities = self.component_log_prob(data) + F.log_softmax(self.alpha, dim=0)
        return torch.logsumexp(responsibilities, dim=-1)

    def data_nll(self, data: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(data).mean()

    def regularizer(self) -> torch.Tensor:
        if self._prior is None:
            return torch.zeros((), dtype=self.means.dtype, device=self.means.device)
        return total_regularizer(
            self.factor,
            self.diagonals(),
            self.means,
            self.precisions(),
            self.alpha,
            self._prior,
        )

    def nll(self, data: torch.Tensor, *, with_regularizer: bool = False) -> torch.Tensor:
        """Mean batch NLL, optionally plus ``ψ/n_train`` (problem (8))."""
        loss = self.data_nll(data)
        if with_regularizer and self._prior is not None:
            loss = loss + self.regularizer() / self._n_train
        return loss

    def orthogonality_error(self) -> torch.Tensor:
        return orthogonality_error(self.factor)
