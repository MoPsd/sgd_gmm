"""Penalizers from Section 2.1 of the paper (eqs. 4–7)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PriorStats:
    """Data-dependent constants for the Wishart / Gamma / mean priors."""

    mean: torch.Tensor
    scale_matrix_inv: torch.Tensor
    gamma_scale: torch.Tensor
    shrink: float = 0.01
    dirichlet_concentration: float = 0.99

    @classmethod
    def from_data(cls, data: torch.Tensor, n_components: int) -> "PriorStats":
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("prior stats need data of shape (N, D) with N >= 2")
        n_rows, dimension = data.shape
        mean = data.mean(dim=0)
        centered = data - mean
        covariance = (centered.T @ centered) / (n_rows - 1)
        covariance = 0.5 * (covariance + covariance.T)
        # S = Cov / K^{2/n}  (paper §2.1)
        scale = float(n_components) ** (2.0 / dimension)
        scale_matrix = covariance / scale
        # Stabilize the Wishart scale inverse on near-singular whitened data.
        eye = torch.eye(dimension, dtype=data.dtype, device=data.device)
        scale_matrix = scale_matrix + 1e-6 * eye
        scale_matrix_inv = torch.linalg.inv(scale_matrix)
        gamma_scale = torch.diagonal(covariance).sum() / (dimension * scale)
        return cls(
            mean=mean,
            scale_matrix_inv=scale_matrix_inv,
            gamma_scale=gamma_scale,
        )


def wishart_on_factor(factor: torch.Tensor, stats: PriorStats) -> torch.Tensor:
    """NLL term ψ1(U U^T) from eq. (4)."""
    gram = factor @ factor.transpose(-1, -2)
    _sign, log_det = torch.linalg.slogdet(gram)
    return 0.5 * torch.trace(gram @ stats.scale_matrix_inv) - 0.5 * log_det


def gamma_on_diagonals(diagonals: torch.Tensor, stats: PriorStats) -> torch.Tensor:
    """NLL term ψ2({D_k}) from eq. (5). ``diagonals`` has shape ``(K, D)``."""
    dimension = diagonals.shape[-1]
    return (
        0.5 * stats.gamma_scale * diagonals.sum()
        - 0.5 * dimension * diagonals.log().sum()
    )


def gaussian_on_means(
    means: torch.Tensor,
    precisions: torch.Tensor,
    stats: PriorStats,
) -> torch.Tensor:
    """NLL term ψ3({μ_k}) from eq. (6).

    ``means`` is ``(K, D)`` and ``precisions`` is ``(K, D, D)``.
    """
    n_components, dimension = means.shape
    kappa = stats.shrink
    centered = means - stats.mean
    # (μ - μ_p)^T P (μ - μ_p)
    quadratic = torch.einsum("ki,kij,kj->k", centered, precisions, centered)
    _sign, log_det = torch.linalg.slogdet(precisions)
    # 1/2 log det(κ / (2π) P) = 1/2 ( D log(κ/(2π)) + log det P )
    log_const = dimension * math.log(kappa / (2.0 * math.pi))
    return (0.5 * kappa * quadratic + 0.5 * (log_const + log_det)).sum()


def dirichlet_on_logits(logits: torch.Tensor, concentration: float = 0.99) -> torch.Tensor:
    """Regularizer φ(α) from eq. (7)."""
    n_components = logits.shape[0]
    return n_components * concentration * torch.logsumexp(logits, dim=0) - concentration * logits.sum()


def total_regularizer(
    factor: torch.Tensor,
    diagonals: torch.Tensor,
    means: torch.Tensor,
    precisions: torch.Tensor,
    logits: torch.Tensor,
    stats: PriorStats,
) -> torch.Tensor:
    """Combined regularizer ψ(U, D, μ) + φ(α)."""
    return (
        wishart_on_factor(factor, stats)
        + gamma_on_diagonals(diagonals, stats)
        + gaussian_on_means(means, precisions, stats)
        + dirichlet_on_logits(logits, stats.dirichlet_concentration)
    )
