"""Geometry of the special orthogonal group for the shared factor ``U``."""

from __future__ import annotations

import torch


def skew(matrix: torch.Tensor) -> torch.Tensor:
    """Return ``Y - Y^T`` (twice the skew-symmetric part of ``Y``)."""
    return matrix - matrix.transpose(-1, -2)


def project_tangent(
    factor: torch.Tensor,
    euclidean_grad: torch.Tensor,
    *,
    mode: str = "half",
) -> torch.Tensor:
    """Project a Euclidean gradient of ``U`` onto ``T_U SO(n)``.

    Paper eq. (12):

    * ``mode='simple'``: ``A = Skew(G U^T)``, then ``Proj = A U``.
    * ``mode='half'`` (default): ``A = Skew([I - 1/2 U U^T] G U^T)``.
    """
    if mode == "simple":
        aux = euclidean_grad @ factor.transpose(-1, -2)
    elif mode == "half":
        eye = torch.eye(factor.shape[-1], dtype=factor.dtype, device=factor.device)
        aux = (eye - 0.5 * (factor @ factor.transpose(-1, -2))) @ (
            euclidean_grad @ factor.transpose(-1, -2)
        )
    else:
        raise ValueError("mode must be 'simple' or 'half'")
    return skew(aux) @ factor


def qr_retract(matrix: torch.Tensor) -> torch.Tensor:
    """QR retraction with a positive diagonal of ``R``."""
    q_factor, r_factor = torch.linalg.qr(matrix)
    diagonal = torch.diag(r_factor)
    signs = torch.sign(diagonal)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q_factor * signs.unsqueeze(0)


def polar_retract(matrix: torch.Tensor) -> torch.Tensor:
    """Polar retraction: orthogonal factor of the SVD of ``Y``."""
    left, _singular, right = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right


def cayley_retract(
    factor: torch.Tensor,
    tangent: torch.Tensor,
    step_size: float | torch.Tensor,
    *,
    n_iters: int = 5,
) -> torch.Tensor:
    """Iterative Cayley retraction along a tangent vector."""
    w_hat = tangent @ factor.transpose(-1, -2)
    w_matrix = 0.5 * skew(w_hat)
    y_new = factor + step_size * tangent
    for _ in range(n_iters):
        y_new = factor + step_size * (w_matrix @ (0.5 * (factor + y_new)))
    return y_new


def retract(
    factor: torch.Tensor,
    tangent: torch.Tensor | None = None,
    step_size: float | torch.Tensor = 1.0,
    *,
    kind: str = "qr",
) -> torch.Tensor:
    """Retraction onto the orthogonal group (``qr``, ``polar``/``svd``, or ``cayley``)."""
    if kind == "qr":
        if tangent is None:
            return qr_retract(factor)
        return qr_retract(factor - step_size * tangent)
    if kind in ("polar", "svd"):
        if tangent is None:
            return polar_retract(factor)
        return polar_retract(factor - step_size * tangent)
    if kind == "cayley":
        if tangent is None:
            raise ValueError("cayley retraction needs a tangent vector")
        return cayley_retract(factor, tangent, -step_size)
    raise ValueError("kind must be 'qr', 'polar'/'svd', or 'cayley'")


def orthogonality_error(matrix: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of ``U^T U - I``."""
    dimension = matrix.shape[-1]
    eye = torch.eye(dimension, dtype=matrix.dtype, device=matrix.device)
    return torch.linalg.matrix_norm(matrix.T @ matrix - eye)
