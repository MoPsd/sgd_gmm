"""Flexibly-tied GMMs with Euclidean and Riemannian first-order learning."""

from sgd_gmm.model import FlexiblyTiedGMM
from sgd_gmm.train import heldout_nll, train

__all__ = [
    "FlexiblyTiedGMM",
    "heldout_nll",
    "train",
]
