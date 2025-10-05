

"""
Core tensor operations for GraphFlow.

This module contains numerically-stable softmax, log-softmax, and
cross-entropy implemented to work with the lightweight Tensor autograd
engine in `Core/tensor.py`.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .tensor import Tensor

# ---------------------------------------------------------------------------
# Softmax family
# ---------------------------------------------------------------------------

def softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """Stable softmax along a given axis.

    Notes
    -----
    We subtract the per-row max as a constant (no gradient through the max),
    which is fine because softmax is translation-invariant.
    """
    # compute max along axis as a constant tensor (no grad needed)
    m = np.max(logits.data, axis=axis, keepdims=True)
    exps = (logits - Tensor(m)).exp()
    sums = exps.sum(axis=axis, keepdims=True)
    return exps / sums


def log_softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """Compute log-softmax via softmax then log (keeps autograd graph intact)."""
    return softmax(logits, axis=axis).log()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def one_hot(targets: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Return a one-hot numpy array for integer class targets (shape: N, C)."""
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D array of class indices")
    if num_classes is None:
        num_classes = int(np.max(targets)) + 1
    N = targets.shape[0]
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), targets] = 1.0
    return Y


def cross_entropy(logits: Tensor, targets: np.ndarray, axis: int = 1) -> Tensor:
    """Categorical cross-entropy using a one-hot projection (no gather op).

    Parameters
    ----------
    logits : Tensor
        Raw (unnormalized) scores of shape (N, C).
    targets : np.ndarray
        Integer class ids of shape (N,).
    axis : int
        Class axis (default 1).

    Returns
    -------
    Tensor
        Scalar loss (mean over N) with a connected autograd graph.

    Notes
    -----
    We avoid advanced indexing on Tensor (not implemented) by multiplying
    one-hot labels with log-softmax and summing.
    """
    if logits.data.ndim != 2:
        raise ValueError("logits must be 2D (N, C)")
    N, C = logits.data.shape
    if targets.shape[0] != N:
        raise ValueError("targets length must match batch size N")

    Y = one_hot(targets, num_classes=C)  # (N, C) numpy
    logp = log_softmax(logits, axis=axis)  # (N, C) Tensor
    # negative log-likelihood: -sum_y y * log p
    nll = (Tensor(Y) * (-logp)).sum(axis=1)  # (N,)
    return nll.mean()  # scalar


# ---------------------------------------------------------------------------
# Metrics (pure numpy helpers)
# ---------------------------------------------------------------------------

def accuracy(logits: Tensor, targets: np.ndarray) -> float:
    """Return numpy accuracy given logits (N, C) and integer targets (N,)."""
    preds = np.argmax(logits.data, axis=1)
    return float((preds == targets).mean())


__all__ = [
    "softmax",
    "log_softmax",
    "cross_entropy",
    "one_hot",
    "accuracy",
]