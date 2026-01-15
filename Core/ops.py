

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


def cross_entropy(
    logits: Tensor,
    targets: np.ndarray,
    axis: int = 1,
    mask: Optional[np.ndarray] = None,
) -> Tensor:
    """Categorical cross-entropy using a one-hot projection (no gather op).

    Parameters
    ----------
    logits : Tensor
        Raw (unnormalized) scores of shape (N, C).
    targets : np.ndarray
        Integer class ids of shape (N,).
    axis : int
        Class axis (default 1).
    mask : Optional[np.ndarray]
        Optional mask for per-sample weighting (shape must match N).

    Returns
    -------
    Tensor
        Scalar loss (mean over N or masked mean) with a connected autograd graph.

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
    nll = (Tensor(Y) * (-logp)).sum(axis=axis)  # (N,)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.float64)
        if mask_arr.shape != nll.data.shape:
            raise ValueError("mask shape must match per-sample loss shape")
        denom = float(mask_arr.sum())
        if denom <= 0.0:
            raise ValueError("mask must select at least one element")
        mask_t = Tensor(mask_arr)
        return (nll * mask_t).sum() * (1.0 / denom)
    return nll.mean()  # scalar


# ---------------------------------------------------------------------------
# Metrics (pure numpy helpers)
# ---------------------------------------------------------------------------

def accuracy(logits: Tensor, targets: np.ndarray) -> float:
    """Return numpy accuracy given logits (N, C) and integer targets (N,)."""
    preds = np.argmax(logits.data, axis=1)
    return float((preds == targets).mean())


def binary_cross_entropy(
    probs: Tensor,
    targets: np.ndarray,
    weight: Optional[np.ndarray] = None,
    normalize_by_weight: bool = False,
) -> Tensor:
    """Binary cross-entropy loss for probabilities (same shape as targets)."""
    if probs.data.shape != targets.shape:
        raise ValueError("targets shape must match probs shape")
    t = Tensor(targets.astype(np.float64))
    loss = -(t * probs.log() + (1.0 - t) * (1.0 - probs).log())
    if weight is not None:
        w = Tensor(weight.astype(np.float64))
        loss = loss * w
        if normalize_by_weight:
            denom = float(w.sum().data)
            if denom <= 0.0:
                raise ValueError("weight must select at least one element")
            return loss.sum() * (1.0 / denom)
    return loss.mean()


def binary_cross_entropy_with_logits(
    logits: Tensor,
    targets: np.ndarray,
    weight: Optional[np.ndarray] = None,
    pos_weight: Optional[float] = None,
    normalize_by_weight: bool = False,
) -> Tensor:
    """Binary cross-entropy on logits (applies sigmoid internally)."""
    if pos_weight is not None:
        w = np.ones_like(targets, dtype=np.float64)
        w = w + (pos_weight - 1.0) * targets
        weight = w if weight is None else weight * w
    return binary_cross_entropy(
        logits.sigmoid(),
        targets,
        weight=weight,
        normalize_by_weight=normalize_by_weight,
    )


__all__ = [
    "softmax",
    "log_softmax",
    "cross_entropy",
    "one_hot",
    "accuracy",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
]
