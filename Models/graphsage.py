

"""
GraphSAGE (mean aggregator) implemented with GraphFlow's minimal Tensor.

This module provides:
- SAGEConv: mean aggregation D^{-1}(A+I) X W with optional ReLU
- GraphSAGE: two-layer node representation model for classification
- sgd: tiny in-place optimizer that clears grads

References
---------
Hamilton, Ying, and Leskovec. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from Core.tensor import Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def row_normalize(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """Row-normalize an adjacency matrix (mean aggregator).

    Returns \tilde{A} = D^{-1}(A + I), where D is the out-degree (row sum).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    A_ = A.astype(np.float64, copy=True)
    if add_self_loops:
        A_ = A_ + np.eye(A_.shape[0], dtype=A_.dtype)
    deg = A_.sum(axis=1, keepdims=True)
    # avoid division by zero
    deg[deg == 0.0] = 1.0
    return A_ / deg


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

@dataclass
class Linear:
    in_dim: int
    out_dim: int
    seed: Optional[int] = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        limit = np.sqrt(6.0 / (self.in_dim + self.out_dim))
        self.W = Tensor(rng.uniform(-limit, limit, size=(self.in_dim, self.out_dim)), requires_grad=True, name="W")
        self.b = Tensor(np.zeros((self.out_dim,)), requires_grad=True, name="b")

    def __call__(self, X: Tensor) -> Tensor:
        return X.matmul(self.W) + self.b

    @property
    def params(self) -> List[Tensor]:
        return [self.W, self.b]


class SAGEConv:
    """GraphSAGE convolution with mean aggregation.

    Computes: H' = \sigma( D^{-1}(A+I) H W + b )
    """

    def __init__(self, in_dim: int, out_dim: int, A: np.ndarray, seed: Optional[int] = None):
        self.lin = Linear(in_dim, out_dim, seed=seed)
        self.A_row = row_normalize(A, add_self_loops=True)

    def __call__(self, X: Tensor, activation: bool = True) -> Tensor:
        # Aggregate neighbors (mean)
        AX = Tensor(self.A_row).matmul(X)
        Z = self.lin(AX)
        return Z.relu() if activation else Z

    @property
    def params(self) -> List[Tensor]:
        return self.lin.params


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphSAGE:
    """Two-layer GraphSAGE with mean aggregation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, A: np.ndarray, seed: int = 0):
        self.conv1 = SAGEConv(in_dim, hidden_dim, A, seed=seed)
        self.conv2 = SAGEConv(hidden_dim, out_dim, A, seed=seed + 1)

    def __call__(self, X: np.ndarray) -> Tensor:
        X_t = Tensor(X)
        h = self.conv1(X_t, activation=True)
        out = self.conv2(h, activation=False)
        return out

    @property
    def params(self) -> List[Tensor]:
        return self.conv1.params + self.conv2.params


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def sgd(params: List[Tensor], lr: float = 0.05):
    for p in params:
        if p.grad is not None:
            p.data -= lr * p.grad
            p.grad = None