

r"""
Graph Convolutional Network (GCN) layers and a tiny 2-layer GCN model
built on GraphFlow's minimal autograd Tensor.

This module provides:
- Linear: affine layer using Tensor
- GCNLayer: \hat A X W with optional ReLU
- GCN: two-layer node classifier/regressor (no training loop inside)
- sgd: minimal optimizer (clears grads)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from Core.tensor import Tensor
from Core.graph import normalize_adjacency

# ---------------------------------------------------------------------------
# Core layers
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


class GCNLayer:
    def __init__(self, in_dim: int, out_dim: int, A: np.ndarray, seed: Optional[int] = None):
        """One GCN layer with fixed (pre-normalized) adjacency.

        Parameters
        ----------
        in_dim, out_dim : int
            Input/output feature dimensions.
        A : np.ndarray
            Raw adjacency; we normalize internally with self-loops.
        seed : Optional[int]
            Seed for weight init.
        """
        self.lin = Linear(in_dim, out_dim, seed=seed)
        self.A_norm = normalize_adjacency(A, add_self_loops=True)

    def __call__(self, X: Tensor, activation: bool = True) -> Tensor:
        # message passing: \hat A X W
        AX = Tensor(self.A_norm).matmul(X)
        Z = self.lin(AX)
        return Z.relu() if activation else Z

    @property
    def params(self) -> List[Tensor]:
        return self.lin.params


class GCN:
    """Two-layer GCN.

    Example
    -------
    >>> model = GCN(in_dim=F, hidden_dim=16, out_dim=C, A=adj)
    >>> logits = model(X_np)   # returns Tensor (N, C)
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, A: np.ndarray, seed: int = 0):
        self.layer1 = GCNLayer(in_dim, hidden_dim, A, seed=seed)
        self.layer2 = GCNLayer(hidden_dim, out_dim, A, seed=seed + 1)

    def __call__(self, X: np.ndarray) -> Tensor:
        X_t = Tensor(X)
        h = self.layer1(X_t, activation=True)
        out = self.layer2(h, activation=False)
        return out

    @property
    def params(self) -> List[Tensor]:
        return self.layer1.params + self.layer2.params


# ---------------------------------------------------------------------------
# Minimal optimizer
# ---------------------------------------------------------------------------

def sgd(params: List[Tensor], lr: float = 0.05):
    """In-place SGD and gradient reset."""
    for p in params:
        if p.grad is not None:
            p.data -= lr * p.grad
            p.grad = None
