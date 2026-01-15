"""
Structure-learning GCN with a differentiable adjacency matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from Core.tensor import Tensor
from Core.structure import LearnableAdjacency, normalize_adjacency_tensor


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


class StructureGCN:
    """Two-layer GCN that learns adjacency end-to-end."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_nodes: int,
        seed: int = 0,
        init_adj: Optional[np.ndarray] = None,
    ):
        self.adj = LearnableAdjacency(n_nodes, seed=seed, init_adj=init_adj)
        self.lin1 = Linear(in_dim, hidden_dim, seed=seed + 1)
        self.lin2 = Linear(hidden_dim, out_dim, seed=seed + 2)

    def __call__(self, X: np.ndarray) -> Tensor:
        X_t = Tensor(X)
        A = self.adj.adjacency()
        A_norm = normalize_adjacency_tensor(A, add_self_loops=True)
        h = A_norm.matmul(X_t)
        h = self.lin1(h).relu()
        h = A_norm.matmul(h)
        out = self.lin2(h)
        return out

    @property
    def params(self) -> List[Tensor]:
        return self.adj.params + self.lin1.params + self.lin2.params


__all__ = ["StructureGCN"]
