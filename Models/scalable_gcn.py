"""
Scalable GCN using edge-index message-passing kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from Core.tensor import Tensor
from Core.edge_index import EdgeIndex, edge_index_from_adjacency, gcn_normalize_edge_index
from Core.kernels import EdgeIndexKernel


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


class ScalableGCN:
    """Two-layer GCN with cached edge-index kernels for scalable message passing."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        A: Optional[np.ndarray] = None,
        edge_index: Optional[EdgeIndex] = None,
        seed: int = 0,
        add_self_loops: bool = True,
    ):
        if edge_index is None:
            if A is None:
                raise ValueError("A or edge_index must be provided")
            edge_index = edge_index_from_adjacency(A, add_self_loops=add_self_loops)
        edge_index = gcn_normalize_edge_index(edge_index)
        self.kernel = EdgeIndexKernel(edge_index)
        self.lin1 = Linear(in_dim, hidden_dim, seed=seed)
        self.lin2 = Linear(hidden_dim, out_dim, seed=seed + 1)

    def __call__(self, X: np.ndarray) -> Tensor:
        X_t = Tensor(X)
        h = self.kernel.gcn(X_t)
        h = self.lin1(h).relu()
        h = self.kernel.gcn(h)
        out = self.lin2(h)
        return out

    @property
    def params(self) -> List[Tensor]:
        return self.lin1.params + self.lin2.params


__all__ = ["ScalableGCN"]
