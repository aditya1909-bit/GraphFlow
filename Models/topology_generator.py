"""
Neural topology generator for graph structure prediction.
Supports pairwise edge scoring and optional guide message passing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from Core.edge_index import EdgeIndex, gcn_normalize_edge_index
from Core.kernels import EdgeIndexKernel, gather_rows
from Core.tensor import Tensor


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


class TopologyGenerator:
    """Generate adjacency logits from node features."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_nodes: Optional[int] = None,
        seed: int = 0,
        guide_edge_index: Optional[EdgeIndex] = None,
    ):
        self.lin1 = Linear(in_dim, hidden_dim, seed=seed)
        self.lin2 = Linear(hidden_dim, hidden_dim, seed=seed + 1)
        self.edge1 = Linear(hidden_dim, hidden_dim, seed=seed + 2)
        self.edge2 = Linear(hidden_dim, 1, seed=seed + 3)
        self.scale = Tensor(np.array([1.0], dtype=np.float64), requires_grad=True, name="scale")
        self.bias = Tensor(np.array([0.0], dtype=np.float64), requires_grad=True, name="bias")
        if n_nodes is not None:
            self.node_bias = Tensor(np.zeros((n_nodes, 1), dtype=np.float64), requires_grad=True, name="node_bias")
        else:
            self.node_bias = None
        self.guide_kernel = None
        if guide_edge_index is not None:
            guide_norm = gcn_normalize_edge_index(guide_edge_index)
            self.guide_kernel = EdgeIndexKernel(guide_norm)

    def __call__(self, X: np.ndarray, pairs: Optional[np.ndarray] = None) -> Tensor:
        X_t = Tensor(X)
        Z = self.lin1(X_t).relu()
        if self.guide_kernel is not None:
            Z = self.guide_kernel.gcn(Z).relu()
        Z = self.lin2(Z)  # allow negative values for non-edge logits
        if pairs is None:
            n = Z.data.shape[0]
            zi = Z.reshape((n, 1, -1)) * Tensor(np.ones((1, n, 1)))
            zj = Z.reshape((1, n, -1)) * Tensor(np.ones((n, 1, 1)))
            prod = zi * zj
            prod2d = prod.reshape((n * n, -1))
            h = self.edge1(prod2d).relu()
            logits2d = self.edge2(h)
            logits = logits2d.reshape((n, n))
        else:
            pairs = np.asarray(pairs, dtype=np.int64)
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("pairs must have shape (E, 2)")
            src = pairs[:, 0]
            dst = pairs[:, 1]
            zi = gather_rows(Z, src)
            zj = gather_rows(Z, dst)
            prod = zi * zj
            h = self.edge1(prod).relu()
            logits = self.edge2(h).reshape((pairs.shape[0],))
        scale = 1.0 / np.sqrt(float(Z.data.shape[1]))
        logits = logits * scale
        logits = logits * self.scale + self.bias
        if self.node_bias is not None:
            if pairs is None:
                logits = logits + self.node_bias + self.node_bias.T()
            else:
                bi = gather_rows(self.node_bias, src).reshape((pairs.shape[0],))
                bj = gather_rows(self.node_bias, dst).reshape((pairs.shape[0],))
                logits = logits + bi + bj
        return logits

    @property
    def params(self) -> List[Tensor]:
        ps = (
            self.lin1.params
            + self.lin2.params
            + self.edge1.params
            + self.edge2.params
            + [self.scale, self.bias]
        )
        if self.node_bias is not None:
            ps.append(self.node_bias)
        return ps


__all__ = ["TopologyGenerator"]
