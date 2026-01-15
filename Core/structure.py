"""Differentiable graph structure utilities for GraphFlow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .tensor import Tensor


def normalize_adjacency_tensor(A: Tensor, add_self_loops: bool = True, eps: float = 1e-12) -> Tensor:
    """Symmetric normalization using Tensor ops for differentiable structure learning."""
    if add_self_loops:
        I = Tensor(np.eye(A.data.shape[0], dtype=np.float64))
        A = A + I
    deg = A.sum(axis=1, keepdims=True)
    d_inv_sqrt = (deg + eps).pow(-0.5)
    return A * d_inv_sqrt * d_inv_sqrt.T()


def init_logits_from_adj(adj: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Initialize logits from a (0/1) adjacency with smoothing."""
    adj = adj.astype(np.float64)
    adj = np.clip(adj, 0.0, 1.0)
    adj = adj * (1.0 - 2.0 * eps) + eps
    return np.log(adj / (1.0 - adj))


@dataclass
class LearnableAdjacency:
    """Learnable adjacency via logits -> sigmoid -> symmetrize."""
    n_nodes: int
    seed: Optional[int] = None
    init_adj: Optional[np.ndarray] = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        if self.init_adj is None:
            init = 0.01 * rng.normal(size=(self.n_nodes, self.n_nodes))
        else:
            init = init_logits_from_adj(self.init_adj)
        self.logits = Tensor(init, requires_grad=True, name="adj_logits")

    def adjacency(self) -> Tensor:
        A = self.logits.sigmoid()
        A = (A + A.T()) * 0.5
        return A

    @property
    def params(self):
        return [self.logits]


__all__ = ["normalize_adjacency_tensor", "init_logits_from_adj", "LearnableAdjacency"]
