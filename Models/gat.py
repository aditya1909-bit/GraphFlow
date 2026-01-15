

"""
Graph Attention Network (GAT) layers and tiny 2-layer models implemented with
GraphFlow's minimal autograd Tensor.

This module includes dense-adjacency GAT (GAT) and edge-index GAT (EdgeGAT)
variants using the additive attention factorization:
e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j).

Notes
-----
- Dense GAT computes attention logits on A+I by masking non-edges.
- EdgeGAT uses edge-index segment softmax and edge-wise aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from Core.edge_index import EdgeIndex, edge_index_from_adjacency
from Core.kernels import edge_softmax, edge_sum, gather_rows
from Core.tensor import Tensor
from Core.ops import softmax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def leaky_relu(x: Tensor, alpha: float = 0.2) -> Tensor:
    # leaky_relu(x) = relu(x) - alpha * relu(-x)
    return x.relu() - (alpha * (-x).relu())


def concat_tensors(tensors: List[Tensor], axis: int = 1) -> Tensor:
    """Concatenate tensors along an axis with gradient split on backward."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, requires_grad=any(t.requires_grad for t in tensors))
    out._prev = tuple(tensors)
    sizes = [t.data.shape[axis] for t in tensors]

    def _backward():
        if out.grad is None:
            return
        idx = 0
        for t, size in zip(tensors, sizes):
            if t.requires_grad:
                slc = [slice(None)] * out.grad.ndim
                slc[axis] = slice(idx, idx + size)
                t._add_grad(out.grad[tuple(slc)])
            idx += size

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Single-head GAT layer
# ---------------------------------------------------------------------------

@dataclass
class GATLayer:
    in_dim: int
    out_dim: int
    A: np.ndarray  # adjacency for masking (will include self-loops internally)
    alpha: float = 0.2
    seed: Optional[int] = None
    use_bias: bool = True

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        limit = np.sqrt(6.0 / (self.in_dim + self.out_dim))
        self.W = Tensor(rng.uniform(-limit, limit, size=(self.in_dim, self.out_dim)), requires_grad=True, name="W")
        # attention vectors (factorized): a_src, a_dst in R^{out_dim}
        self.a_src = Tensor(rng.uniform(-limit, limit, size=(self.out_dim, 1)), requires_grad=True, name="a_src")
        self.a_dst = Tensor(rng.uniform(-limit, limit, size=(self.out_dim, 1)), requires_grad=True, name="a_dst")
        if self.use_bias:
            self.b = Tensor(np.zeros((self.out_dim,)), requires_grad=True, name="b")
        else:
            self.b = None

        # Precompute edge mask on A + I
        A_tilde = (self.A + np.eye(self.A.shape[0]))
        self._edge_mask = (A_tilde > 0)
        mask = np.zeros_like(A_tilde, dtype=np.float64)
        mask[~self._edge_mask] = -1e9
        self._attn_mask = mask

    @property
    def params(self):
        ps = [self.W, self.a_src, self.a_dst]
        if self.b is not None:
            ps.append(self.b)
        return ps

    def __call__(self, X: Tensor, activation: bool = True) -> Tensor:
        # Linear projection
        H = X.matmul(self.W)  # (N, F')

        # Attention coefficients
        f_i = H.matmul(self.a_src)          # (N, 1)
        f_j = H.matmul(self.a_dst)          # (N, 1)
        e = f_i + f_j.T()                   # (N, N)
        e = leaky_relu(e, alpha=self.alpha) # (N, N)

        # Mask non-edges by adding a large negative constant
        logits = e + Tensor(self._attn_mask)

        # Normalized attention across neighbors j
        alpha = softmax(logits, axis=1)      # (N, N)

        # Neighborhood aggregation: sum_j alpha_ij * H_j
        out = alpha.matmul(H)                # (N, F')

        if self.b is not None:
            out = out + self.b
        return out.relu() if activation else out


# ---------------------------------------------------------------------------
# Multi-head wrapper
# ---------------------------------------------------------------------------

class MultiHeadGAT:
    def __init__(self, in_dim: int, out_dim: int, A: np.ndarray, num_heads: int = 4,
                 concat: bool = True, alpha: float = 0.2, seed: Optional[int] = None):
        self.heads: List[GATLayer] = []
        rng = np.random.default_rng(seed)
        for h in range(num_heads):
            self.heads.append(
                GATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    A=A,
                    alpha=alpha,
                    seed=int(rng.integers(0, 1_000_000)),
                )
            )
        self.concat = concat

    @property
    def params(self):
        ps = []
        for head in self.heads:
            ps.extend(head.params)
        return ps

    def __call__(self, X: Tensor, activation: bool = True) -> Tensor:
        outs = [head(X, activation=activation) for head in self.heads]
        if self.concat:
            # Concatenate feature dimension: (N, H*out_dim)
            return concat_tensors(outs, axis=1)
        else:
            # Average the heads: (N, out_dim)
            out = outs[0]
            for t in outs[1:]:
                out = out + t
            return out * (1.0 / len(outs))


# ---------------------------------------------------------------------------
# Tiny 2-layer GAT model for node classification
# ---------------------------------------------------------------------------

class GAT:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, A: np.ndarray,
                 num_heads: int = 4, alpha: float = 0.2, seed: int = 0):
        # First layer: multi-head with concatenation
        self.gat1 = MultiHeadGAT(in_dim, hidden_dim, A, num_heads=num_heads, concat=True, alpha=alpha, seed=seed)
        # Second layer: single-head (averaged) to output classes
        # Using average (concat=False) to keep output dimension = hidden_dim
        self.gat2 = MultiHeadGAT(hidden_dim * num_heads, out_dim, A, num_heads=1, concat=False, alpha=alpha, seed=seed+1)

    @property
    def params(self):
        return self.gat1.params + self.gat2.params

    def __call__(self, X: np.ndarray) -> Tensor:
        X_t = Tensor(X)
        h = self.gat1(X_t, activation=True)
        out = self.gat2(h, activation=False)
        return out


# ---------------------------------------------------------------------------
# Edge-index (scalable) GAT layers
# ---------------------------------------------------------------------------

@dataclass
class EdgeGATLayer:
    in_dim: int
    out_dim: int
    edge_index: EdgeIndex
    alpha: float = 0.2
    seed: Optional[int] = None
    use_bias: bool = True

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        limit = np.sqrt(6.0 / (self.in_dim + self.out_dim))
        self.W = Tensor(rng.uniform(-limit, limit, size=(self.in_dim, self.out_dim)), requires_grad=True, name="W")
        self.a_src = Tensor(rng.uniform(-limit, limit, size=(self.out_dim, 1)), requires_grad=True, name="a_src")
        self.a_dst = Tensor(rng.uniform(-limit, limit, size=(self.out_dim, 1)), requires_grad=True, name="a_dst")
        if self.use_bias:
            self.b = Tensor(np.zeros((self.out_dim,)), requires_grad=True, name="b")
        else:
            self.b = None
        if np.all(self.edge_index.weight == 1.0):
            self._edge_weight = None
        else:
            self._edge_weight = Tensor(self.edge_index.weight)

    @property
    def params(self):
        ps = [self.W, self.a_src, self.a_dst]
        if self.b is not None:
            ps.append(self.b)
        return ps

    def __call__(self, X: Tensor, activation: bool = True) -> Tensor:
        H = X.matmul(self.W)  # (N, F')
        h_src = gather_rows(H, self.edge_index.src)  # (E, F')
        h_dst = gather_rows(H, self.edge_index.dst)  # (E, F')
        f_src = h_src.matmul(self.a_src)            # (E, 1)
        f_dst = h_dst.matmul(self.a_dst)            # (E, 1)
        e = leaky_relu(f_src + f_dst, alpha=self.alpha)
        e_flat = e.reshape((e.data.shape[0],))
        attn = edge_softmax(e_flat, dst=self.edge_index.dst, num_nodes=self.edge_index.num_nodes)
        if self._edge_weight is not None:
            attn = attn * self._edge_weight
        out = edge_sum(
            self.edge_index.src,
            self.edge_index.dst,
            H,
            weight=attn,
            num_nodes=self.edge_index.num_nodes,
        )
        if self.b is not None:
            out = out + self.b
        return out.relu() if activation else out


class EdgeMultiHeadGAT:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_index: EdgeIndex,
        num_heads: int = 4,
        concat: bool = True,
        alpha: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.heads: List[EdgeGATLayer] = []
        rng = np.random.default_rng(seed)
        for _ in range(num_heads):
            self.heads.append(
                EdgeGATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    edge_index=edge_index,
                    alpha=alpha,
                    seed=int(rng.integers(0, 1_000_000)),
                )
            )
        self.concat = concat

    @property
    def params(self):
        ps = []
        for head in self.heads:
            ps.extend(head.params)
        return ps

    def __call__(self, X: Tensor, activation: bool = True) -> Tensor:
        outs = [head(X, activation=activation) for head in self.heads]
        if self.concat:
            return concat_tensors(outs, axis=1)
        out = outs[0]
        for t in outs[1:]:
            out = out + t
        return out * (1.0 / len(outs))


class EdgeGAT:
    """Edge-index GAT for scalable attention message passing."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        A: Optional[np.ndarray] = None,
        edge_index: Optional[EdgeIndex] = None,
        num_heads: int = 4,
        alpha: float = 0.2,
        seed: int = 0,
        add_self_loops: bool = True,
    ):
        if edge_index is None:
            if A is None:
                raise ValueError("A or edge_index must be provided")
            edge_index = edge_index_from_adjacency(A, add_self_loops=add_self_loops)
        self.gat1 = EdgeMultiHeadGAT(
            in_dim,
            hidden_dim,
            edge_index,
            num_heads=num_heads,
            concat=True,
            alpha=alpha,
            seed=seed,
        )
        self.gat2 = EdgeMultiHeadGAT(
            hidden_dim * num_heads,
            out_dim,
            edge_index,
            num_heads=1,
            concat=False,
            alpha=alpha,
            seed=seed + 1,
        )

    @property
    def params(self):
        return self.gat1.params + self.gat2.params

    def __call__(self, X: np.ndarray) -> Tensor:
        X_t = Tensor(X)
        h = self.gat1(X_t, activation=True)
        out = self.gat2(h, activation=False)
        return out
