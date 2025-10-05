

"""
Graph Attention Network (GAT) layers and a tiny 2-layer GAT model
implemented with GraphFlow's minimal autograd Tensor.

This implementation supports single-head and multi-head attention using the
"additive" factorization e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j).

Notes
-----
- We compute attention logits only on edges of A+I (self-loops included) by
  masking non-edges with a large negative number before softmax.
- Broadcasting and masked operations are kept NumPy-simple to stay compatible
  with the lightweight Tensor engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from Core.tensor import Tensor
from Core.ops import softmax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def leaky_relu(x: Tensor, alpha: float = 0.2) -> Tensor:
    # leaky_relu(x) = relu(x) - alpha * relu(-x)
    return x.relu() - (alpha * (-x).relu())


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
        e = f_i + f_j.T                     # (N, N)
        e = leaky_relu(e, alpha=self.alpha) # (N, N)

        # Mask non-edges by setting to a large negative number (constant)
        logits_np = e.data.copy()
        logits_np[~self._edge_mask] = -1e9
        logits = Tensor(logits_np, requires_grad=False)  # logits are constants where masked

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
            out_np = np.concatenate([t.data for t in outs], axis=1)
            return Tensor(out_np, requires_grad=any(t.requires_grad for t in outs))
        else:
            # Average the heads: (N, out_dim)
            out_np = np.mean([t.data for t in outs], axis=0)
            return Tensor(out_np, requires_grad=any(t.requires_grad for t in outs))


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