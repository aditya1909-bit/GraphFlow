from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

Array = np.ndarray


@dataclass
class EdgeIndex:
    src: Array
    dst: Array
    weight: Array
    num_nodes: int

    def __post_init__(self):
        self.src = np.asarray(self.src, dtype=np.int64)
        self.dst = np.asarray(self.dst, dtype=np.int64)
        self.weight = np.asarray(self.weight, dtype=np.float64)
        if self.src.shape != self.dst.shape:
            raise ValueError("src and dst must have the same shape")
        if self.weight.shape != self.src.shape:
            raise ValueError("weight must have the same shape as src")

    @property
    def num_edges(self) -> int:
        return int(self.src.size)


def edge_index_from_adjacency(A: Array, add_self_loops: bool = True) -> EdgeIndex:
    """Build an edge index from a dense adjacency matrix."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (N, N)")
    n = int(A.shape[0])
    A = A.astype(np.float64, copy=False)
    dst, src = np.nonzero(A)
    weight = A[dst, src].astype(np.float64, copy=False)
    if add_self_loops:
        loops = np.arange(n, dtype=np.int64)
        src = np.concatenate([src, loops])
        dst = np.concatenate([dst, loops])
        weight = np.concatenate([weight, np.ones(n, dtype=np.float64)])
    return EdgeIndex(src=src, dst=dst, weight=weight, num_nodes=n)


def edge_index_from_pairs(
    pairs: Array,
    num_nodes: int,
    weight: Optional[Array] = None,
    add_self_loops: bool = False,
) -> EdgeIndex:
    """Build an edge index from an (E, 2) array of (src, dst) pairs."""
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (E, 2)")
    src = pairs[:, 0]
    dst = pairs[:, 1]
    if weight is None:
        weight = np.ones(src.shape[0], dtype=np.float64)
    else:
        weight = np.asarray(weight, dtype=np.float64)
    if weight.shape != src.shape:
        raise ValueError("weight must have shape (E,)")
    if add_self_loops:
        loops = np.arange(int(num_nodes), dtype=np.int64)
        src = np.concatenate([src, loops])
        dst = np.concatenate([dst, loops])
        weight = np.concatenate([weight, np.ones(int(num_nodes), dtype=np.float64)])
    return EdgeIndex(src=src, dst=dst, weight=weight, num_nodes=int(num_nodes))


def gcn_normalize_edge_index(edge_index: EdgeIndex, eps: float = 1e-12) -> EdgeIndex:
    """Return edge weights normalized as D^{-1/2} A D^{-1/2}."""
    deg = np.zeros(edge_index.num_nodes, dtype=np.float64)
    np.add.at(deg, edge_index.dst, edge_index.weight)
    deg_inv_sqrt = np.power(deg + eps, -0.5)
    weight = edge_index.weight * deg_inv_sqrt[edge_index.dst] * deg_inv_sqrt[edge_index.src]
    return EdgeIndex(edge_index.src, edge_index.dst, weight, edge_index.num_nodes)


def row_normalize_edge_index(edge_index: EdgeIndex, eps: float = 1e-12) -> EdgeIndex:
    """Return edge weights normalized as D^{-1} A for mean aggregation."""
    deg = np.zeros(edge_index.num_nodes, dtype=np.float64)
    np.add.at(deg, edge_index.dst, edge_index.weight)
    inv_deg = 1.0 / (deg + eps)
    weight = edge_index.weight * inv_deg[edge_index.dst]
    return EdgeIndex(edge_index.src, edge_index.dst, weight, edge_index.num_nodes)


__all__ = [
    "EdgeIndex",
    "edge_index_from_adjacency",
    "edge_index_from_pairs",
    "gcn_normalize_edge_index",
    "row_normalize_edge_index",
]
