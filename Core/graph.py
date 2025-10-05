

"""
Graph utilities for GraphFlow.

This module provides:
- normalize_adjacency: symmetric normalization \hat{A} = D^{-1/2} (A + I) D^{-1/2}
- train_val_test_split: boolean masks for dataset splits
- make_two_community_graph: simple 2-block stochastic block model with noisy features
- to_numpy_adjacency: convert a NetworkX graph to a NumPy adjacency matrix
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import networkx as nx

Array = np.ndarray

# ---------------------------------------------------------------------------
# Core graph helpers
# ---------------------------------------------------------------------------

def normalize_adjacency(A: Array, add_self_loops: bool = True) -> Array:
    """Return symmetrically-normalized adjacency matrix.

    Computes \hat{A} = D^{-1/2} (A + I) D^{-1/2} when ``add_self_loops`` is True,
    else D^{-1/2} A D^{-1/2}. Assumes ``A`` is square and non-negative.

    Parameters
    ----------
    A : (N, N) np.ndarray
        Adjacency matrix.
    add_self_loops : bool, default True
        Whether to add the identity before normalization.

    Returns
    -------
    (N, N) np.ndarray
        Symmetrically-normalized adjacency.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (N,N) matrix")

    A_norm = A.astype(np.float64, copy=True)
    if add_self_loops:
        A_norm = A_norm + np.eye(A_norm.shape[0], dtype=A_norm.dtype)

    deg = A_norm.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_norm @ D_inv_sqrt


def train_val_test_split(
    n: int,
    train: float = 0.6,
    val: float = 0.2,
    seed: Optional[int] = None,
) -> Dict[str, Array]:
    """Create boolean masks for train/val/test splits over ``n`` nodes.

    Fractions are normalized if they do not sum to 1. The split is shuffled
    deterministically with ``seed``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if train <= 0 or val < 0 or train + val >= 1.0:
        # allow any values but normalize for safety
        total = max(train, 1e-12) + max(val, 0.0) + 1.0  # remaining goes to test
        train = train / (train + val + (1 - (train + val))) if (train + val) < 1 else 0.6
        val = val / (train + val + (1 - (train + val))) if (train + val) < 1 else 0.2

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(train * n))
    n_val = int(round(val * n))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - 1 - n_train)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    masks = {
        "train": np.isin(np.arange(n), train_idx),
        "val": np.isin(np.arange(n), val_idx),
        "test": np.isin(np.arange(n), test_idx),
    }
    return masks



def make_two_community_graph(
    n_per: int = 50,
    p_in: float = 0.15,
    p_out: float = 0.02,
    seed: Optional[int] = 42,
) -> Tuple[Array, Array, Array, Dict[str, Array]]:
    """Generate a 2-community stochastic block model dataset.

    Returns adjacency ``A``, features ``X``, labels ``y``, and split masks.

    Features are a noisy 2-d one-hot community indicator.
    """
    if n_per < 2:
        raise ValueError("n_per must be >= 2")

    sizes = [n_per, n_per]
    probs = [[p_in, p_out], [p_out, p_in]]

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    A = to_numpy_adjacency(G)

    y = np.array([0] * n_per + [1] * n_per, dtype=int)
    X = np.zeros((2 * n_per, 2), dtype=float)
    X[:n_per, 0] = 1.0
    X[n_per:, 1] = 1.0
    rng = np.random.default_rng(seed)
    X = X + 0.1 * rng.normal(size=X.shape)

    masks = train_val_test_split(A.shape[0], train=0.6, val=0.2, seed=seed)
    return A, X, y, masks



def to_numpy_adjacency(G: nx.Graph) -> Array:
    """Convert a NetworkX graph to a NumPy adjacency matrix (float64)."""
    A = nx.to_numpy_array(G, dtype=float)
    return A


__all__ = [
    "normalize_adjacency",
    "train_val_test_split",
    "make_two_community_graph",
    "to_numpy_adjacency",
]