"""Vectorized message-passing kernels for GraphFlow."""
from __future__ import annotations

from typing import Optional, Union
import numpy as np

from .edge_index import EdgeIndex
from .tensor import Tensor


class MessagePassingKernel:
    """Cache adjacency as a constant Tensor to reduce graph overhead."""

    def __init__(self, A_norm: np.ndarray):
        self.A = Tensor(A_norm)

    def gcn(self, X: Tensor) -> Tensor:
        return self.A.matmul(X)

    def mean_aggregate(self, X: Tensor) -> Tensor:
        return self.A.matmul(X)


def gcn_message_passing(A_norm: np.ndarray, X: Tensor) -> Tensor:
    """One-shot vectorized GCN message passing."""
    return Tensor(A_norm).matmul(X)


def gather_rows(X: Tensor, idx: np.ndarray) -> Tensor:
    """Gather rows from X with gradient accumulation on backward."""
    idx_arr = np.asarray(idx, dtype=np.int64)
    out = Tensor(X.data[idx_arr], requires_grad=X.requires_grad)
    out._prev = (X,)

    def _backward():
        if out.grad is None or not X.requires_grad:
            return
        grad = np.zeros_like(X.data)
        np.add.at(grad, idx_arr, out.grad)
        X._add_grad(grad)

    out._backward = _backward
    return out


def edge_sum(
    src: np.ndarray,
    dst: np.ndarray,
    X: Tensor,
    weight: Optional[Union[np.ndarray, Tensor]] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    """Aggregate edge messages sum_j w_ij * X_j into destination nodes."""
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    if src_arr.shape != dst_arr.shape:
        raise ValueError("src and dst must have the same shape")
    if num_nodes is None:
        num_nodes = int(X.data.shape[0])

    weight_t = None
    if weight is None:
        weight_data = np.ones(src_arr.shape[0], dtype=np.float64)
    elif isinstance(weight, Tensor):
        weight_t = weight
        weight_data = np.asarray(weight.data, dtype=np.float64).reshape(-1)
    else:
        weight_data = np.asarray(weight, dtype=np.float64).reshape(-1)
    if weight_data.shape[0] != src_arr.shape[0]:
        raise ValueError("weight must have shape (E,)")

    msg = X.data[src_arr] * weight_data[:, None]
    out_data = np.zeros((num_nodes, X.data.shape[1]), dtype=np.float64)
    np.add.at(out_data, dst_arr, msg)
    out = Tensor(out_data, requires_grad=X.requires_grad or (weight_t is not None and weight_t.requires_grad))
    out._prev = tuple(t for t in (X, weight_t) if isinstance(t, Tensor))

    def _backward():
        if out.grad is None:
            return
        if X.requires_grad:
            grad_x = np.zeros_like(X.data)
            np.add.at(grad_x, src_arr, out.grad[dst_arr] * weight_data[:, None])
            X._add_grad(grad_x)
        if weight_t is not None and weight_t.requires_grad:
            grad_w = np.sum(out.grad[dst_arr] * X.data[src_arr], axis=1)
            weight_t._add_grad(grad_w.reshape(weight_t.data.shape))

    out._backward = _backward
    return out


def edge_softmax(
    scores: Tensor,
    dst: np.ndarray,
    num_nodes: Optional[int] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Segment softmax over edges grouped by destination node."""
    dst_arr = np.asarray(dst, dtype=np.int64)
    if dst_arr.size == 0:
        out = Tensor(scores.data, requires_grad=scores.requires_grad)
        out._prev = (scores,)

        def _backward():
            if out.grad is None or not scores.requires_grad:
                return
            scores._add_grad(out.grad)

        out._backward = _backward
        return out
    if num_nodes is None:
        num_nodes = int(dst_arr.max()) + 1

    s = scores.data.reshape(-1)
    max_per = np.full(num_nodes, -np.inf, dtype=np.float64)
    np.maximum.at(max_per, dst_arr, s)
    exp = np.exp(s - max_per[dst_arr])
    denom = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(denom, dst_arr, exp)
    out_data = exp / (denom[dst_arr] + eps)
    out_data = out_data.reshape(scores.data.shape)

    out = Tensor(out_data, requires_grad=scores.requires_grad)
    out._prev = (scores,)

    def _backward():
        if out.grad is None or not scores.requires_grad:
            return
        grad_out = out.grad.reshape(-1)
        alpha = out_data.reshape(-1)
        tmp = grad_out * alpha
        sum_per = np.zeros(num_nodes, dtype=np.float64)
        np.add.at(sum_per, dst_arr, tmp)
        grad_scores = alpha * (grad_out - sum_per[dst_arr])
        scores._add_grad(grad_scores.reshape(scores.data.shape))

    out._backward = _backward
    return out


class EdgeIndexKernel:
    """Cached edge-index message passing kernel."""

    def __init__(self, edge_index: EdgeIndex):
        self.src = edge_index.src
        self.dst = edge_index.dst
        self.weight = edge_index.weight
        self.num_nodes = edge_index.num_nodes

    def gcn(self, X: Tensor) -> Tensor:
        return edge_sum(self.src, self.dst, X, weight=self.weight, num_nodes=self.num_nodes)

    def mean_aggregate(self, X: Tensor) -> Tensor:
        return edge_sum(self.src, self.dst, X, weight=self.weight, num_nodes=self.num_nodes)


__all__ = [
    "MessagePassingKernel",
    "gcn_message_passing",
    "gather_rows",
    "edge_sum",
    "edge_softmax",
    "EdgeIndexKernel",
]
