

"""
Graph-level regression demo using a tiny GCN + mean pooling.

Target: estimate the (noisy) average clustering coefficient of a graph.
Dataset: random Erdos–Rényi graphs with node-degree features.

This example is self-contained (does not rely on your Models/ module) and
uses the lightweight Tensor autograd from GraphFlow.Core.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import networkx as nx

from Core.tensor import Tensor
from Core.graph import normalize_adjacency

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_er_graph(n_nodes: int, p: float, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (A, X, y) where y is avg clustering coefficient with noise."""
    G = nx.fast_gnp_random_graph(n_nodes, p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)

    # Node features: degree (1-d) and constant bias (1), total 2 dims
    deg = np.sum(A, axis=1, keepdims=True)
    X = np.concatenate([deg / (n_nodes - 1 + 1e-12), np.ones_like(deg)], axis=1)

    y_true = float(nx.average_clustering(G))
    rng = np.random.default_rng(seed)
    y = y_true + rng.normal(0.02)  # small observational noise
    return A, X, y


def make_dataset(N: int = 200, n_nodes: int = 24, p_range=(0.05, 0.35), seed: int = 0):
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(N):
        p = rng.uniform(*p_range)
        A, X, y = make_er_graph(n_nodes=n_nodes, p=p, seed=seed + i + 7)
        graphs.append((A, X, y))
    # train/val/test split (60/20/20)
    n_train = int(0.6 * N)
    n_val = int(0.2 * N)
    train = graphs[:n_train]
    val = graphs[n_train:n_train + n_val]
    test = graphs[n_train + n_val:]
    return train, val, test

# ---------------------------------------------------------------------------
# Tiny GCN for graph-level regression
# ---------------------------------------------------------------------------

@dataclass
class Linear:
    in_dim: int
    out_dim: int
    seed: int | None = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        limit = np.sqrt(6 / (self.in_dim + self.out_dim))
        self.W = Tensor(rng.uniform(-limit, limit, size=(self.in_dim, self.out_dim)), requires_grad=True, name="W")
        self.b = Tensor(np.zeros((self.out_dim,)), requires_grad=True, name="b")

    def __call__(self, X: Tensor) -> Tensor:
        return X.matmul(self.W) + self.b

    @property
    def params(self):
        return [self.W, self.b]


class GCNLayer:
    def __init__(self, in_dim: int, out_dim: int, A_norm: np.ndarray, seed: int | None = None):
        self.lin = Linear(in_dim, out_dim, seed=seed)
        self.A_norm = A_norm

    def __call__(self, X: Tensor, activation: bool = True) -> Tensor:
        AX = Tensor(self.A_norm).matmul(X)
        Z = self.lin(AX)
        return Z.relu() if activation else Z

    @property
    def params(self):
        return self.lin.params


class GraphRegressor:
    def __init__(self, in_dim: int, hidden: int, A: np.ndarray, seed: int = 0):
        A_norm = normalize_adjacency(A, add_self_loops=True)
        self.gcn1 = GCNLayer(in_dim, hidden, A_norm, seed=seed)
        self.gcn2 = GCNLayer(hidden, hidden, A_norm, seed=seed + 1)
        self.readout = Linear(hidden, 1, seed=seed + 2)  # mean pooled -> scalar

    def __call__(self, X: np.ndarray) -> Tensor:
        X_t = Tensor(X)
        h1 = self.gcn1(X_t, activation=True)
        h2 = self.gcn2(h1, activation=True)
        g = h2.mean(axis=0, keepdims=True)  # (1, hidden)
        yhat = self.readout(g)               # (1, 1)
        return yhat

    @property
    def params(self):
        return self.gcn1.params + self.gcn2.params + self.readout.params


# ---------------------------------------------------------------------------
# Training utils
# ---------------------------------------------------------------------------

def mse(pred: Tensor, target: float) -> Tensor:
    t = Tensor(np.array([[target]], dtype=np.float64))
    return ((pred - t) * (pred - t)).mean()


def sgd(params, lr=0.05):
    for p in params:
        if p.grad is not None:
            p.data -= lr * p.grad
            p.grad = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_epoch(model: GraphRegressor, batch: List[Tuple[np.ndarray, np.ndarray, float]], lr: float) -> float:
    losses = []
    for A, X, y in batch:
        # For each graph, build a model tied to that graph's topology but shared weights
        # We reuse the model's A (constructed at init); to adapt per-graph A we would re-wrap layers.
        # Here, we assume a fixed topology across dataset for simplicity of the demo.
        yhat = model(X)
        loss = mse(yhat, y)
        loss.backward()
        sgd(model.params, lr=lr)
        losses.append(loss.data.item())
    return float(np.mean(losses))


def evaluate(model: GraphRegressor, data: List[Tuple[np.ndarray, np.ndarray, float]]):
    errs = []
    for _, X, y in data:
        pred = model(X).data.item()
        errs.append((pred - y) ** 2)
    rmse = float(np.sqrt(np.mean(errs)))
    return rmse


def main():
    # Build dataset of ER graphs *with a fixed topology* for simplicity
    # (If you want per-graph topology, you would re-init layers per graph or support batching.)
    train, val, test = make_dataset(N=120, n_nodes=24, p_range=(0.08, 0.30), seed=3)

    # Use the first train graph's adjacency as the shared topology for GCN demo
    A0, X0, _ = train[0]
    model = GraphRegressor(in_dim=X0.shape[1], hidden=32, A=A0, seed=7)

    epochs = 40
    lr = 0.05
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, train, lr)
        val_rmse = evaluate(model, val)
        if ep % 5 == 0:
            print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_RMSE={val_rmse:.4f}")

    test_rmse = evaluate(model, test)
    print(f"Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()