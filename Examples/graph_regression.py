

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
from Core.optim import Adam

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_er_graph(n_nodes: int, p: float, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (A, X, y) where y is avg clustering coefficient with noise."""
    G = nx.fast_gnp_random_graph(n_nodes, p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)

    rng = np.random.default_rng(seed)
    # Node features: degree, local clustering coefficient, and bias
    deg = np.sum(A, axis=1, keepdims=True)
    deg_feat = deg / (n_nodes - 1 + 1e-12)
    clust_dict = nx.clustering(G)
    clust = np.array([clust_dict[i] for i in range(n_nodes)], dtype=np.float64).reshape(-1, 1)
    clust = clust + 0.02 * rng.normal(size=clust.shape)
    X = np.concatenate([deg_feat, clust, np.ones_like(deg)], axis=1)

    y_true = float(nx.average_clustering(G))
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
    def __init__(self, in_dim: int, out_dim: int, seed: int | None = None):
        self.lin = Linear(in_dim, out_dim, seed=seed)

    def __call__(self, X: Tensor, A_norm: np.ndarray, activation: bool = True) -> Tensor:
        AX = Tensor(A_norm).matmul(X)
        Z = self.lin(AX)
        return Z.relu() if activation else Z

    @property
    def params(self):
        return self.lin.params


class GraphRegressor:
    def __init__(self, in_dim: int, hidden: int, seed: int = 0):
        self.gcn1 = GCNLayer(in_dim, hidden, seed=seed)
        self.gcn2 = GCNLayer(hidden, hidden, seed=seed + 1)
        self.gcn3 = GCNLayer(hidden, hidden, seed=seed + 2)
        self.readout1 = Linear(hidden, hidden, seed=seed + 3)
        self.readout2 = Linear(hidden, 1, seed=seed + 4)  # mean pooled -> scalar

    def __call__(self, X: np.ndarray, A: np.ndarray, train: bool = True, drop_p: float = 0.2) -> Tensor:
        A_norm = normalize_adjacency(A, add_self_loops=True)
        X_t = Tensor(X)
        h1 = self.gcn1(X_t, A_norm, activation=True)
        h1 = dropout(h1, p=drop_p, train=train)
        h2 = self.gcn2(h1, A_norm, activation=True)
        h2 = h2 + h1  # residual
        h2 = dropout(h2, p=drop_p, train=train)
        h3 = self.gcn3(h2, A_norm, activation=True)
        h3 = h3 + h2  # residual
        g = h3.mean(axis=0, keepdims=True)  # (1, hidden)
        r1 = self.readout1(g).relu()
        r1 = dropout(r1, p=drop_p, train=train)
        yhat = self.readout2(r1)             # (1, 1)
        return yhat

    @property
    def params(self):
        return (
            self.gcn1.params
            + self.gcn2.params
            + self.gcn3.params
            + self.readout1.params
            + self.readout2.params
        )


# ---------------------------------------------------------------------------
# Training utils
# ---------------------------------------------------------------------------

def mse(pred: Tensor, target: float) -> Tensor:
    t = Tensor(np.array([[target]], dtype=np.float64))
    return ((pred - t) * (pred - t)).mean()

def dropout(x: Tensor, p: float, train: bool = True, seed: int | None = None) -> Tensor:
    """Inverted dropout implemented with a constant mask Tensor."""
    if (not train) or p <= 0.0:
        return x
    rng = np.random.default_rng(seed)
    keep = 1.0 - p
    mask = (rng.random(x.data.shape) < keep).astype(np.float64) / keep
    return x * Tensor(mask)


def get_params_state(params: List[Tensor]) -> List[np.ndarray]:
    return [p.data.copy() for p in params]


def set_params_state(params: List[Tensor], state: List[np.ndarray]):
    for p, v in zip(params, state):
        p.data[:] = v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_epoch(
    model: GraphRegressor,
    batch: List[Tuple[np.ndarray, np.ndarray, float]],
    optimizer: Adam,
    y_mean: float,
    y_std: float,
    rng: np.random.Generator,
    drop_p: float,
) -> float:
    losses = []
    for i in rng.permutation(len(batch)):
        A, X, y = batch[i]
        y_norm = (y - y_mean) / y_std
        yhat = model(X, A, train=True, drop_p=drop_p)
        loss = mse(yhat, y_norm)
        loss.backward()
        optimizer.step(model.params)
        losses.append(loss.data.item())
    return float(np.mean(losses))


def evaluate(
    model: GraphRegressor,
    data: List[Tuple[np.ndarray, np.ndarray, float]],
    y_mean: float,
    y_std: float,
):
    errs = []
    for A, X, y in data:
        pred_norm = model(X, A, train=False, drop_p=0.0).data.item()
        pred = pred_norm * y_std + y_mean
        errs.append((pred - y) ** 2)
    rmse = float(np.sqrt(np.mean(errs)))
    return rmse


def main():
    train, val, test = make_dataset(N=120, n_nodes=24, p_range=(0.08, 0.30), seed=3)

    _, X0, _ = train[0]
    model = GraphRegressor(in_dim=X0.shape[1], hidden=64, seed=7)

    y_train = np.array([y for _, _, y in train], dtype=np.float64)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-12)

    epochs = 200
    optimizer = Adam(lr=0.005)
    rng = np.random.default_rng(123)
    drop_p = 0.1
    best_rmse = float("inf")
    best_state = get_params_state(model.params)
    patience = 20
    bad_epochs = 0
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, train, optimizer, y_mean, y_std, rng, drop_p=drop_p)
        val_rmse = evaluate(model, val, y_mean, y_std)
        if ep % 5 == 0:
            print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_RMSE={val_rmse:.4f}")
        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_state = get_params_state(model.params)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {ep:02d} | best_val_RMSE={best_rmse:.4f}")
                break

    set_params_state(model.params, best_state)
    test_rmse = evaluate(model, test, y_mean, y_std)
    print(f"Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
