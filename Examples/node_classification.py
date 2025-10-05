

"""
Node classification on a synthetic 2‑community graph with a tiny GCN.
Uses GraphFlow's minimal autograd and ops.
"""
from __future__ import annotations

import numpy as np

from Core.graph import make_two_community_graph, normalize_adjacency
from Core.tensor import Tensor
from Core.ops import cross_entropy, accuracy


# ---------------------------------------------------------------------------
# Minimal GCN
# ---------------------------------------------------------------------------
class Linear:
    def __init__(self, in_dim: int, out_dim: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.W = Tensor(rng.uniform(-limit, limit, size=(in_dim, out_dim)), requires_grad=True, name="W")
        self.b = Tensor(np.zeros((out_dim,)), requires_grad=True, name="b")

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


class GCN:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, A: np.ndarray, seed: int = 0):
        A_norm = normalize_adjacency(A, add_self_loops=True)
        self.gcn1 = GCNLayer(in_dim, hidden_dim, A_norm, seed=seed)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, A_norm, seed=seed + 1)

    def __call__(self, X: np.ndarray) -> Tensor:
        X_t = Tensor(X)
        h = self.gcn1(X_t, activation=True)
        out = self.gcn2(h, activation=False)
        return out

    @property
    def params(self):
        return self.gcn1.params + self.gcn2.params


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def sgd(params, lr=0.05):
    for p in params:
        if p.grad is not None:
            p.data -= lr * p.grad
            p.grad = None


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------

def main():
    # Data
    A, X, y, masks = make_two_community_graph(n_per=60, p_in=0.15, p_out=0.02, seed=7)
    num_feats = X.shape[1]
    num_classes = int(np.max(y)) + 1

    # Model
    model = GCN(in_dim=num_feats, hidden_dim=16, out_dim=num_classes, A=A, seed=0)

    # Train
    epochs = 200
    lr = 0.05
    for ep in range(1, epochs + 1):
        logits = model(X)  # Tensor (N, C)
        # Compute loss on train nodes only
        train_idx = masks["train"]
        loss = cross_entropy(Tensor(logits.data[train_idx], requires_grad=True), y[train_idx])
        loss.backward()
        sgd(model.params, lr=lr)

        if ep % 20 == 0:
            acc_tr = accuracy(logits, y[masks["train"]])
            acc_va = accuracy(Tensor(logits.data[masks["val"]]), y[masks["val"]])
            print(f"Epoch {ep:03d} | loss={loss.data.item():.4f} | train_acc={acc_tr:.3f} | val_acc={acc_va:.3f}")

    # Test
    logits = model(X)
    test_acc = accuracy(Tensor(logits.data[masks["test"]]), y[masks["test"]])
    print(f"Test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()