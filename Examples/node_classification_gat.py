"""
Node classification on a synthetic 2-community graph with an edge-index GAT.
Uses GraphFlow's minimal autograd and ops.
"""
from __future__ import annotations

import numpy as np

from Core.edge_index import edge_index_from_adjacency
from Core.graph import make_two_community_graph
from Core.ops import cross_entropy, accuracy
from Core.tensor import Tensor
from Core.optim import Adam
from Models.gat import EdgeGAT


def main():
    # Data
    A, X, y, masks = make_two_community_graph(n_per=60, p_in=0.15, p_out=0.02, seed=7)
    num_feats = X.shape[1]
    num_classes = int(np.max(y)) + 1

    # Model
    edge_index = edge_index_from_adjacency(A, add_self_loops=True)
    model = EdgeGAT(
        in_dim=num_feats,
        hidden_dim=8,
        out_dim=num_classes,
        edge_index=edge_index,
        num_heads=4,
        alpha=0.2,
        seed=0,
    )

    # Train
    epochs = 200
    optimizer = Adam(lr=0.01)
    for ep in range(1, epochs + 1):
        logits = model(X)  # Tensor (N, C)
        loss = cross_entropy(logits, y, mask=masks["train"])
        loss.backward()
        optimizer.step(model.params)

        if ep % 20 == 0:
            acc_tr = accuracy(Tensor(logits.data[masks["train"]]), y[masks["train"]])
            acc_va = accuracy(Tensor(logits.data[masks["val"]]), y[masks["val"]])
            print(f"Epoch {ep:03d} | loss={loss.data.item():.4f} | train_acc={acc_tr:.3f} | val_acc={acc_va:.3f}")

    # Test
    logits = model(X)
    test_acc = accuracy(Tensor(logits.data[masks["test"]]), y[masks["test"]])
    print(f"Test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
