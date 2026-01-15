"""
Reproducible benchmark: GCN node classification on a toy SBM graph.
Uses edge-index kernels for scalable message passing.
"""
from __future__ import annotations

import numpy as np

from Core.edge_index import edge_index_from_adjacency
from Core.graph import make_two_community_graph
from Core.ops import cross_entropy, accuracy
from Core.tensor import Tensor
from Core.optim import Adam
from Models.scalable_gcn import ScalableGCN


def run(seed: int = 7, epochs: int = 200) -> dict:
    A, X, y, masks = make_two_community_graph(n_per=60, p_in=0.15, p_out=0.02, seed=seed)
    edge_index = edge_index_from_adjacency(A, add_self_loops=True)
    model = ScalableGCN(
        in_dim=X.shape[1],
        hidden_dim=16,
        out_dim=int(np.max(y)) + 1,
        edge_index=edge_index,
        seed=0,
    )
    optimizer = Adam(lr=0.01)

    for _ in range(epochs):
        logits = model(X)
        loss = cross_entropy(logits, y, mask=masks["train"])
        loss.backward()
        optimizer.step(model.params)

    logits = model(X)
    return {
        "train_acc": accuracy(Tensor(logits.data[masks["train"]]), y[masks["train"]]),
        "val_acc": accuracy(Tensor(logits.data[masks["val"]]), y[masks["val"]]),
        "test_acc": accuracy(Tensor(logits.data[masks["test"]]), y[masks["test"]]),
    }


def main():
    metrics = run()
    print(f"GCN node classification | train={metrics['train_acc']:.3f} "
          f"val={metrics['val_acc']:.3f} test={metrics['test_acc']:.3f}")


if __name__ == "__main__":
    main()
