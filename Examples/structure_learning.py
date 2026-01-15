"""
Structure learning demo: learn adjacency + GCN weights end-to-end.
"""
from __future__ import annotations

import numpy as np

from Core.graph import make_two_community_graph
from Core.ops import cross_entropy, accuracy
from Core.tensor import Tensor
from Core.optim import Adam
from Models.structure_gcn import StructureGCN


def main():
    # Data
    A_true, X, y, masks = make_two_community_graph(n_per=60, p_in=0.15, p_out=0.02, seed=7)
    num_feats = X.shape[1]
    num_classes = int(np.max(y)) + 1

    # Model (learns adjacency)
    model = StructureGCN(
        in_dim=num_feats,
        hidden_dim=16,
        out_dim=num_classes,
        n_nodes=A_true.shape[0],
        seed=0,
        init_adj=A_true,
    )
    optimizer = Adam(lr=0.01)

    epochs = 200
    reg_lambda = 5e-4  # sparsity penalty on adjacency
    prior_lambda = 5e-3  # keep adjacency near the observed structure
    warmup_epochs = 40
    for ep in range(1, epochs + 1):
        logits = model(X)
        loss = cross_entropy(logits, y, mask=masks["train"])

        # Encourage sparse adjacency and stay close to observed structure
        A = model.adj.adjacency()
        loss = loss + (A.sum() * reg_lambda)
        loss = loss + ((A - Tensor(A_true)) * (A - Tensor(A_true))).sum() * prior_lambda

        loss.backward()
        if ep <= warmup_epochs:
            optimizer.step(model.lin1.params + model.lin2.params)
        else:
            optimizer.step(model.params)

        if ep % 20 == 0:
            acc_tr = accuracy(Tensor(logits.data[masks["train"]]), y[masks["train"]])
            acc_va = accuracy(Tensor(logits.data[masks["val"]]), y[masks["val"]])
            print(f"Epoch {ep:03d} | loss={loss.data.item():.4f} | train_acc={acc_tr:.3f} | val_acc={acc_va:.3f}")

    logits = model(X)
    test_acc = accuracy(Tensor(logits.data[masks["test"]]), y[masks["test"]])
    print(f"Test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
