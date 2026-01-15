"""
Reproducible benchmark: topology generation on a toy SBM graph.
Uses pairwise scoring with a guide edge-index kernel.
"""
from __future__ import annotations

import numpy as np

from Core.edge_index import edge_index_from_adjacency
from Core.graph import make_two_community_graph
from Core.ops import binary_cross_entropy_with_logits
from Core.optim import Adam
from Models.topology_generator import TopologyGenerator


def edge_accuracy(logits, labels):
    probs = logits.sigmoid().data
    preds = (probs > 0.5).astype(int)
    return float((preds == labels).mean())


def edge_recall(logits, labels):
    probs = logits.sigmoid().data
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    tp = float(((preds == 1) & (labels == 1)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    return tp / (tp + fn + 1e-12)


def edge_specificity(logits, labels):
    probs = logits.sigmoid().data
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    return tn / (tn + fp + 1e-12)


def edge_split(A, rng, val_frac=0.1):
    triu = np.triu(np.ones_like(A, dtype=bool), k=1)
    pos_idx = np.column_stack(np.where((A > 0) & triu))
    neg_idx = np.column_stack(np.where((A == 0) & triu))
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_pos_val = max(1, int(val_frac * pos_idx.shape[0]))
    n_neg_val = max(1, int(val_frac * neg_idx.shape[0]))
    val_pos = pos_idx[:n_pos_val]
    val_neg = neg_idx[:n_neg_val]
    train_pos = pos_idx[n_pos_val:]
    train_neg = neg_idx[n_neg_val:]
    return train_pos, train_neg, val_pos, val_neg


def build_pairs(pos_idx, neg_idx):
    pairs = np.vstack([pos_idx, neg_idx])
    labels = np.concatenate([np.ones(pos_idx.shape[0]), np.zeros(neg_idx.shape[0])]).astype(np.float64)
    return pairs, labels


def sample_edge_batch(pos_idx, neg_idx, rng, n_pos=512):
    n_pos = min(n_pos, pos_idx.shape[0])
    n_neg = min(n_pos, neg_idx.shape[0])
    pos_sel = pos_idx[rng.choice(pos_idx.shape[0], size=n_pos, replace=False)]
    neg_sel = neg_idx[rng.choice(neg_idx.shape[0], size=n_neg, replace=False)]
    pairs, labels = build_pairs(pos_sel, neg_sel)
    return pairs, labels


def augment_features(A, X):
    n = A.shape[0]
    deg = A.sum(axis=1, keepdims=True)
    deg_feat = deg / (n - 1 + 1e-12)
    A2 = A @ A
    tri = np.diag(A2).reshape(-1, 1)
    denom = np.maximum(deg * (deg - 1), 1.0)
    clust = tri / denom
    return np.concatenate([X, deg_feat, clust], axis=1)


def run(seed: int = 7, epochs: int = 300) -> dict:
    A, X, _, _ = make_two_community_graph(n_per=60, p_in=0.15, p_out=0.02, seed=seed)
    np.fill_diagonal(A, 0.0)
    X = augment_features(A, X)

    guide_edge_index = edge_index_from_adjacency(A, add_self_loops=True)
    model = TopologyGenerator(
        in_dim=X.shape[1],
        hidden_dim=128,
        n_nodes=A.shape[0],
        seed=0,
        guide_edge_index=guide_edge_index,
    )
    optimizer = Adam(lr=0.003)
    rng = np.random.default_rng(123)
    train_pos, train_neg, val_pos, val_neg = edge_split(A, rng, val_frac=0.1)
    val_pairs, val_labels = build_pairs(val_pos, val_neg)

    best_val = float("inf")
    best_state = [p.data.copy() for p in model.params]

    for _ in range(epochs):
        pairs, labels = sample_edge_batch(train_pos, train_neg, rng, n_pos=512)
        logits = model(X, pairs=pairs)
        loss = binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step(model.params)

        val_logits = model(X, pairs=val_pairs)
        val_loss = binary_cross_entropy_with_logits(val_logits, val_labels).data.item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = [p.data.copy() for p in model.params]

    for p, v in zip(model.params, best_state):
        p.data[:] = v
    logits = model(X, pairs=val_pairs)
    return {
        "val_acc": edge_accuracy(logits, val_labels),
        "val_recall": edge_recall(logits, val_labels),
        "val_spec": edge_specificity(logits, val_labels),
    }


def main():
    metrics = run()
    print(
        "Topology generation | "
        f"val_acc={metrics['val_acc']:.3f} "
        f"val_recall={metrics['val_recall']:.3f} "
        f"val_spec={metrics['val_spec']:.3f}"
    )


if __name__ == "__main__":
    main()
