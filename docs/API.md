# GraphFlow API

This document summarizes the core public API in GraphFlow. Everything is
NumPy‑only and built on the custom `Tensor` autograd engine.

## Core

### `Core.tensor.Tensor`
- Autograd tensor supporting basic arithmetic, matmul, reductions, and activations.
- Key methods: `backward()`, `matmul()`, `sum()`, `mean()`, `relu()`, `exp()`,
  `log()`, `sigmoid()`, `reshape()`.

### `Core.ops`
- `softmax(logits, axis=-1)`
- `log_softmax(logits, axis=-1)`
- `cross_entropy(logits, targets, axis=1, mask=None)`
- `binary_cross_entropy(probs, targets, weight=None, normalize_by_weight=False)`
- `binary_cross_entropy_with_logits(logits, targets, weight=None, pos_weight=None, normalize_by_weight=False)`
- `one_hot(targets, num_classes=None)`
- `accuracy(logits, targets)`

### `Core.graph`
- `normalize_adjacency(A, add_self_loops=True)`
- `train_val_test_split(n, train=0.6, val=0.2, seed=None)`
- `make_two_community_graph(...)`
- `to_numpy_adjacency(G)`

### `Core.edge_index`
- `EdgeIndex(src, dst, weight, num_nodes)`
- `edge_index_from_adjacency(A, add_self_loops=True)`
- `edge_index_from_pairs(pairs, num_nodes, weight=None, add_self_loops=False)`
- `gcn_normalize_edge_index(edge_index, eps=1e-12)`
- `row_normalize_edge_index(edge_index, eps=1e-12)`

### `Core.structure`
- `normalize_adjacency_tensor(A, add_self_loops=True, eps=1e-12)`
- `init_logits_from_adj(adj, eps=1e-3)`
- `LearnableAdjacency(n_nodes, seed=None, init_adj=None)`

### `Core.optim`
- `SGD(lr=0.05)`
- `Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)`

### `Core.kernels`
- `MessagePassingKernel(A_norm)`
- `EdgeIndexKernel(edge_index)`
- `gcn_message_passing(A_norm, X)`
- `gather_rows(X, idx)`
- `edge_sum(src, dst, X, weight=None, num_nodes=None)`
- `edge_softmax(scores, dst, num_nodes=None, eps=1e-12)`

## Models

- `Models.gcn.GCN`
- `Models.gat.GAT`
- `Models.gat.EdgeGAT`
- `Models.graphsage.GraphSAGE`
- `Models.structure_gcn.StructureGCN`
- `Models.scalable_gcn.ScalableGCN`
- `Models.topology_generator.TopologyGenerator`

Notes:
- `TopologyGenerator` supports pairwise edge scoring via `pairs=(E,2)` and optional `guide_edge_index`.

## Examples

Each example is runnable via `python -m Examples.<module>`:
- `node_classification`
- `node_classification_gat`
- `graph_regression`
- `structure_learning`
- `topology_generation`
