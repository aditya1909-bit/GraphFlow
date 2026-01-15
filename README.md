

# GraphFlow

**GraphFlow** is a minimal, educational framework for building Graph Neural Networks (GNNs) **from scratch** — featuring a tiny reverse‑mode autodiff engine (`Tensor`) and implementations of **GCN**, **GAT**, and **GraphSAGE**.

Everything is written in pure Python + NumPy — no PyTorch or TensorFlow — to show exactly how message passing and gradient propagation work under the hood.

---

## Features

- **Tiny Autograd Engine** — `Tensor` supports reverse‑mode differentiation for scalar and matrix ops.
- **Core Ops** — softmax, cross‑entropy, one‑hot, accuracy.
- **Optimizers** — minimal SGD and Adam.
- **Graph Utilities** — normalization, two‑community synthetic datasets.
- **Structure Learning** — differentiable adjacency with end‑to‑end optimization.
- **Neural Topology Generation** — learn adjacency from node features.
- **Vectorized Kernels** — edge-index message-passing and cached adjacency for efficient computation graphs.
- **Models:**
  - `GCN` — Graph Convolutional Network.
  - `GAT` — Graph Attention Network (multi‑head).
  - `EdgeGAT` — edge-index GAT for scalable attention.
  - `GraphSAGE` — mean aggregator variant.
  - `ScalableGCN` — cached message‑passing kernels.
  - `TopologyGenerator` — neural adjacency generator.
- **Examples:**
  - `node_classification.py` — train a GCN on a toy SBM graph.
  - `graph_regression.py` — predict graph‑level statistics.
  - `structure_learning.py` — learn adjacency + weights end‑to‑end (with adjacency prior).
  - `topology_generation.py` — learn to generate adjacency from features.

---

## Installation

```bash
# clone
$ git clone https://github.com/adityadutta/GraphFlow.git
$ cd GraphFlow

# create virtual environment
$ python3 -m venv .venv
$ source .venv/bin/activate

# install dependencies
$ pip install -r requirements.txt
```

---

## Examples

### Node Classification (GCN)
```bash
python -m Examples.node_classification
```

### Graph Regression (GCN)
```bash
python -m Examples.graph_regression
```

### Node Classification with GAT
```bash
python -m Examples.node_classification_gat
```

### Structure Learning (learn adjacency)
```bash
python -m Examples.structure_learning
```

### Topology Generation (neural adjacency)
```bash
python -m Examples.topology_generation
```

---

## API Documentation

- `docs/API.md` — public API overview for Core, Models, and Examples.

---

## Benchmarks (Reproducible)

```bash
python -m scripts.benchmark_node_classification
python -m scripts.benchmark_topology_generation
```

Benchmarks are deterministic (seeded) and intended for educational comparisons.

---

## Structure Learning Objective

GraphFlow can learn a graph structure end‑to‑end by parameterizing adjacency
as logits (passed through a sigmoid), then optimizing both the adjacency and
GCN weights with a small sparsity penalty and a prior that keeps edges close
to the observed graph.

If you want to peek at the learned structure, you can compute its density:

```python
A = model.adj.adjacency().data
print("adj_density:", A.mean())
```

---

## Project Structure

```
GraphFlow/
├── Core/
│   ├── tensor.py           # minimal autograd engine
│   ├── ops.py              # softmax, cross‑entropy, etc.
│   ├── optim.py            # minimal SGD/Adam optimizers
│   ├── graph.py            # normalization, toy graph gen
│   ├── edge_index.py       # edge-index helpers
│   ├── structure.py        # differentiable adjacency helpers
│   └── kernels.py          # vectorized message-passing kernels
│
├── Models/
│   ├── gcn.py              # Graph Convolutional Network
│   ├── gat.py              # Graph Attention Network
│   ├── graphsage.py        # GraphSAGE mean aggregator
│   ├── structure_gcn.py    # GCN with learnable adjacency
│   ├── scalable_gcn.py     # GCN with cached message passing
│   └── topology_generator.py # adjacency generator
│
├── Examples/
│   ├── node_classification.py
│   ├── graph_regression.py
│   ├── node_classification_gat.py
│   ├── structure_learning.py
│   └── topology_generation.py
│
├── scripts/
│   ├── benchmark_node_classification.py
│   └── benchmark_topology_generation.py
│
└── docs/
    └── API.md
```

---

## Conceptual Overview

GraphFlow implements message‑passing networks using only `Tensor` primitives:

\[
H^{(l+1)} = \sigma(\hat A H^{(l)} W^{(l)})
\]

for GCN, with variants for GAT and GraphSAGE. Each model operates directly on NumPy arrays and builds computation graphs automatically for backpropagation.

---

## Author
**Aditya Dutta** — [@adityadutta](https://github.com/adityadutta)

---

## License
MIT License © 2025 Aditya Dutta
