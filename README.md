

# GraphFlow

**GraphFlow** is a minimal, educational framework for building Graph Neural Networks (GNNs) **from scratch** — featuring a tiny reverse‑mode autodiff engine (`Tensor`) and implementations of **GCN**, **GAT**, and **GraphSAGE**.

Everything is written in pure Python + NumPy — no PyTorch or TensorFlow — to show exactly how message passing and gradient propagation work under the hood.

---

## Features

- **Tiny Autograd Engine** — `Tensor` supports reverse‑mode differentiation for scalar and matrix ops.
- **Core Ops** — softmax, cross‑entropy, one‑hot, accuracy.
- **Graph Utilities** — normalization, two‑community synthetic datasets.
- **Models:**
  - `GCN` — Graph Convolutional Network.
  - `GAT` — Graph Attention Network (multi‑head).
  - `GraphSAGE` — mean aggregator variant.
- **Examples:**
  - `node_classification.py` — train a GCN on a toy SBM graph.
  - `graph_regression.py` — predict graph‑level statistics.

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
python -m GraphFlow.Examples.node_classification
```

### Graph Regression (GCN)
```bash
python -m GraphFlow.Examples.graph_regression
```

### Node Classification with GAT
```bash
python -m GraphFlow.Examples.node_classification_gat
```

---

## Project Structure

```
GraphFlow/
├── Core/
│   ├── tensor.py           # minimal autograd engine
│   ├── ops.py              # softmax, cross‑entropy, etc.
│   └── graph.py            # normalization, toy graph gen
│
├── Models/
│   ├── gcn.py              # Graph Convolutional Network
│   ├── gat.py              # Graph Attention Network
│   └── graphsage.py        # GraphSAGE mean aggregator
│
└── Examples/
    ├── node_classification.py
    ├── graph_regression.py
    └── node_classification_gat.py
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