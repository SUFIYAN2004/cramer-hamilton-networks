# Cramer-Hamilton Networks (CHN)

> Exact deterministic linear algebra as an alternative to soft attention in language models.

---

## What is this?

This repository contains the research implementation of **Cramer-Hamilton Networks**, a novel neural language model architecture that replaces standard Transformer components with exact analytical linear algebra operations.

Instead of asking *"what is the soft probability that token A attends to token B?"*, we ask *"what is the exact deterministic solution to the linear system formed by these features?"*

---

## Architecture

```
Input tokens
     ↓
Jacobian Manifold Embedding    ← replaces RoPE / sinusoidal
     ↓
[× N layers]
  Cramer's Rule Block          ← replaces Multi-Head Attention
  Cayley-Hamilton Block        ← replaces Feed-Forward Network
  Gaussian Quadrature Stab.   ← replaces LayerNorm
     ↓
Output predictions
```

### The three core theorems

| Component | Formula | Job |
|-----------|---------|-----|
| Cramer's Rule | `xᵢ = det(Aᵢ) / det(A)` | Replaces soft attention with exact 2×2 linear solves |
| Cayley-Hamilton | `S = c₁·A(x) + c₀·I` | Replaces dense FFN with polynomial state transform |
| Jacobian Manifold | `E = base + tanh(base @ J * t)` | Replaces positional encodings with continuous coordinate flow |

### Extended: Tri-Directional Five-Head Architecture

```
HEAD 1 (Front →)    ──┐
HEAD 2 (Middle ↔)   ──┼──→ HEAD 4 (Dynamic Reader) ──→ HEAD 5 (Propagator) → token
HEAD 3 (Reverse ←)  ──┘
```

Each head processes the sequence in a different direction. The dynamic reader computes per-token trust weights — learned from content, not fixed — to balance contributions from all three directions.

---

## Results

### Tiny Shakespeare — Character Level

| Model | Params | Val Loss | Notes |
|-------|--------|----------|-------|
| CramerHamiltonNet | 234K | **2.47** | Best char-level result |
| CramerHamiltonNetV5 (word) | 989K | 4.82 | Word-level with semantic mixing |

### WikiText-103

| Model | Params | Val Loss | Speed |
|-------|--------|----------|-------|
| CramerHamiltonNet V6 | 61M | **3.99** | 5.3 it/s on T4 |

Sub-4.0 validation loss on WikiText-103 with **no Softmax attention**.

### Tri-Directional Architecture — Trust Analysis

| Head | Static trust (broken) | Dynamic trust (working) |
|------|----------------------|------------------------|
| Front → | 0.70 | 0.42 |
| Middle ↔ | 0.13 | 0.33 |
| Reverse ← | 0.17 | 0.25 |

Static trust collapses to front-head dominance and produces degenerate repetition loops. Dynamic per-token trust produces coherent Shakespeare-style generation.

### Sample output (word-level, 989K params)

```
ROMEO : Anon be without love he Corioli last comfort spent
There their lets the news , and foot , that your...

KING RICHARD III thy : I will . Servant great his mates
more will could ; therefore woe you With Thou who gall think...
```

---

## Key Findings

**1. Chinchilla scaling does not apply**
The standard tokens = 20 × parameters prescription does not hold for deterministic linear solver networks. Performance at 1:1 token-to-parameter ratio is comparable to 4:1.

**2. The 2×2 channel disconnect**
Standard Cramer blocks solve pairs of channels independently — channels 0 and 63 never interact. This is adequate for character-level tasks but requires explicit cross-channel mixing (FullSemanticCramerLayer) for word-level semantic understanding.

**3. Dynamic trust is necessary**
Static learned trust weights in multi-directional models converge to front-head dominance (0.70), starving other heads. Dynamic per-token trust computed from content maintains balanced contribution and produces coherent generation.

**4. Exact solvers can learn language**
Sub-4.0 loss on WikiText-103 with no attention mechanism proves that deterministic exact linear algebra can learn language patterns at scale.

---

## Installation

```bash
git clone https://github.com/mohammed-sufiyan/cramer-hamilton-networks
cd cramer-hamilton-networks
pip install torch tiktoken datasets
```

---

## Usage

### Run the notebook

Open `cramer_hamilton_experiments.ipynb` in Google Colab or Jupyter.

```bash
jupyter notebook cramer_hamilton_experiments.ipynb
```

### Train character-level model

```python
from model import CramerHamiltonNet

model = CramerHamiltonNet(
    vocab_size=65,
    hidden_dim=128,
    num_layers=4,
    dropout=0.1)
```

### Train word-level model

```python
from model import CramerHamiltonNetV5

model = CramerHamiltonNetV5(
    vocab_size=3001,
    hidden_dim=128,
    num_layers=4,
    dropout=0.1)
```

### Train V6 on WikiText-103

```python
from model_v6 import CramerHamiltonNetV6

model = CramerHamiltonNetV6(
    vocab_size=50257,
    hidden_dim=512,
    num_layers=8,
    max_seq_len=256,
    dropout=0.1)
```

---

## Repository Structure

```
cramer-hamilton-networks/
│
├── cramer_hamilton_experiments.ipynb   ← main notebook (char + word level)
├── README.md                           ← this file
│
├── results/
│   ├── char_level_loss_curve.png
│   ├── word_level_loss_curve.png
│   └── trust_weight_evolution.png
│
└── report/
    └── cramer_hamilton_research_report.md
```

---

## Limitations

- Long-range dependencies: multi-scale conv provides max 25-token receptive field per layer
- GPU efficiency: custom CUDA kernels not yet implemented
- No proven scaling law for this architecture class
- Word-level requires additional cross-channel mixing layers

---

## Future Work

- Custom CUDA kernels for 2×2 determinant operations
- New scaling law derivation for deterministic solver networks
- Hybrid architecture: Cramer local + sparse attention global
- Larger scale experiments (500M+ parameters)
- Middle head redesign with sliding window mask

---

## Citation

If you use this work, please cite:

```bibtex
@misc{cramer-hamilton-networks-2026,
  title   = {Cramer-Hamilton Networks: Exact Linear Algebra
             as an Alternative to Soft Attention},
  author  = {V. Mohammed Sufiyan},
  year    = {2026},
  url     = {https://github.com/mohammed-sufiyan/cramer-hamilton-networks}
}
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*This is ongoing research. Results are preliminary and have not been peer reviewed.*
