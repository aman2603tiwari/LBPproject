# Smart Contract Vulnerability Detection using Graph Neural Networks

> **Binary Classification:** F1 = 0.94 · Accuracy = 0.98  
> **Multiclass Classification (6 classes):** Macro F1 = 0.80 · Accuracy = 0.88  
> **vs. Slither & Mythril (industry tools):** F1 = 0.03

An end-to-end automated pipeline that parses Solidity smart contracts into structural graphs and trains a Graph Attention Network (GAT) to classify contracts into six vulnerability categories — outperforming industry-standard static analysis tools by **26.7×** on Macro F1.

---

## Table of Contents

- [Motivation](#motivation)
- [Core Idea](#core-idea)
- [Pipeline Overview](#pipeline-overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Limitations & Future Work](#limitations--future-work)

---

## Motivation

Smart contracts are immutable once deployed. A single vulnerability cannot be patched after the fact — which makes pre-deployment detection critical and failure catastrophically expensive.

| Hack / Exploit | Year | Vulnerability | Loss (USD) |
|---|---|---|---|
| The DAO | 2016 | Reentrancy | $60M |
| Parity Multisig | 2017 | Access Control | $30M |
| Wormhole Bridge | 2022 | Logic Error | $320M |
| Euler Finance | 2023 | Flash Loan | $197M |

> Total DeFi losses from smart contract exploits exceed **$5 billion** as of 2024.

Existing tools fall short:
- **Slither** — rule-based static analysis; misses novel variants
- **Mythril** — symbolic execution; times out on large contracts
- **Manual audit** — $10,000–$100,000 per contract; not scalable

---

## Core Idea

Vulnerabilities are **structural patterns** in code:
- Reentrancy creates a specific shape in the call graph
- Integer overflow appears at arithmetic nodes without bounds checks
- Access control failures leave unguarded state transitions

GNNs are architecturally designed to detect patterns in graph structure. Rather than writing explicit rules (like Slither), the model **learns these patterns from real exploited contracts** and generalises to new variants.

---

## Pipeline Overview

```
Phase 1: Dataset      →  495-row hack database (DeFiHackLabs, ArXiv, Reddit, News)
Phase 2: Contracts    →  175 labelled .sol files (Etherscan v2 + SmartBugs + SWC)
Phase 3A: Graphs      →  175 graph JSON files (AST + CFG + PDG per contract)
Phase 3B: Training    →  175 → 811 augmented graphs → GAT model checkpoint
Phase 4: Evaluation   →  GNN vs. Slither vs. Mythril comparison report
```

| Stage | Script | Key Output |
|---|---|---|
| Dataset scraping | `scraper.py`, `hack.py` | `hack_database.csv` (495 rows) |
| Contract collection | `collect_contracts.py` | 175 `.sol` files |
| SmartBugs augmentation | `get_smartbugs.py` | +133 contracts |
| Graph construction | `build_graphs.py` | `graphs/*.json` |
| Graph augmentation | `augment_graphs.py` | `graphs_augmented/*.json` (811 graphs) |
| Model training | `train_gnn.py` | `gnn_multiclass_best.pt` |
| Evaluation | `evaluate_industry.py` | `comparison_report.txt`, charts |

---

## Results

### Binary Classification (Safe vs. Vulnerable)

| Metric | Score |
|---|---|
| Accuracy | **0.9755** |
| F1 (weighted) | **0.9375** |

### Multiclass Classification (6 Vulnerability Types)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| logic_error | 0.98 | 0.91 | **0.94** |
| safe | 0.89 | 0.89 | 0.89 |
| access_control | 0.80 | 0.80 | 0.80 |
| reentrancy | 0.71 | 0.86 | 0.77 |
| other_vuln | 0.71 | 0.83 | 0.77 |
| integer_overflow | 0.56 | 0.71 | 0.62 |
| **Macro avg** | 0.77 | 0.83 | **0.80** |

### Industry Tool Comparison

| Tool | Binary F1 | Macro F1 | Coverage |
|---|---|---|---|
| **GNN (ours)** | **0.9375** | **0.8003** | 175/175 |
| Slither | 0.0135 | 0.0263 | 175/175 |
| Mythril | 0.0135 | 0.0263 | 175/175 |

> Slither and Mythril predicted "safe" for **every single contract** in the benchmark. The GNN's version-agnosticism (operating on pre-compiled graphs) is a genuine architectural advantage over tools that require modern Solidity pragma support.

---

## Project Structure

```
smart-contract-gnn/
├── dataset_output/
│   ├── hack_database.csv          # 495-row vulnerability database
│   └── news_*.csv                 # Scraped news articles
├── contracts/
│   ├── vulnerable/                # Labelled vulnerable .sol files
│   └── safe/                      # Audited safe contracts
├── graphs/                        # Raw graph JSONs (175 files)
├── graphs_augmented/              # Augmented graphs (811 files)
├── models/
│   ├── gnn_multiclass_best.pt     # Trained GAT checkpoint
│   └── gnn_config.json            # Architecture config for inference
├── results/
│   ├── comparison_report.txt      # GNN vs Slither vs Mythril
│   └── *.png                      # Training curves, confusion matrices
├── scraper.py                     # RSS news scraping
├── hack.py                        # DeFiHackLabs scraping
├── add_scam_type.py               # Label enrichment
├── collect_contracts.py           # Etherscan v2 API downloader
├── get_smartbugs.py               # SmartBugs + SWC downloader
├── build_graphs.py                # .sol → graph JSON (AST/CFG/PDG)
├── augment_graphs.py              # Graph augmentation pipeline
├── train_gnn.py                   # GAT model training
├── evaluate_industry.py           # Benchmarking script
└── regenerate_results.py          # Rebuild charts without re-running tools
```

---

## Installation

```bash
git clone https://github.com/<your-username>/smart-contract-gnn.git
cd smart-contract-gnn

pip install -r requirements.txt

# Install Slither and Mythril for benchmarking (optional)
pip install slither-analyzer mythril
```

**Key dependencies:** `torch`, `torch-geometric`, `py-solc-x`, `requests`, `pandas`, `matplotlib`

---

## Usage

### Run the full pipeline

```bash
# 1. Build hack database
python hack.py
python scraper.py

# 2. Download contracts
python collect_contracts.py
python get_smartbugs.py

# 3. Build graphs
python build_graphs.py

# 4. Augment and train
python augment_graphs.py
python train_gnn.py

# 5. Evaluate against industry tools
python evaluate_industry.py
```

### Run inference on a new contract

```python
import json
import torch
from train_gnn import GATClassifier

config = json.load(open("models/gnn_config.json"))
model = GATClassifier(**config)
model.load_state_dict(torch.load("models/gnn_multiclass_best.pt"))
model.eval()

# Build a graph from your .sol file, then:
# prediction = model(graph.x, graph.edge_index, graph.batch)
```

---

## Model Architecture

A **Graph Attention Network (GAT)** was chosen over GCN because attention heads learn which neighbouring nodes matter more — an external `.call()` edge carries fundamentally different information than a variable declaration edge.

```
Input: Node feature matrix [N, 16]
  ↓
GAT Layer 1: 4-head attention, hidden_dim=128  →  [N, 512]
  ↓
GAT Layer 2: 1-head attention, hidden_dim=128  →  [N, 128]
  ↓
Global Mean Pool  →  [1, 128]
  ↓
BatchNorm + FC (128→64, ReLU)  →  [1, 64]
  ↓
BatchNorm + FC (64→6)  →  [1, 6]
  ↓
Softmax  →  Class probabilities
```

Each node carries a **16-dimensional feature vector** encoding: node type, presence of `.call()`/`.send()`/`delegatecall`, state modification flag, visibility, AST depth, child count, keyword frequencies, loop counts, `require`/`revert` patterns, and `payable` flag.

**Graph types per contract:** AST (Abstract Syntax Tree), CFG (Control Flow Graph), PDG (Program Dependency Graph)

**Training details:** Adam optimiser, LR=1e-3, cosine annealing scheduler, dropout=0.3, gradient clipping=1.0, early stopping (patience=20), stratified 70/12/18 train/val/test split.

---

## Dataset

- **495-row hack database** aggregated from DeFiHackLabs (469 records), Google News (428 articles), ArXiv (28 papers), and Reddit (35 posts)
- **175 labelled Solidity contracts** from Etherscan v2 API (real exploited protocols) and SmartBugs-Curated + SWC Registry (academic benchmarks)
- **811 training graphs** after class-balanced augmentation (Gaussian noise, feature masking, edge dropout)

| Vulnerability Type | Contracts |
|---|---|
| logic_error | 77 |
| reentrancy | 35 |
| access_control | 21 |
| integer_overflow | 18 |
| safe | 15 |
| other_vuln (merged) | 9 |

---

## Limitations & Future Work

**Current limitations:**
- 175 real contracts (811 augmented) is small by production ML standards
- Regex fallback parser produces lower-quality graphs than full `solc` AST compilation
- Static analysis only — runtime-dependent exploits (complex flash loan sequences) may not be detectable from code structure alone
- Single-file analysis; multi-file contracts are concatenated rather than properly resolved

**Planned improvements:**
- Integrate Slither's SlithIR as a richer graph construction source
- Expand to 1,000+ contracts using DeFiVulnLabs PoC repository
- Add cross-contract analysis for multi-contract DeFi protocols
- Implement incremental learning for emerging vulnerability types
- Deploy as a pre-commit hook or CI/CD integration

---

## Citation

```
Tiwari, A. (2026). Smart Contract Vulnerability Detection Using Graph Neural Networks.
LBP Project Technical Report.
```

---

*Built by Aman Tiwari · LBP Project · 2026*
