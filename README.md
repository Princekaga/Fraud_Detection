# Fraud Detection with Graph Neural Networks
### Uncovering Illicit Financial Networks on the Elliptic Bitcoin Dataset

Anti-money-laundering (AML) as **graph representation learning**: Bitcoin transactions are nodes, payment flows are edges, and GNNs (GCN / GraphSAGE / GAT, PyTorch Geometric) learn to spot illicit activity from both transaction features and multi-hop network structure — benchmarked against an XGBoost tabular baseline under a deployment-faithful temporal split.

## Headline results (test window t = 35–49, thresholds tuned on validation only)

| Model | PR-AUC | Macro F1 | Illicit F1 | Precision | Recall | FPR % |
|---|---|---|---|---|---|---|
| XGBoost (166 feats, baseline) | **0.789** | 0.871 | 0.758 | 0.785 | **0.733** | 1.39 |
| GCN (all features) | 0.275 | 0.676 | 0.406 | 0.330 | 0.529 | 7.47 |
| GAT (all features) | 0.382 | 0.723 | 0.490 | 0.413 | 0.600 | 5.92 |
| GraphSAGE (all features) | 0.636 | 0.773 | 0.577 | 0.551 | 0.607 | 3.44 |
| GraphSAGE (94 local feats only) | 0.681 | 0.827 | 0.675 | 0.735 | 0.623 | 1.56 |
| **Hybrid: SAGE embeddings + XGBoost** | 0.764 | **0.880** | **0.773** | **0.953** | 0.650 | **0.22** |

Two findings worth stealing:
1. **Structure is learned, not engineered** — GraphSAGE given only the 94 local features *beats* the same model given all 166 (which include 72 hand-engineered neighbourhood aggregates).
2. **GNN + trees is the winning combo** — feeding GraphSAGE embeddings into XGBoost cuts the false-positive rate **6×** (1.39 % → 0.22 %) at comparable recall: ~84 % fewer wasted investigations.

## Repository layout
```
├── GNN_Fraud_Detection.ipynb   # executed end-to-end notebook (start here)
├── REPORT.md / REPORT.docx     # comprehensive project report
├── report/figures/             # all figures
├── src/
│   ├── data.py                 # loading, preprocessing, temporal split
│   ├── models.py               # GCN / GraphSAGE / GAT + Focal Loss
│   ├── metrics.py              # PR-AUC, F1, FPR, threshold tuning
│   ├── train.py                # GNN training CLI
│   ├── train_xgb.py            # XGBoost baseline + hybrid CLI
│   ├── predict.py              # inference on a (new) graph snapshot
│   └── explain.py              # GNNExplainer evidence for a flagged tx
├── models/                     # trained weights (GNN .pt, XGBoost .json)
└── results/                    # metrics JSONs incl. summary.json
```

## Setup

```bash
pip install -r requirements.txt
```

**Dataset** (not committed — 690 MB): download the [Elliptic Data Set from Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and extract so that `elliptic_bitcoin_dataset/` sits in the repo root with its three CSVs (`elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`).

## Usage

```bash
# train a GNN (GCN | GraphSAGE | GAT) on all features (AF) or local-only (LF)
python src/train.py --model GraphSAGE --featset LF

# XGBoost baseline, then hybrid (needs trained GraphSAGE-LF weights)
python src/train_xgb.py
python src/train_xgb.py --hybrid

# score a snapshot with the trained hybrid pipeline -> results/predictions.csv
python src/predict.py --timestep 49 --threshold 0.5

# explain a flagged transaction (evidence subgraph + feature attribution)
python src/explain.py
```

Pre-trained weights ship in `models/`, so `predict.py` and `explain.py` work immediately after downloading the dataset. The notebook reproduces everything from scratch (≈ 45–60 min on CPU).

## Method summary
- **Preprocessing:** txId-safe parsing, label encoding (illicit/licit/unknown), contiguous re-indexing, undirected `edge_index`, temporal split (train 1–29 / val 30–34 / test 35–49).
- **Models:** 2-layer GNNs (hidden 64, dropout 0.5) trained full-batch with **Focal Loss** (α 0.75, γ 2), Adam, early stopping on validation PR-AUC. XGBoost with `scale_pos_weight` and `aucpr` early stopping.
- **Evaluation:** PR-AUC, macro-F1, illicit precision/recall, FPR; per-time-step F1 exposes concept drift after the dark-market shutdown (t = 43).
- **Explainability:** GNNExplainer masks over 2-hop subgraphs give analyst-reviewable evidence.

See [REPORT.md](REPORT.md) for the full write-up and sources.
