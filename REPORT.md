# Leveraging Graph Neural Networks to Uncover Illicit Financial Networks
### Comprehensive Project Report — Anti-Money-Laundering on the Elliptic Bitcoin Dataset

---

## 1. Members of the Team
- **Abhinav Mishra** (solo project)

## 2. Overall Project Summary
Anti-money-laundering (AML) systems built on rule engines or tabular machine learning evaluate each transaction in isolation, ignoring the network of counterparties around it. This project reframes fraud detection on the Elliptic Bitcoin dataset as a **graph representation learning** problem: transactions are nodes, payment flows are edges, and Graph Neural Networks (GNNs) learn embeddings that mix a transaction's own features with multi-hop network context through message passing.

We benchmark three GNN architectures (GCN, GraphSAGE, GAT) trained with Focal Loss against a strong XGBoost tabular baseline, under a strict **temporal split** that mimics production deployment. Two headline findings:

1. **Hand-engineered graph features are learnable.** A GraphSAGE model given only the 94 transaction-local features (none of the 72 manually aggregated neighbourhood features) outperforms the same model given all 166 features (test PR-AUC 0.681 vs 0.636) — the network learns neighbourhood structure directly from the graph.
2. **A hybrid GraphSAGE-embedding + XGBoost model gives the best operating point**: illicit-class F1 **0.773** at **95.3 % precision** and a false-positive rate of **0.22 %** — a **~6× FPR reduction** versus the XGBoost baseline (1.39 %) at comparable recall, directly reducing wasted investigator effort.

All models degrade sharply after the dark-market shutdown at time step 43, quantifying how concept drift — not model capacity — is the binding constraint for production AML.

## 3. Details of the Dataset
**Source:** Elliptic Data Set (Kaggle), the benchmark for graph-based financial anomaly detection, introduced in Weber et al. (KDD 2019).

| Property | Value |
|---|---|
| Nodes (Bitcoin transactions) | 203,769 |
| Directed edges (payment flows) | 234,355 |
| Node features | 166 (anonymised, standardised) |
| Time steps | 49 (≈ 2 weeks each); every edge connects nodes within the same step → 49 disjoint graph snapshots |
| Labels | 4,545 illicit (2.2 %) · 42,019 licit (20.6 %) · 157,205 unknown (77.2 %) |
| Avg / max degree | 2.30 / 473, no isolated nodes |

**Feature structure:** column 1 is the time step; columns 2–95 are **94 local features (LF)** describing the transaction itself (fees, volumes, input/output counts, etc.); columns 96–167 are **72 aggregated features (AF)** — hand-engineered one-hop neighbourhood statistics (max/min/std/correlation of neighbour features). Class `1` = illicit, `2` = licit.

Files: `elliptic_txs_features.csv` (690 MB), `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`.

## 4. Details of Data Preprocessing
1. **Type-safe parsing** — feature matrix read as `float32`; `txId` parsed as integer (reading IDs as float32 silently corrupts them — e.g. 230425980 → 230425984).
2. **Label encoding** — `1`→1 (illicit), `2`→0 (licit), `unknown`→−1. Unknown nodes are excluded from all losses and metrics but **kept in the graph** so they relay messages between labelled nodes (semi-supervised setting).
3. **Node re-indexing** — raw 9-digit `txId`s mapped to contiguous indices 0…N−1; the edge list becomes a `(2, 234355)` int64 `edge_index` tensor (PyTorch Geometric format).
4. **Undirected message passing** — reverse edges added (suspicion propagates both up- and downstream of a payment), matching the original Elliptic paper's setup.
5. **No re-scaling** — features ship already standardised by Elliptic.
6. **Temporal split** — train t = 1–29 (26,381 labelled nodes), validation t = 30–34 (3,513; early stopping + threshold tuning), test t = 35–49 (16,670). A random split would leak future information; the temporal split reproduces the deployment reality that models score *future* transactions.

## 5. Feature Extraction and Weighting Methods
- **Feature sets:** AF = all 166 features; LF = 94 local features only. The LF ablation tests whether the GNN can replace the 72 manual aggregates.
- **Learned structural features:** the trained GraphSAGE-LF model's 64-dimensional penultimate-layer embeddings are extracted and concatenated with the 166 raw features to feed the hybrid XGBoost model (166 + 64 = 230 dims).
- **Class weighting:** GNNs use **Focal Loss** (α = 0.75 on the illicit class, γ = 2) so the 2 % minority class dominates gradients; XGBoost uses `scale_pos_weight = N_licit/N_illicit ≈ 8.2`.
- **Decision thresholds** are tuned on the validation window (max illicit F1) and frozen before testing — no test-set leakage.

## 6. Libraries Used
| Library | Version | Role |
|---|---|---|
| torch | 1.13.1 | tensor ops, autograd, training |
| torch-geometric | 2.3.1 | GCN/SAGE/GAT layers, GNNExplainer, graph utilities |
| xgboost | 3.2.0 | tabular baseline + hybrid classifier |
| scikit-learn | 1.7.2 | metrics (PR-AUC, F1, confusion), PR curves |
| pandas | 2.3.3 / numpy 1.26.4 | data wrangling |
| matplotlib | 3.10.9 / networkx 3.4.2 | figures, subgraph visualisation |
| nbformat | 5.10.4 | notebook tooling |

## 7. Architecture of the Models Employed
All GNNs: 2 layers (2-hop receptive field), hidden width 64, dropout 0.5, linear classification head, Adam (lr 5e-3, weight-decay 5e-4), full-batch transductive training, early stopping on validation PR-AUC (patience 40).

- **GCN** — `GCNConv(166→64) → ReLU → dropout → GCNConv(64→64) → ReLU → dropout → Linear(64→2)`. Symmetric-normalised mean aggregation.
- **GraphSAGE** — same skeleton with `SAGEConv` (mean aggregator): `h′ = W₁h + W₂·mean(neighbours)`. The separate self-transform preserves local evidence, and the architecture is **inductive** — it embeds unseen nodes, as required for real-time screening; scales via neighbourhood-sampled mini-batches (`NeighborLoader`) on larger graphs.
- **GAT** — two `GATConv` layers with 4 attention heads each (64 total channels), ELU activations, attention dropout 0.3. Attention lets the model weight suspicious neighbours above benign ones.
- **Focal Loss** — `FL = α_t (1−p_t)^γ · CE`, α_illicit = 0.75, γ = 2.
- **XGBoost baseline** — 400 trees (early-stopped on validation `aucpr`), depth 6, lr 0.1, `hist` method.
- **Hybrid** — XGBoost with identical hyperparameters over `[166 raw features ‖ 64 GraphSAGE-LF embeddings]`.
- **Explainability** — GNNExplainer (100 epochs) over the 2-hop subgraph of flagged transactions produces edge-importance masks and feature attributions.

## 8. Final Results
Test window t = 35–49 (16,670 labelled transactions, 6.5 % illicit). Thresholds tuned on validation only.

| Model | PR-AUC | Macro F1 | Illicit F1 | Precision | Recall | FPR % |
|---|---|---|---|---|---|---|
| XGBoost (166 feats, baseline) | **0.789** | 0.871 | 0.758 | 0.785 | **0.733** | 1.39 |
| GCN (AF) | 0.275 | 0.676 | 0.406 | 0.330 | 0.529 | 7.47 |
| GAT (AF) | 0.382 | 0.723 | 0.490 | 0.413 | 0.600 | 5.92 |
| GraphSAGE (AF) | 0.636 | 0.773 | 0.577 | 0.551 | 0.607 | 3.44 |
| GraphSAGE (LF-94) | 0.681 | 0.827 | 0.675 | 0.735 | 0.623 | 1.56 |
| **Hybrid SAGE-emb + XGBoost** | 0.764 | **0.880** | **0.773** | **0.953** | 0.650 | **0.22** |

Key observations:
- **Hybrid wins the operating point that matters:** highest illicit F1 and macro F1, 95 % alert precision, and a 6× lower FPR than the baseline — of every 100 alerts raised, ~95 are true hits vs ~79 for XGBoost.
- **GraphSAGE ≫ GCN/GAT** on this dataset: GCN's normalisation dilutes self-features; GraphSAGE's explicit self-transform preserves them.
- **LF ablation:** removing the 72 engineered features *improves* GraphSAGE (0.636 → 0.681 PR-AUC) — the GNN learns the structure itself and avoids overfitting to training-era neighbourhood statistics.
- **Concept drift dominates:** per-time-step F1 collapses for *all* models after the dark-market shutdown (t = 43), with near-zero illicit detection for t ≥ 44.
- **GNNExplainer** produces analyst-readable evidence: for a sampled flagged transaction (p = 0.91), the decisive subgraph links it to a cluster of known-illicit transactions, with time step and volume-related local features carrying the largest attributions.

## 9. Conclusions, Improvements over Legacy Approaches, and Limitations
**Conclusions.** Graph structure carries real, learnable signal for AML: GNN embeddings encode neighbourhood context that (a) replaces manual feature engineering and (b) sharpens a tree-based classifier into a far more precise alerting system. Pure GNNs, however, do not beat a well-tuned XGBoost on curated features — the practical recipe is **GNN for structure, trees for decision boundaries**.

**Improvements over legacy approaches.**
- Drops the I.I.D. assumption — decisions use who-transacts-with-whom context, catching ring structures invisible to row-wise models.
- Eliminates the feature-engineering bottleneck — the LF experiment shows multi-hop structure is learned automatically.
- False-positive reduction — FPR 1.39 % → 0.22 % at similar recall, i.e. ~84 % fewer false alerts to investigate.
- Inductive scoring (GraphSAGE) — new transactions are embedded without retraining; mini-batch neighbourhood sampling provides the scaling path to million-node graphs.
- Auditability — GNNExplainer supplies the subgraph-plus-features evidence trail that compliance review requires.

**Limitations.**
- **Concept drift:** static models fail on post-shutdown patterns; production systems need continual retraining or temporal GNNs (EvolveGCN, TGN).
- **Anonymised features** cap the semantic interpretability of attributions.
- **77 % unlabeled nodes** are used only passively; self-supervised pre-training or label propagation could extract more from them.
- **Full-batch training** was sufficient here but is not the scaling path; the GraphSAGE mini-batch pipeline is.
- Threshold choice is deployment-specific: banks trading recall for precision (or vice versa) should re-tune on their own validation stream.

## 10. Sources
1. M. Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*, KDD Workshop 2019, arXiv:1908.02591.
2. T. Kipf & M. Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017, arXiv:1609.02907.
3. W. Hamilton, R. Ying, J. Leskovec, *Inductive Representation Learning on Large Graphs*, NeurIPS 2017, arXiv:1706.02216.
4. P. Veličković et al., *Graph Attention Networks*, ICLR 2018, arXiv:1710.10903.
5. T.-Y. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017, arXiv:1708.02002.
6. R. Ying et al., *GNNExplainer: Generating Explanations for Graph Neural Networks*, NeurIPS 2019, arXiv:1903.03894.
7. Elliptic Data Set, Kaggle: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
8. PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io
