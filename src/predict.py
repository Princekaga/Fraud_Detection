"""Score transactions with the trained hybrid pipeline (GraphSAGE-LF embeddings + XGBoost).

Usage:
    python src/predict.py --data-dir elliptic_bitcoin_dataset --timestep 49 --threshold 0.5
GraphSAGE is inductive, so this works on new, unseen graph snapshots with the same
166-feature schema: pass features + within-snapshot edges.
"""
import argparse, os, sys
import numpy as np
import torch
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from data import load_elliptic
from models import GraphSAGE


def score(X_new, edge_index_new, sage_weights="models/GraphSAGE_LF.pt",
          xgb_model="models/hybrid_AF.json"):
    """X_new: (N,166) float32; edge_index_new: (2,E) int64 directed edges.
    Returns per-transaction illicit probability."""
    xx = torch.from_numpy(np.asarray(X_new, dtype=np.float32))
    ei = torch.from_numpy(np.asarray(edge_index_new, dtype=np.int64))
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    sage = GraphSAGE(94)
    sage.load_state_dict(torch.load(sage_weights)); sage.eval()
    with torch.no_grad():
        sage(xx[:, :94], ei)
    clf = xgb.XGBClassifier(); clf.load_model(xgb_model)
    feats = np.concatenate([xx.numpy(), sage.embeddings.numpy()], axis=1)
    return clf.predict_proba(feats)[:, 1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="elliptic_bitcoin_dataset")
    ap.add_argument("--timestep", type=int, default=49, help="snapshot to score as if newly arrived")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/predictions.csv")
    a = ap.parse_args()

    d = load_elliptic(a.data_dir)
    m = d["timestep"] == a.timestep
    nodes = np.where(m)[0]
    remap = {n: i for i, n in enumerate(nodes)}
    half = d["edge_index"][:, :d["edge_index"].shape[1] // 2].numpy()  # original direction
    em = m[half[0]] & m[half[1]]
    ei_new = np.stack([[remap[s] for s in half[0][em]], [remap[t] for t in half[1][em]]])
    prob = score(d["x"].numpy()[m], ei_new)

    flagged = (prob >= a.threshold).sum()
    print(f"scored {m.sum()} transactions from t={a.timestep}; {flagged} flagged (threshold {a.threshold})")
    import csv
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["txId", "illicit_probability", "flagged"])
        for n, p in zip(nodes, prob):
            w.writerow([d["tx_ids"][n], f"{p:.6f}", int(p >= a.threshold)])
    print("wrote", a.out)
