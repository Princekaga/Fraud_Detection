"""XGBoost baseline (tabular, no graph) and hybrid (raw features + GraphSAGE embeddings).

Usage:
    python src/train_xgb.py --data-dir elliptic_bitcoin_dataset            # baseline
    python src/train_xgb.py --hybrid --sage-weights models/GraphSAGE_LF.pt # hybrid
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from data import load_elliptic, temporal_masks
from metrics import evaluate
from models import GraphSAGE


def main(data_dir="elliptic_bitcoin_dataset", hybrid=False,
         sage_weights="models/GraphSAGE_LF.pt", out_models="models", out_results="results"):
    d = load_elliptic(data_dir)
    X, y_t, ts = d["x"].numpy(), d["y"], d["timestep"]
    tr_m, va_m, te_m = temporal_masks(y_t, ts)
    y = y_t.numpy(); tr, va, te = tr_m.numpy(), va_m.numpy(), te_m.numpy()

    if hybrid:
        sage = GraphSAGE(94)
        sage.load_state_dict(torch.load(sage_weights)); sage.eval()
        with torch.no_grad():
            sage(d["x"][:, :94], d["edge_index"])
        X = np.concatenate([X, sage.embeddings.numpy()], axis=1)
        tag = "hybrid_AF"
    else:
        tag = "xgboost_AF"

    spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    t0 = time.time()
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            scale_pos_weight=spw, tree_method="hist", n_jobs=-1,
                            eval_metric="aucpr", early_stopping_rounds=30)
    clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    prob = clf.predict_proba(X)[:, 1]
    os.makedirs(out_models, exist_ok=True); os.makedirs(out_results, exist_ok=True)
    clf.save_model(f"{out_models}/{tag}.json")
    np.save(f"{out_results}/prob_{tag}.npy", prob)
    res = {"model": tag, "scale_pos_weight": round(spw, 2),
           "train_time_s": round(time.time() - t0, 1),
           "test": evaluate(y[te], prob[te])}
    json.dump(res, open(f"{out_results}/{tag}.json", "w"), indent=2)
    print(json.dumps(res["test"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="elliptic_bitcoin_dataset")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--sage-weights", default="models/GraphSAGE_LF.pt")
    a = ap.parse_args()
    main(a.data_dir, a.hybrid, a.sage_weights)
