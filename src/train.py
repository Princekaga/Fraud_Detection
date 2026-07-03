"""Train GNNs (GCN / GraphSAGE / GAT) on the Elliptic dataset.

Usage:
    python src/train.py --model GraphSAGE --featset AF --data-dir elliptic_bitcoin_dataset
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
from data import load_elliptic, temporal_masks
from metrics import evaluate
from models import MODELS, FocalLoss


def train(model_name="GraphSAGE", featset="AF", data_dir="elliptic_bitcoin_dataset",
          hidden=64, epochs=300, lr=5e-3, weight_decay=5e-4, patience=40, eval_every=5,
          out_models="models", out_results="results", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    d = load_elliptic(data_dir)
    x, ei, y, ts = d["x"], d["edge_index"], d["y"], d["timestep"]
    if featset == "LF":
        x = x[:, :94]                      # time step + 93 local features
    tr, va, te = temporal_masks(y, ts)

    model = MODELS[model_name](x.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = FocalLoss(alpha=0.75, gamma=2.0)
    best_val, best_state, wait, t0 = -1.0, None, 0, time.time()

    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        loss = crit(model(x, ei)[tr], y[tr])
        loss.backward(); opt.step()
        if epoch % eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                prob = F.softmax(model(x, ei), dim=1)[:, 1].numpy()
            val_ap = average_precision_score(y[va].numpy(), prob[va.numpy()])
            if val_ap > best_val:
                best_val, wait = float(val_ap), 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += eval_every
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  val PR-AUC {val_ap:.4f}  best {best_val:.4f}")
            if wait >= patience:
                break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        prob = F.softmax(model(x, ei), dim=1)[:, 1].numpy()
    tag = f"{model_name}_{featset}"
    os.makedirs(out_models, exist_ok=True); os.makedirs(out_results, exist_ok=True)
    torch.save(best_state, f"{out_models}/{tag}.pt")
    np.save(f"{out_results}/prob_{tag}.npy", prob)
    res = {"model": model_name, "featset": featset, "hidden": hidden, "epochs_run": epoch,
           "val_pr_auc": best_val, "train_time_s": round(time.time() - t0, 1),
           "test": evaluate(y[te].numpy(), prob[te.numpy()])}
    json.dump(res, open(f"{out_results}/{tag}.json", "w"), indent=2)
    print(json.dumps(res["test"], indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GraphSAGE", choices=["GCN", "GraphSAGE", "GAT"])
    ap.add_argument("--featset", default="AF", choices=["AF", "LF"])
    ap.add_argument("--data-dir", default="elliptic_bitcoin_dataset")
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    train(a.model, a.featset, a.data_dir, epochs=a.epochs)
