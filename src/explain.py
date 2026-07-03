"""GNNExplainer: evidence subgraph + feature attribution for a flagged transaction.

Usage:
    python src/explain.py --data-dir elliptic_bitcoin_dataset
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from data import load_elliptic, temporal_masks
from models import GraphSAGE


def main(data_dir, weights="models/GraphSAGE_LF.pt", node=None, out_dir="results"):
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.utils import k_hop_subgraph

    d = load_elliptic(data_dir)
    x, ei, y, ts = d["x"][:, :94], d["edge_index"], d["y"], d["timestep"]
    _, _, te = temporal_masks(y, ts)
    model = GraphSAGE(94)
    model.load_state_dict(torch.load(weights)); model.eval()
    with torch.no_grad():
        prob = F.softmax(model(x, ei), dim=1)[:, 1]
    if node is None:
        node = int(((y == 1) & te & (prob > 0.9)).nonzero().flatten()[0])
    print(f"explaining node {node} (t={int(ts[node])}, p_illicit={float(prob[node]):.3f})")

    subset, sub_ei, mapping, _ = k_hop_subgraph(node, 2, ei, relabel_nodes=True)
    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=100),
        explanation_type="model", node_mask_type="attributes", edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"))
    expl = explainer(x[subset], sub_ei, index=int(mapping))
    feat_imp = expl.node_mask.detach().numpy().sum(0)
    out = {"node_index": node, "timestep": int(ts[node]),
           "pred_prob_illicit": float(prob[node]),
           "subgraph_nodes": int(len(subset)), "subgraph_edges": int(sub_ei.shape[1]),
           "top10_feature_indices": np.argsort(-feat_imp)[:10].tolist()}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(f"{out_dir}/explainer.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="elliptic_bitcoin_dataset")
    ap.add_argument("--weights", default="models/GraphSAGE_LF.pt")
    ap.add_argument("--node", type=int, default=None)
    a = ap.parse_args()
    main(a.data_dir, a.weights, a.node)
