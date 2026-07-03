"""Load and preprocess the Elliptic dataset into graph tensors."""
import os
import numpy as np
import pandas as pd
import torch


def load_elliptic(data_dir="elliptic_bitcoin_dataset"):
    """Returns dict with x (N,166 float32), edge_index (2,2E undirected int64),
    y (N int64: 1 illicit / 0 licit / -1 unknown), timestep (N int16), tx_ids."""
    feats = pd.read_csv(f"{data_dir}/elliptic_txs_features.csv", header=None,
                        dtype={i: np.float32 for i in range(1, 167)}, converters={0: int})
    tx_ids = feats[0].values.astype(np.int64)
    X = feats.iloc[:, 1:].values.astype(np.float32)
    del feats
    timestep = X[:, 0].astype(np.int16)

    classes = pd.read_csv(f"{data_dir}/elliptic_txs_classes.csv")
    lab = dict(zip(classes["txId"].values,
                   classes["class"].map({"1": 1, "2": 0, "unknown": -1}).astype(np.int8).values))
    y = np.array([lab[t] for t in tx_ids], dtype=np.int64)

    idx = {t: i for i, t in enumerate(tx_ids)}
    edges = pd.read_csv(f"{data_dir}/elliptic_txs_edgelist.csv")
    src = edges["txId1"].map(idx).values.astype(np.int64)
    dst = edges["txId2"].map(idx).values.astype(np.int64)
    ei = torch.from_numpy(np.stack([src, dst]))
    ei = torch.cat([ei, ei.flip(0)], dim=1)          # undirected message passing

    return {"x": torch.from_numpy(X), "edge_index": ei, "y": torch.from_numpy(y),
            "timestep": timestep, "tx_ids": tx_ids}


def temporal_masks(y, timestep):
    """Deployment-faithful split: train t1-29, val t30-34, test t35-49 (labelled only)."""
    lab = (y >= 0).numpy() if isinstance(y, torch.Tensor) else (y >= 0)
    tr = torch.from_numpy(lab & (timestep <= 29))
    va = torch.from_numpy(lab & (timestep >= 30) & (timestep <= 34))
    te = torch.from_numpy(lab & (timestep >= 35))
    return tr, va, te
