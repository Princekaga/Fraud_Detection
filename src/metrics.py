"""Imbalance-aware evaluation metrics."""
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
import numpy as np


def evaluate(y_true, prob, thresh=0.5):
    pred = (prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"pr_auc": float(average_precision_score(y_true, prob)),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "illicit_f1": float(f1_score(y_true, pred, pos_label=1)),
            "illicit_precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
            "illicit_recall": float(recall_score(y_true, pred, pos_label=1)),
            "fpr": float(fp / (fp + tn)),
            "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def tune_threshold(y_val, prob_val):
    """Max-F1 threshold on validation data (never on test)."""
    p, r, th = precision_recall_curve(y_val, prob_val)
    f1s = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(th[np.argmax(f1s[:-1])])
