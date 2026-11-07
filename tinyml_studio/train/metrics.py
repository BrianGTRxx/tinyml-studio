"""Métricas de clasificación y regresión (numpy puro, sin depender de sklearn)."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype="int64")
    np.add.at(cm, (y, pred), 1)
    return cm


def classification_metrics(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    n = len(classes)
    cm = confusion_matrix(y, pred, n)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    pred_n = cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_n > 0, tp / pred_n, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = support > 0  # las clases sin muestras no cuentan en los promedios
    return {
        "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
        "balanced_accuracy": float(rec[present].mean()) if present.any() else 0.0,
        "f1_macro": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            c: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
        "n": int(cm.sum()),
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = y - pred
    ss_res, ss_tot = float((err**2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "n": int(len(y)),
    }

