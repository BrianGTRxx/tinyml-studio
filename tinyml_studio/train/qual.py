"""Quality blocks (curves, per-split metrics, overfit view) attached to every training result."""

from __future__ import annotations

import numpy as np

from .curves import (
    classification_quality,
    pr_curve_points,
    regression_quality,
    roc_curve_points,
    split_comparison,
)

MAX_ROWS = 20000  # cap for the train split (speed); deterministic subsample


def _sub(n: int, seed: int = 0) -> np.ndarray:
    if n <= MAX_ROWS:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, MAX_ROWS, replace=False))


def sk_scores(est, X: np.ndarray) -> np.ndarray | None:
    """Class scores of a scikit-learn classifier (probabilities when available)."""
    try:
        if hasattr(est, "predict_proba"):
            return np.asarray(est.predict_proba(X))
        if hasattr(est, "decision_function"):
            s = np.asarray(est.decision_function(X))
            return np.stack([-s, s], 1) if s.ndim == 1 else s
    except Exception:  # noqa: BLE001 - quality is optional, never block training
        return None
    return None


def classification_block(
    y: dict[str, np.ndarray], proba: dict[str, np.ndarray | None], classes: list[str]
) -> dict:
    splits, curves = {}, {}
    for s, yy in y.items():
        p = proba.get(s)
        idx = _sub(len(yy))
        yy, p = yy[idx], (p[idx] if p is not None else None)
        pred = p.argmax(1) if p is not None else None
        if pred is None:
            continue
        splits[s] = classification_quality(yy, pred, p, classes)
        if s == "test":
            curves = {
                "roc": roc_curve_points(yy, p, classes=classes),
                "pr": pr_curve_points(yy, p, classes=classes),
            }
    return {
        "kind": "classification",
        "splits": splits,
        "curves": curves,
        "comparison": split_comparison(
            {
                s: {k: v for k, v in q.items() if isinstance(v, (int, float))}
                for s, q in splits.items()
            }
        ),
    }


def regression_block(y: dict[str, np.ndarray], pred: dict[str, np.ndarray]) -> dict:
    splits = {s: regression_quality(y[s], pred[s]) for s in y if s in pred}
    return {
        "kind": "regression",
        "splits": splits,
        "comparison": split_comparison(
            {
                s: {k: v for k, v in q.items() if isinstance(v, (int, float))}
                for s, q in splits.items()
            }
        ),
    }

