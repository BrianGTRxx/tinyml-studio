"""Curves and rich quality reports (ROC, PR, per-class / per-label metrics, regression).

Pure numpy and JSON-serializable: every function returns plain ``dict``/``list``/``float``/``None``
(NaN and inf are converted to ``None``). Curves are computed on the full data and only then
downsampled deterministically (endpoints are always kept).
"""

from __future__ import annotations

import numpy as np

from tinyml_studio.i18n import tr

# ----------------------------------------------------------------------------- helpers


def _clean(obj):
    """Recursively convert numpy types to Python and non-finite floats to ``None``."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


def _downsample_idx(n: int, max_points: int) -> np.ndarray:
    """Deterministic evenly spaced indices in ``[0, n)`` that include 0 and n-1."""
    if max_points < 2 or n <= max_points:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))


def _as_labels(y_true, n_classes: int | None = None) -> np.ndarray:
    """Accept integer labels (n,) or one-hot / indicator (n, k); return integer labels."""
    y = np.asarray(y_true)
    if y.ndim == 2:
        return y.argmax(1).astype("int64")
    return y.astype("int64")


def _as_proba(proba, n_classes: int | None = None) -> np.ndarray:
    p = np.asarray(proba, dtype="float64")
    if p.ndim == 1:  # binary: probability of the positive class
        p = np.stack([1.0 - p, p], axis=1)
    return p


def _class_names(classes, k: int) -> list[str]:
    names = [str(c) for c in classes] if classes is not None else []
    return names if len(names) == k else [str(i) for i in range(k)]


def _binary_curves(y: np.ndarray, score: np.ndarray):
    """Full ROC and PR curves for one binary problem (ties handled like sklearn).

    Returns ``None`` when there are no positives or no negatives.
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype="float64")
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0 or len(y) == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, len(yy) - 1]
    tps = np.cumsum(yy)[idx].astype(float)
    fps = (1 + idx) - tps
    fpr = np.r_[0.0, fps / nneg]
    tpr = np.r_[0.0, tps / npos]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    precision = tps / (tps + fps)
    recall = tps / npos
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    # PR curve starts at recall 0 with the precision of the first point (as plotted).
    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "recall": np.r_[0.0, recall],
        "precision": np.r_[precision[0], precision],
        "ap": ap,
    }


def _curves_for_classes(y_true, proba, classes, max_points):
    p = _as_proba(proba)
    names = _class_names(classes, p.shape[1])
    out = {}
    y = _as_labels(y_true)
    for i, name in enumerate(names):
        out[name] = _binary_curves(y == i, p[:, i])
    return out, y, p, names


# ----------------------------------------------------------------------------- curves


def roc_curve_points(
    y_true, proba, max_points: int = 100, classes=None
) -> dict:
    """One-vs-rest ROC curve per class.

    ``y_true``: integer labels (n,) or one-hot (n, k). ``proba``: (n, k) or (n,) for binary.
    Returns ``{class: {"fpr": [...], "tpr": [...], "auc": float | None}}``. A class without
    positives or negatives in this split gives empty lists and ``auc=None``.
    """
    curves, _, _, _ = _curves_for_classes(y_true, proba, classes, max_points)
    out = {}
    for name, c in curves.items():
        if c is None:
            out[name] = {"fpr": [], "tpr": [], "auc": None}
            continue
        ix = _downsample_idx(len(c["fpr"]), max_points)
        out[name] = {"fpr": c["fpr"][ix], "tpr": c["tpr"][ix], "auc": c["auc"]}
    return _clean(out)


def pr_curve_points(
    y_true, proba, max_points: int = 100, classes=None
) -> dict:
    """One-vs-rest precision-recall curve per class.

    Returns ``{class: {"recall": [...], "precision": [...], "ap": float | None}}``.
    """
    curves, _, _, _ = _curves_for_classes(y_true, proba, classes, max_points)
    out = {}
    for name, c in curves.items():
        if c is None:
            out[name] = {"recall": [], "precision": [], "ap": None}
            continue
        ix = _downsample_idx(len(c["recall"]), max_points)
        out[name] = {"recall": c["recall"][ix], "precision": c["precision"][ix], "ap": c["ap"]}
    return _clean(out)


# ----------------------------------------------------------------------------- classification


def _prf(tp, fp, fn):
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1


def _mean(vals) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def matthews_corrcoef(cm) -> float | None:
    """Multiclass Matthews correlation coefficient (Gorodkin 2004) from a confusion matrix.

    ``cm[i, j]`` = rows of true class ``i`` predicted as ``j``. Returns 0.0 when it is undefined
    (a constant predictor or a single true class), as scikit-learn does; ``None`` when empty.
    """
    cm = np.asarray(cm, dtype="float64")
    s = cm.sum()
    if s == 0:
        return None
    c = np.trace(cm)
    t, p = cm.sum(1), cm.sum(0)
    den = float(np.sqrt((s * s - p @ p) * (s * s - t @ t)))
    return float((c * s - t @ p) / den) if den > 0 else 0.0


def bootstrap_f1_macro(y_true, y_pred, n_classes: int, n_boot: int = 200, seed: int = 0):
    """Bootstrap replicates (``n_boot``,) of the macro F1 (classes absent from a resample are
    excluded, as in :func:`classification_quality`). Deterministic for a given ``seed`` and ``n``:
    models evaluated on the same rows get the SAME resamples (paired comparison)."""
    y = np.asarray(y_true).astype("int64")
    pr = np.asarray(y_pred).astype("int64")
    n, k = len(y), int(n_classes)
    if n == 0:
        return np.full(n_boot, np.nan)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    out = np.empty(n_boot)
    for b in range(n_boot):
        cm = np.bincount(y[idx[b]] * k + pr[idx[b]], minlength=k * k).reshape(k, k)
        tp = np.diag(cm).astype(float)
        sup = cm.sum(1)
        _, _, f1 = _prf(tp, cm.sum(0) - tp, sup - tp)
        out[b] = f1[sup > 0].mean()
    return out


def bootstrap_f1_macro_ci(
    y_true, y_pred, n_classes: int, n_boot: int = 200, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Percentile bootstrap CI of macro F1: ``{"point", "lo", "hi", "alpha", "n_boot"}``."""
    y = np.asarray(y_true).astype("int64")
    pr = np.asarray(y_pred).astype("int64")
    k = int(n_classes)
    cm = np.zeros((k, k), dtype="int64")
    np.add.at(cm, (y, pr), 1)
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1)
    _, _, f1 = _prf(tp, cm.sum(0) - tp, sup - tp)
    point = float(f1[sup > 0].mean()) if (sup > 0).any() else None
    rep = bootstrap_f1_macro(y, pr, k, n_boot, seed)
    lo, hi = (
        np.nanpercentile(rep, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(y) else (None,) * 2
    )
    return _clean({"point": point, "lo": lo, "hi": hi, "alpha": alpha, "n_boot": int(n_boot)})


def classification_quality(y_true, y_pred, proba, classes, max_points: int = 100) -> dict:
    """Full multiclass quality report.

    Returns a dict with: ``n``, ``accuracy``, ``balanced_accuracy``, ``mcc`` (multiclass Matthews),
    ``precision_macro/weighted``,
    ``recall_macro/weighted``, ``f1_macro/weighted``, ``per_class`` (``{class: {precision, recall,
    f1, support, auc, ap}}``), ``roc_auc_macro``, ``mAP`` (mean of per-class AP over classes with
    support), ``confusion_matrix`` (raw), ``confusion_matrix_norm`` (row-normalized), ``classes``
    and ``notes`` (list of user-facing strings). ``proba`` may be ``None`` (AUC/AP then ``None``).
    Classes absent from this split are excluded from macro averages and have AUC/AP ``None``.
    """
    y = _as_labels(y_true)
    pred = np.asarray(y_pred).astype("int64")
    p = _as_proba(proba) if proba is not None else None
    k = p.shape[1] if p is not None else len(classes)
    names = _class_names(classes, k)
    cm = np.zeros((k, k), dtype="int64")
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    fp = cm.sum(0) - tp
    fn = sup - tp
    prec, rec, f1 = _prf(tp, fp, fn)
    present = sup > 0
    n = float(cm.sum())
    w = sup / n if n else sup

    def macro(a):
        return float(a[present].mean()) if present.any() else None

    def weighted(a):
        return float((a * w).sum()) if n else None

    aucs: dict[str, float | None] = {}
    aps: dict[str, float | None] = {}
    for i, name in enumerate(names):
        c = _binary_curves(y == i, p[:, i]) if p is not None else None
        aucs[name] = c["auc"] if c else None
        aps[name] = c["ap"] if c else None
    with np.errstate(divide="ignore", invalid="ignore"):
        cmn = np.where(sup[:, None] > 0, cm / np.where(sup[:, None] > 0, sup[:, None], 1), 0.0)
    out = {
        "n": int(n),
        "classes": names,
        "accuracy": float(tp.sum() / n) if n else None,
        "balanced_accuracy": macro(rec),
        "mcc": matthews_corrcoef(cm),
        "precision_macro": macro(prec),
        "recall_macro": macro(rec),
        "f1_macro": macro(f1),
        "precision_weighted": weighted(prec),
        "recall_weighted": weighted(rec),
        "f1_weighted": weighted(f1),
        "per_class": {
            nm: {
                "precision": prec[i],
                "recall": rec[i],
                "f1": f1[i],
                "support": int(sup[i]),
                "auc": aucs[nm],
                "ap": aps[nm],
            }
            for i, nm in enumerate(names)
        },
        "roc_auc_macro": _mean(aucs.values()),
        "mAP": _mean(aps.values()),
        "confusion_matrix": cm,
        "confusion_matrix_norm": cmn,
        "notes": _classification_notes(names, present, p is None),
    }
    return _clean(out)


def _classification_notes(names, present, no_proba) -> list[str]:
    notes = [
        tr(
            "mAP50 (mAP@IoU0.5) es una métrica de detección de objetos: Necesita cajas "
            "delimitadoras (bounding boxes) y no aplica a este modelo. Aquí mAP es la media "
            "de la precisión promedio (AP) de todas las clases.",
            "mAP50 (mAP@IoU0.5) is an object-detection metric: It needs bounding boxes and "
            "does not apply to this model. Here mAP is the mean average precision (AP) "
            "across classes.",
        )
    ]
    absent = [n for n, ok in zip(names, present, strict=False) if not ok]
    if absent:
        notes.append(
            tr(
                "Clases sin muestras en esta partición (AUC/AP no definidos): {c}.",
                "Classes with no samples in this split (AUC/AP undefined): {c}.",
                c=", ".join(absent),
            )
        )
    if no_proba:
        notes.append(
            tr(
                "Sin probabilidades: No se calculan ROC, AUC ni AP.",
                "No probabilities given: ROC, AUC and AP are not computed.",
            )
        )
    return notes


# ----------------------------------------------------------------------------- regression


def regression_quality(y_true, y_pred, max_points: int = 300, bins: int = 20) -> dict:
    """Regression report: ``n``, ``mae``, ``rmse``, ``r2``, ``residual_hist`` (``{bins, counts}``,
    ``bins`` are the ``len(counts)+1`` edges of residual = y_true - y_pred) and ``scatter``
    (``{y_true, y_pred}`` downsampled with a fixed seed). Non-finite pairs are dropped.
    """
    y = np.asarray(y_true, dtype="float64").ravel()
    p = np.asarray(y_pred, dtype="float64").ravel()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return _clean(
            {
                "n": 0,
                "mae": None,
                "rmse": None,
                "r2": None,
                "residual_hist": {"bins": [], "counts": []},
                "scatter": {"y_true": [], "y_pred": []},
            }
        )
    err = y - p
    ss_res, ss_tot = float((err**2).sum()), float(((y - y.mean()) ** 2).sum())
    counts, edges = np.histogram(err, bins=bins)
    if len(y) > max_points:
        sel = np.sort(np.random.default_rng(0).choice(len(y), size=max_points, replace=False))
    else:
        sel = np.arange(len(y))
    return _clean(
        {
            "n": int(len(y)),
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            "residual_hist": {"bins": edges, "counts": counts},
            "scatter": {"y_true": y[sel], "y_pred": p[sel]},
        }
    )


# ----------------------------------------------------------------------------- splits


def split_comparison(metrics_by_split: dict, gap_threshold: float = 0.1) -> dict:
    """Compare scalar metrics across splits (e.g. ``{"train": {...}, "val": {...}}``).

    Returns ``{"splits": [...], "metrics": {metric: {split: value}}, "gaps": {metric: train -
    other}, "notes": [...]}``. ``gaps`` only exists for metrics where higher is better (accuracy,
    F1, AUC, mAP, R²...); a note warns of possible overfitting when the train-vs-other gap
    exceeds ``gap_threshold``.
    """
    splits = list(metrics_by_split)
    keys: list[str] = []
    for s in splits:
        for k, v in (metrics_by_split[s] or {}).items():
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) and k != "n":
                if k not in keys:
                    keys.append(k)
    metrics = {k: {s: (metrics_by_split[s] or {}).get(k) for s in splits} for k in keys}
    higher_better = ("accuracy", "f1", "precision", "recall", "auc", "map", "r2")
    gaps: dict[str, dict[str, float | None]] = {}
    notes: list[str] = []
    if "train" in splits:
        for k, vals in metrics.items():
            if not any(h in k.lower() for h in higher_better):
                continue
            tv = vals.get("train")
            gaps[k] = {}
            for s in splits:
                if s == "train":
                    continue
                v = vals.get(s)
                ok = tv is not None and v is not None and np.isfinite(tv) and np.isfinite(v)
                gaps[k][s] = float(tv - v) if ok else None
                if ok and tv - v > gap_threshold:
                    notes.append(
                        tr(
                            "Posible sobreajuste: {m} baja {d:.2f} de train a {s}.",
                            "Possible overfitting: {m} drops {d:.2f} from train to {s}.",
                            m=k,
                            d=float(tv - v),
                            s=s,
                        )
                    )
    return _clean({"splits": splits, "metrics": metrics, "gaps": gaps, "notes": notes})
