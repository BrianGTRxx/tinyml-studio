"""Benchmark: entrena muchos modelos sobre LOS MISMOS splits y ayuda a decidir cuál es mejor.

Reglas de honestidad:

* La selección usa SOLO validación (o validación cruzada sobre TRAIN si ``cv_folds >= 2``); el test
  nunca elige: solo se usa, por transparencia, para comparar estadísticamente los modelos.
* Regla de una desviación estándar (1-SE; Breiman et al. 1984, Hastie et al. 2009 §7.10): entre
  los modelos cuyo puntaje de selección está a menos de 1 error estándar del mejor se prefiere el
  exportable, más pequeño y más rápido.
* Validación cruzada por grupos / bloques temporales cuando la receta parte así (Roberts et al.
  2017, "Cross-validation strategies for data with temporal, spatial, hierarchical, or
  phylogenetic structure"). Con repetidos ajustes sobre la misma partición el test deja de ser
  insesgado (Cawley & Talbot 2010).
* Todo es reproducible con ``seed``; los remuestreos bootstrap usan un generador con semilla.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import re
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio import progress as _progress
from tinyml_studio.clean import prepare_splits
from tinyml_studio.i18n import tr

from . import models as M
from .curves import (
    _clean,
    _prf,
    bootstrap_f1_macro,
    bootstrap_f1_macro_ci,
    classification_quality,
    matthews_corrcoef,
    regression_quality,
)
from .features import FeatureEncoder
from .run import feature_columns, train_models

CLS_METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "mcc", "mAP", "roc_auc_macro")
REG_METRICS = ("r2", "mae", "rmse")
LOWER_BETTER = frozenset({"mae", "rmse"})
CV_METRICS = frozenset({"f1_macro", "accuracy", "balanced_accuracy", "mcc", "r2", "mae", "rmse"})
KERAS_KEYS = {
    "name", "hidden", "epochs", "batch_size", "learning_rate", "patience", "activation",
    "dropout", "l2", "optimizer", "weight_decay", "nesterov", "lr_schedule", "lr_final_frac",
}  # fmt: skip
SUBDIR_RE = re.compile(r"^[A-Za-z0-9_:\-]+$")
ALPHA = 0.05
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- pure statistics


def exact_mcnemar(b: int, c: int) -> float:
    """McNemar exacto (binomial, dos colas) con ``b`` y ``c`` los pares discordantes.

    ``p = min(1, 2 * P(X <= min(b, c)))``, ``X ~ Binomial(b + c, 1/2)``; ``1.0`` sin discordantes.
    """
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(1.0, 2.0 * tail / 2**n))


def holm_adjust(pvals: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni (control del error de la familia de comparaciones)."""
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def _cm(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)


def _cls_from_cm(metric: str, cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    n = cm.sum()
    if metric == "accuracy":
        return float(tp.sum() / n) if n else float("nan")
    _, rec, f1 = _prf(tp, cm.sum(0) - tp, sup - tp)
    present = sup > 0
    if metric == "balanced_accuracy":
        return float(rec[present].mean()) if present.any() else float("nan")
    if metric == "f1_macro":
        return float(f1[present].mean()) if present.any() else float("nan")
    if metric == "mcc":
        v = matthews_corrcoef(cm)
        return float("nan") if v is None else v
    raise ValueError(metric)


def _reg_score(metric: str, y: np.ndarray, p: np.ndarray) -> float:
    err = y - p
    if metric == "mae":
        return float(np.abs(err).mean())
    if metric == "rmse":
        return float(np.sqrt((err**2).mean()))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def score_fn(task: str, metric: str, y: np.ndarray, pred: np.ndarray, k: int = 0) -> float:
    """Métrica escalar barata (solo numpy) usada en CV y bootstrap."""
    if task == "classification":
        return _cls_from_cm(metric, _cm(np.asarray(y, int), np.asarray(pred, int), k))
    return _reg_score(metric, np.asarray(y, float), np.asarray(pred, float))


def bootstrap_scores(
    task: str, metric: str, y, pred, k: int = 0, n_boot: int = 200, seed: int = 0
) -> np.ndarray:
    """Réplicas bootstrap de ``metric``. Mismo ``seed`` y ``n`` => mismos remuestreos (pareado)."""
    if task == "classification" and metric == "f1_macro":
        return bootstrap_f1_macro(y, pred, k, n_boot, seed)
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    if n == 0:
        return np.full(n_boot, np.nan)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    return np.array([score_fn(task, metric, y[i], pred[i], k) for i in idx])


def _ci(rep: np.ndarray, point: float | None) -> dict:
    ok = rep[np.isfinite(rep)]
    lo, hi = np.percentile(ok, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]) if len(ok) else (None,) * 2
    return _clean({"point": point, "lo": lo, "hi": hi, "alpha": ALPHA, "n_boot": int(len(rep))})


# ----------------------------------------------------------------------------- CV on TRAIN


def make_cv_folds(
    split: dict[str, Any], df, y: np.ndarray, k: int, seed: int, task: str
) -> dict[str, Any]:
    """Pliegues (índices sobre ``df`` = train limpio) acordes con la partición de la receta.

    * ``group``: GroupKFold; un grupo nunca está en train y validación a la vez.
    * ``temporal_blocks``: cada pliegue es un tramo temporal contiguo de cada grupo (CV por
      bloques, Roberts 2017). Sin hueco de guarda entre tramos: hay fuga leve en los bordes.
    * ``stratified``/``random``: StratifiedKFold (clasificación) o KFold, barajados con ``seed``.
    """
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

    n = len(y)
    strat = (split or {}).get("strategy", "random")
    notes: list[str] = []
    k = max(2, min(int(k), n))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    scheme = strat
    gcol, tcol = (split or {}).get("group_column"), (split or {}).get("time_column")

    if strat == "group" and gcol in df.columns and df[gcol].nunique() >= 2:
        groups = df[gcol].to_numpy()
        kk = min(k, len(np.unique(groups)))
        if kk < k:
            notes.append(
                tr(
                    "solo hay {g} grupos en train: Se usan {g} pliegues",
                    "only {g} groups in train: Using {g} folds",
                    g=kk,
                )
            )
        folds = list(GroupKFold(n_splits=kk).split(np.zeros(n), y, groups))
        scheme = "group"
    elif strat == "temporal_blocks":
        fid = np.zeros(n, dtype=int)
        pos = np.arange(n)
        parts = (
            [pos]
            if gcol is None or gcol not in df.columns
            else [pos[i] for i in df.groupby(gcol, sort=True).indices.values()]
        )
        for g in parts:
            if tcol and tcol in df.columns:
                g = g[np.argsort(df[tcol].to_numpy()[g], kind="stable")]
            fid[g] = np.minimum(k - 1, (np.arange(len(g)) * k) // max(len(g), 1))
        for f in range(k):
            va = np.flatnonzero(fid == f)
            tr_i = np.flatnonzero(fid != f)
            if len(va) and len(tr_i):
                folds.append((tr_i, va))
        scheme = "temporal_blocks"
    else:
        if strat in ("group", "temporal_blocks"):
            notes.append(
                tr(
                    "la columna de grupo no está disponible en train: CV estratificada normal "
                    "(puede ser optimista)",
                    "the group column is not available in train: Plain stratified CV "
                    "(may be optimistic)",
                )
            )
        if task == "classification" and np.bincount(y).min() >= k:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            folds = list(skf.split(np.zeros(n), y))
            scheme = "stratified"
        else:
            if task == "classification":
                notes.append(
                    tr(
                        "alguna clase tiene menos de {k} filas en train: CV sin estratificar",
                        "some class has fewer than {k} rows in train: Unstratified CV",
                        k=k,
                    )
                )
            folds = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n)))
            scheme = "random"
    return {"folds": folds, "scheme": scheme, "k": len(folds), "notes": notes}


def _cv_scores(est_factory, y, folds, task, metric, k_classes, enc_fit, df_train, cap):
    """Puntaje por pliegue; el codificador se reajusta SOLO con el train del pliegue."""
    scores = []
    for tr_i, va_i in folds:
        enc_f = enc_fit(df_train.iloc[tr_i])
        Xtr, Xva = enc_f.transform(df_train.iloc[tr_i]), enc_f.transform(df_train.iloc[va_i])
        ytr, yva = y[tr_i], y[va_i]
        if cap and len(Xtr) > cap:
            sel = np.random.default_rng(0).choice(len(Xtr), cap, replace=False)
            Xtr, ytr = Xtr[sel], ytr[sel]
        est = est_factory()
        est.fit(Xtr, ytr)
        scores.append(score_fn(task, metric, yva, est.predict(Xva), k_classes))
    return scores


# ----------------------------------------------------------------------------- selection


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Nombres no dominados en (puntaje ↑, tamaño ↓). ``points[name] = (score, size)``."""
    front = []
    for a, (sa, za) in points.items():
        dominated = any(
            sb >= sa and zb <= za and (sb > sa or zb < za)
            for b, (sb, zb) in points.items()
            if b != a
        )
        if not dominated:
            front.append(a)
    return front


def select_model(rows: list[dict], *, higher_better: bool = True) -> dict[str, Any]:
    """Regla de 1 error estándar sobre ``rows`` (dicts con name, score, se, fits_flash,
    exportable, size_bytes, predict_us). Los modelos con ``fits_flash is False`` se marcan pero no
    compiten. Devuelve ``{best, selected, within_1se, none_fit}`` (nombres)."""
    sign = 1.0 if higher_better else -1.0
    scored = [r for r in rows if r.get("score") is not None and np.isfinite(r["score"])]
    if not scored:
        return {"best": None, "selected": None, "within_1se": [], "none_fit": False}
    elig = [r for r in scored if r.get("fits_flash") is not False]
    none_fit = not elig
    pool = elig or scored
    best = max(pool, key=lambda r: sign * r["score"])
    se = best.get("se") or 0.0
    thr = sign * best["score"] - se
    within = [r for r in pool if sign * r["score"] >= thr - 1e-12]

    def pref(r):
        size = r.get("size_bytes")
        return (
            not r.get("exportable"),
            size is None,
            size if size is not None else 0,
            r.get("predict_us") if r.get("predict_us") is not None else float("inf"),
            -sign * r["score"],
            r["name"],
        )

    chosen = min(within, key=pref)
    overall = max(scored, key=lambda r: sign * r["score"])
    return {
        "best": overall["name"],
        "best_eligible": best["name"],
        "selected": chosen["name"],
        "within_1se": [r["name"] for r in sorted(within, key=pref)],
        "none_fit": none_fit,
    }


# ----------------------------------------------------------------------------- helpers


def normalize_keras_presets(keras: list[dict] | None) -> list[dict]:
    """Valida los presets y les pone un nombre único (``name`` o ``h32-16``)."""
    out: list[dict] = []
    seen: set[str] = set()
    for i, cfg in enumerate(keras or []):
        bad = set(cfg) - KERAS_KEYS
        if bad:
            raise ValueError(
                tr(
                    "preset Keras con claves desconocidas: {b}",
                    "Keras preset with unknown keys: {b}",
                    b=sorted(bad),
                )
            )
        c = dict(cfg)
        hidden = tuple(int(h) for h in c.get("hidden", (32, 16)))
        label = str(c.pop("name", "") or "h" + "-".join(str(h) for h in hidden))
        label = re.sub(r"[^A-Za-z0-9_\-]", "-", label)[:32] or f"p{i}"
        base, j = label, 2
        while label in seen:
            label, j = f"{base}-{j}", j + 1
        seen.add(label)
        c["hidden"] = hidden
        out.append(
            {"label": label, "name": f"keras_mlp:{label}", "dir": f"keras_mlp_{label}", "cfg": c}
        )
    return out


def _tf_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


@contextlib.contextmanager
def _scaled_progress(lo: float, hi: float) -> Iterator[None]:
    """Reescala los ``emit`` de los entrenadores al tramo [lo, hi] (la barra no retrocede)."""
    orig = _progress.emit

    def scaled(stage, pct=None, msg=None, **kw):
        orig(stage, None if pct is None else lo + (hi - lo) * float(pct) / 100.0, msg, **kw)

    patched = [
        m
        for name, m in list(sys.modules.items())
        if name.startswith("tinyml_studio") and getattr(m, "emit", None) is orig
    ]
    for m in patched:
        m.emit = scaled
    _progress.emit = scaled
    try:
        yield
    finally:
        for m in patched:
            m.emit = orig
        _progress.emit = orig


def _round(v, nd=4):
    return None if v is None else round(float(v), nd)


def _split_metrics(task, quality, split, y, pred, classes) -> dict[str, float | None]:
    keys = CLS_METRICS if task == "classification" else REG_METRICS
    blk = ((quality or {}).get("splits") or {}).get(split)
    if not (blk and all(k in blk for k in keys)) and pred is not None:
        blk = (
            classification_quality(y, pred, None, classes)
            if task == "classification"
            else regression_quality(y, pred)
        )
    blk = blk or {}
    return {k: _round(blk.get(k)) for k in keys}


def _row_of_classical(name, res, meta_all) -> dict:
    m = meta_all.get(name, {})
    return {
        "name": name,
        "family": m.get("family", "classical"),
        "kind": "classical",
        "params": res.get("params"),
        "fit_seconds": res.get("fit_seconds"),
        "predict_us_per_row": res.get("predict_us_per_row_cpu_cloud"),
        "complexity": res.get("complexity"),
        "notes": list(res.get("notes", [])),
    }


def _meta() -> dict[str, dict]:
    try:
        return M.meta()
    except Exception:  # noqa: BLE001 - la ficha es opcional
        return {}


def _classic_flash(name, est, prep, enc, task, n_classes) -> int | None:
    """Bytes de tablas del modelo en C (mismo emisor que la exportación); None si no se puede."""
    try:
        from tinyml_studio.export.cgen import EMITTERS
        from tinyml_studio.export.spec import build_spec

        if name not in EMITTERS:
            return None
        spec = build_spec(prep.pipeline, enc)
        _, data_bytes, _, _ = EMITTERS[name](est, name, spec, task, n_classes)
        return int(data_bytes)
    except Exception:  # noqa: BLE001
        return None


def _keras_flash(hidden, n_in, n_out) -> int:
    """Pesos int8 + sesgos int32 de un MLP denso (aprox. del .tflite int8, sin ops ni arena)."""
    sizes = [n_in, *hidden, n_out]
    w = sum(a * b for a, b in zip(sizes[:-1], sizes[1:], strict=True))
    b = sum(sizes[1:])
    return int(w + 4 * b)


# ----------------------------------------------------------------------------- main


def run_benchmark(
    recipe: dict[str, Any],
    *,
    models: list[str] | None = None,
    keras: list[dict] | None = None,
    cv_folds: int = 0,
    seed: int = 0,
    selection: str = "f1_macro",
    max_flash_kb: float | None = None,
    out_dir: str | Path | None = None,
    n_boot: int = 200,
    exclude: tuple[str, ...] = (),
    params: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Entrena, compara y recomienda. Ver el docstring del módulo y ``docs`` para las reglas.

    ``models=None`` => todos los clásicos de la tarea; ``models=[]`` => ninguno (solo Keras).
    ``keras`` = lista de diccionarios de hiperparámetros (hidden, epochs, batch_size,
    learning_rate, patience, activation, dropout, l2, name). Escribe ``out_dir/classical`` y
    ``out_dir/keras_mlp_<preset>``; sin ``out_dir`` usa una carpeta temporal que se borra.
    """
    t_start = time.perf_counter()
    task = recipe["task"]
    if task not in ("classification", "regression"):
        raise ValueError(
            tr(
                "tarea no entrenable con modelos supervisados: {task}",
                "task cannot be trained with supervised models: {task}",
                task=task,
            )
        )
    cls = task == "classification"
    valid = CLS_METRICS if cls else REG_METRICS
    if not cls and selection == "f1_macro":
        selection = "r2"
    if selection not in valid:
        raise ValueError(
            tr(
                "métrica de selección '{s}' no válida para {t}: {v}",
                "selection metric '{s}' not valid for {t}: {v}",
                s=selection,
                t=task,
                v=list(valid),
            )
        )
    cv_folds = int(cv_folds or 0)
    if cv_folds == 1 or cv_folds < 0 or cv_folds > 10:
        raise ValueError(tr("cv_folds debe ser 0 o 2..10", "cv_folds must be 0 or 2..10"))
    if cv_folds and selection not in CV_METRICS:
        raise ValueError(
            tr(
                "la métrica '{s}' no se puede usar con CV (usa una de {v})",
                "metric '{s}' cannot be used with CV (use one of {v})",
                s=selection,
                v=sorted(CV_METRICS),
            )
        )
    presets = normalize_keras_presets(keras)
    skipped: list[dict] = []
    if presets and not _tf_available():
        for p in presets:
            skipped.append(
                {
                    "name": p["name"],
                    "reason": tr(
                        "TensorFlow no está instalado: Se omite este modelo",
                        "TensorFlow is not installed: This model is skipped",
                    ),
                }
            )
        presets = []
    if models is not None and len(models) == 0 and not presets:
        raise ValueError(tr("no hay ningún modelo que entrenar", "there is no model to train"))
    higher = selection not in LOWER_BETTER

    tmp = None
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="tinyml_bench_")
        root = Path(tmp.name)
    else:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
    try:
        return _run(
            recipe, task, cls, models, presets, skipped, cv_folds, seed, selection, higher,
            max_flash_kb, root, out_dir is not None, n_boot, exclude, params, t_start,
        )  # fmt: skip
    finally:
        if tmp is not None:
            tmp.cleanup()


def _run(
    recipe, task, cls, models, presets, skipped, cv_folds, seed, selection, higher,
    max_flash_kb, root, keep, n_boot, exclude, params, t_start,
):  # fmt: skip
    import joblib

    emit = _progress.emit
    run_dirs: dict[str, str] = {}
    results: dict[str, dict] = {}  # nombre -> bloque de resultados del entrenador
    training_warnings: list[str] = []
    meta_all = _meta()
    n_total = (len(models) if models else len(M.available(task))) + len(presets)
    train_hi = 70.0 if cv_folds else 85.0

    # -- 1) clásicos (ya entrenan sobre los splits compartidos y guardan en root/classical)
    classical_res = None
    if models is None or models:
        share = 100.0 * (len(models) if models else len(M.available(task))) / max(n_total, 1)
        with _scaled_progress(3, 3 + (train_hi - 3) * share / 100.0):
            classical_res = train_models(
                recipe,
                models,
                seed=seed,
                params=params,
                exclude=exclude,
                out_dir=root / "classical",
            )
        for n, r in classical_res["models"].items():
            results[n] = {"kind": "classical", "res": r, "dir": "classical"}
            run_dirs[n] = "classical"
        training_warnings += classical_res.get("warnings", [])

    # -- 2) Keras MLP
    keras_res: dict[str, dict] = {}
    if presets:
        from .keras_mlp import train_keras_mlp

        base = 3 + (train_hi - 3) * (1 - len(presets) / max(n_total, 1))
        step = (train_hi - base) / len(presets)
        for i, p in enumerate(presets):
            emit(
                "keras",
                base + step * i,
                tr(
                    "Entrenando {n} ({i}/{m})…",
                    "Training {n} ({i}/{m})…",
                    n=p["name"],
                    i=i + 1,
                    m=len(presets),
                ),
                model=p["name"],
            )
            try:
                with _scaled_progress(base + step * i, base + step * (i + 1)):
                    kr = train_keras_mlp(
                        recipe, **p["cfg"], seed=seed, exclude=exclude, out_dir=root / p["dir"]
                    )
            except Exception as exc:  # noqa: BLE001 - un preset roto no tumba el benchmark
                skipped.append({"name": p["name"], "reason": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            keras_res[p["name"]] = kr
            results[p["name"]] = {"kind": "keras", "res": kr, "dir": p["dir"], "preset": p}
            run_dirs[p["name"]] = p["dir"]
            training_warnings += [w for w in kr.get("warnings", []) if w not in training_warnings]
    if not results:
        raise ValueError(tr("ningún modelo se pudo entrenar", "no model could be trained"))

    # -- 3) datos compartidos (mismos splits y codificador que usaron los entrenadores)
    emit("evaluate", train_hi + 2, tr("Evaluando en val y test…", "Evaluating on val and test…"))
    first_dir = root / next(iter(results.values()))["dir"]
    enc = FeatureEncoder.from_dict(
        json.loads((first_dir / "encoder.json").read_text(encoding="utf-8"))
    )
    prep = prepare_splits(recipe, seed=seed)
    target = prep.target
    X = {s: enc.transform(df) for s, df in prep.clean.items()}
    if cls:
        y = {s: enc.encode_target(df[target]) for s, df in prep.clean.items()}
    else:
        y = {s: np.asarray(df[target], dtype="float64") for s, df in prep.clean.items()}
    classes = enc.classes
    k = len(classes)
    n_out = k if cls else 1

    rows: list[dict] = []
    preds: dict[str, dict[str, np.ndarray]] = {}
    estimators: dict[str, Any] = {}
    for name, info in results.items():
        r = info["res"]
        if info["kind"] == "classical":
            row = _row_of_classical(name, r, meta_all)
            est = joblib.load(root / info["dir"] / f"model_{name}.joblib")
            estimators[name] = est
            pr = {s: np.asarray(est.predict(X[s])) for s in ("val", "test")}
            q = r.get("quality")
            fl = _classic_flash(name, est, prep, enc, task, k)
            row["exportable"] = bool(meta_all.get(name, {}).get("exportable_c", fl is not None))
            row["export_job"] = "export_c"
            row["est_flash_bytes"] = fl
            row["size_basis"] = "c_tables" if fl is not None else None
        else:
            import tensorflow as tf

            hp = r["hyperparameters"]
            row = {
                "name": name,
                "family": "keras_mlp",
                "kind": "keras",
                "params": {kk: v for kk, v in hp.items()},
                "fit_seconds": r.get("fit_seconds"),
                "complexity": {"parameters": r.get("parameters")},
                "notes": [],
            }
            mdl = tf.keras.models.load_model(root / info["dir"] / "model.keras", compile=False)
            t0 = time.perf_counter()
            raw = {s: mdl.predict(X[s], verbose=0) for s in ("val", "test")}
            row["predict_us_per_row"] = round(
                (time.perf_counter() - t0) / (len(X["val"]) + len(X["test"])) * 1e6, 2
            )
            pr = {s: (v.argmax(1) if cls else v[:, 0]) for s, v in raw.items()}
            q = r.get("quality")
            row["exportable"] = True
            row["export_job"] = "export_tflite"
            row["est_flash_bytes"] = _keras_flash(hp["hidden"], X["train"].shape[1], n_out)
            row["size_basis"] = "int8_weights_estimate"
        preds[name] = pr
        row["run_subdir"] = info["dir"]
        row["metrics"] = {
            s: _split_metrics(task, q, s, y[s], pr.get(s), classes) for s in SPLITS
        }  # train sale del bloque de calidad (muestreado a 20000 filas), val/test de la predicción
        row["size_bytes"] = row["est_flash_bytes"]
        row["fits_flash"] = (
            None
            if max_flash_kb is None or row["est_flash_bytes"] is None
            else bool(row["est_flash_bytes"] <= max_flash_kb * 1024)
        )
        rows.append(row)

    # -- 4) validación cruzada sobre TRAIN (solo clásicos; Keras es demasiado caro)
    cv_meta: dict[str, Any] | None = None
    if cv_folds >= 2:
        feats = feature_columns(recipe, prep.clean["train"], exclude)
        df_tr = prep.clean["train"].reset_index(drop=True)
        fobj = make_cv_folds(recipe.get("split", {}), df_tr, y["train"], cv_folds, seed, task)
        cv_meta = {
            "metric": selection,
            "folds": fobj["k"],
            "scheme": fobj["scheme"],
            "notes": fobj["notes"]
            + [
                tr(
                    "la limpieza (imputación, escalado) se ajustó con todo train antes de partir "
                    "en pliegues: Fuga leve DENTRO de train; el codificador sí se reajusta por "
                    "pliegue. El test no interviene.",
                    "cleaning (imputation, scaling) was fitted on the whole train before folding: "
                    "slight leakage INSIDE train; the encoder is refitted per fold. Test is "
                    "not used.",
                )
            ],
        }
        classical_rows = [r for r in rows if r["kind"] == "classical"]
        for i, row in enumerate(classical_rows):
            name = row["name"]
            emit(
                "cv",
                train_hi + 2 + 20 * i / max(len(classical_rows), 1),
                tr("CV de {n} ({i}/{m})…", "CV of {n} ({i}/{m})…", n=name, i=i + 1,
                   m=len(classical_rows)),
                model=name,
            )  # fmt: skip
            row["cv"] = None
            try:
                eff = row["params"] or {}
                cap = eff.get("max_train_rows") if name in getattr(M, "ROW_CAPPED", ()) else None
                scores = _cv_scores(
                    lambda name=name: M.build(name, task, seed, (params or {}).get(name))[0],
                    y["train"], fobj["folds"], task, selection, k,
                    lambda d: FeatureEncoder.fit(d, feats, target, task, recipe.get("encoding")),
                    df_tr, cap,
                )  # fmt: skip
                sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                row["cv"] = {
                    "metric": selection,
                    "folds": len(scores),
                    "mean": _round(np.mean(scores)),
                    "std": _round(sd),
                    "se": _round(sd / math.sqrt(len(scores))),
                    "scores": [_round(s) for s in scores],
                }
            except Exception as exc:  # noqa: BLE001
                row["notes"].append(
                    tr("CV falló: {e}", "CV failed: {e}", e=f"{type(exc).__name__}: {exc}"[:200])
                )

    # -- 5) puntaje de selección (val o CV) y error estándar
    emit("select", 90, tr("Comparando y seleccionando…", "Comparing and selecting…"))
    use_cv = bool(cv_meta) and all(r.get("cv") for r in rows)
    warnings: list[str] = []
    if cv_meta and not use_cv:
        warnings.append(
            tr(
                "la CV solo cubre modelos clásicos (o falló en alguno): La selección usa "
                "validación para que todos compitan en igualdad; la CV se muestra como "
                "información (media±desv.)",
                "CV only covers classical models (or failed for some): Selection uses the "
                "validation split so everyone competes on equal terms; CV is shown for "
                "information (mean±std)",
            )
        )
    for row in rows:
        p = preds[row["name"]]
        v = row["metrics"]["val"].get(selection)
        if use_cv:
            row["selection_score"], row["selection_se"] = row["cv"]["mean"], row["cv"]["se"]
            row["selection_source"] = "cv"
        else:
            row["selection_score"] = v
            row["selection_source"] = "val"
            se = None
            if selection in (
                "f1_macro",
                "accuracy",
                "balanced_accuracy",
                "mcc",
                "r2",
                "mae",
                "rmse",
            ):
                rep = bootstrap_scores(task, selection, y["val"], p["val"], k, n_boot, seed)
                rep = rep[np.isfinite(rep)]
                se = float(np.std(rep, ddof=1)) if len(rep) > 1 else None
            row["selection_se"] = _round(se)
    if not use_cv and any(r["selection_se"] is None for r in rows):
        warnings.append(
            tr(
                "la métrica '{m}' no tiene error estándar aquí: No se aplica la regla de 1-SE",
                "metric '{m}' has no standard error here: The 1-SE rule is not applied",
                m=selection,
            )
        )

    sel = select_model(
        [
            {
                "name": r["name"],
                "score": r["selection_score"],
                "se": r["selection_se"],
                "fits_flash": r["fits_flash"],
                "exportable": r["exportable"],
                "size_bytes": r["size_bytes"],
                "predict_us": r.get("predict_us_per_row"),
            }
            for r in rows
        ],
        higher_better=higher,
    )
    by_name = {r["name"]: r for r in rows}
    chosen = sel["selected"]

    # -- 6) rankings por métrica (en VAL: el test no ordena nada)
    metric_keys = CLS_METRICS if cls else REG_METRICS
    rank_by_metric: dict[str, list[str]] = {}
    for mk in metric_keys:
        have = [r for r in rows if r["metrics"]["val"].get(mk) is not None]
        if len(have) != len(rows):
            continue
        sgn = -1.0 if mk in LOWER_BETTER else 1.0
        order = sorted(have, key=lambda r: (-sgn * r["metrics"]["val"][mk], r["name"]))
        rank_by_metric[mk] = [r["name"] for r in order]
        for r in have:
            vals = [x["metrics"]["val"][mk] for x in have]
            better = sum(1 for x in vals if sgn * x > sgn * r["metrics"]["val"][mk])
            r.setdefault("ranks", {})[mk] = better + 1

    # -- 7) Pareto (puntaje ↑, tamaño ↓)
    pts = {
        r["name"]: ((1 if higher else -1) * r["selection_score"], r["size_bytes"])
        for r in rows
        if r["selection_score"] is not None and r["size_bytes"] is not None
    }
    front = pareto_front(pts)
    for r in rows:
        r["pareto"] = r["name"] in front
        r["selected"] = r["name"] == chosen
        r["within_1se"] = r["name"] in sel["within_1se"]
    n_no_size = sum(1 for r in rows if r["size_bytes"] is None)
    if n_no_size:
        warnings.append(
            tr(
                "{n} modelo(s) sin estimación de tamaño (sin exportador a C): Quedan fuera del "
                "frente de Pareto",
                "{n} model(s) without a size estimate (no C exporter): Left out of the Pareto "
                "front",
                n=n_no_size,
            )
        )

    # -- 8) comparación estadística en TEST (transparencia, NO selección)
    test_cmp: dict[str, Any] = {"split": "test", "reference": chosen, "used_for_selection": False}
    ref = chosen
    yt = y["test"]
    n_test = len(yt)
    others = [r["name"] for r in rows if r["name"] != ref]
    if cls:
        test_cmp["method"] = "mcnemar_exact_holm"
        ref_ok = preds[ref]["test"] == yt
        raw_p, bc = [], {}
        for nme in others:
            ok = preds[nme]["test"] == yt
            b, c = int((ref_ok & ~ok).sum()), int((~ref_ok & ok).sum())
            bc[nme] = (b, c)
            raw_p.append(exact_mcnemar(b, c))
        adj = holm_adjust(raw_p) if raw_p else []
        for nme, p_raw, p_adj in zip(others, raw_p, adj, strict=True):
            by_name[nme]["vs_selected_test"] = {
                "reference_only_correct": bc[nme][0],
                "other_only_correct": bc[nme][1],
                "mcnemar_p": _round(p_raw, 6),
                "mcnemar_p_holm": _round(p_adj, 6),
            }
            by_name[nme]["statistically_tied_with_best"] = bool(p_adj >= ALPHA)
    else:
        test_cmp["method"] = "paired_bootstrap_ci_of_difference"
        ref_rep = bootstrap_scores(task, selection, yt, preds[ref]["test"], k, n_boot, seed)
        for nme in others:
            d = ref_rep - bootstrap_scores(task, selection, yt, preds[nme]["test"], k, n_boot, seed)
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            by_name[nme]["vs_selected_test"] = {"diff_ci_lo": _round(lo), "diff_ci_hi": _round(hi)}
            by_name[nme]["statistically_tied_with_best"] = bool(lo <= 0.0 <= hi)
    by_name[ref]["statistically_tied_with_best"] = True
    by_name[ref]["vs_selected_test"] = None
    ci_metric = "f1_macro" if cls else selection
    for r in rows:
        p = preds[r["name"]]["test"]
        if cls:
            r["test_ci"] = {"metric": "f1_macro", **bootstrap_f1_macro_ci(yt, p, k, n_boot, seed)}
        else:
            rep = bootstrap_scores(task, ci_metric, yt, p, k, n_boot, seed)
            r["test_ci"] = {
                "metric": ci_metric,
                **_ci(rep, _round(score_fn(task, ci_metric, yt, p))),
            }
    test_cmp["alpha"] = ALPHA
    test_cmp["tied_with_selected"] = [
        n for n in others if by_name[n]["statistically_tied_with_best"]
    ]

    # -- 9) advertencias
    split_spec = recipe.get("split") or {}
    if split_spec.get("strategy") == "temporal_blocks":
        warnings.append(
            tr(
                "la partición es por bloques temporales dentro de cada grupo: El test sale de las "
                "mismas sesiones que el train, así que NO mide la generalización a sesiones, "
                "hogares o dispositivos nuevos",
                "the split is by temporal blocks within each group: The test set comes from the "
                "same sessions as train, so it does NOT measure generalization to new sessions, "
                "households or devices",
            )
        )
    if cls:
        small = [classes[i] for i, c in enumerate(np.bincount(yt, minlength=k)) if 0 < c < 20]
        if small:
            warnings.append(
                tr(
                    "clases con < 20 filas en test (métricas e intervalos poco fiables): {s}",
                    "classes with < 20 rows in test (unreliable metrics and intervals): {s}",
                    s=", ".join(map(str, small)),
                )
            )
    if n_test < 100:
        warnings.append(
            tr(
                "el test tiene solo {n} filas: Los intervalos son anchos",
                "the test set has only {n} rows: Intervals are wide",
                n=n_test,
            )
        )
    tied = test_cmp["tied_with_selected"]
    if tied:
        warnings.append(
            tr(
                "en test NO se distingue estadísticamente a {sel} de: {t} (alfa={a}, "
                "comparaciones múltiples con Holm); las diferencias de métrica pueden ser ruido, "
                "decide también por tamaño/velocidad",
                "on test, {sel} is NOT statistically distinguishable from: {t} (alpha={a}, "
                "multiple comparisons with Holm); metric differences may be noise, decide on "
                "size/speed too",
                sel=chosen,
                t=tied,
                a=ALPHA,
            )
        )
    if sel["none_fit"]:
        warnings.append(
            tr(
                "ningún modelo cabe en {kb} KB de flash (solo tablas del modelo): Se muestra el "
                "mejor sin filtrar",
                "no model fits in {kb} KB of flash (model tables only): The best unfiltered is "
                "shown",
                kb=max_flash_kb,
            )
        )
    elif max_flash_kb is not None:
        nofit = [r["name"] for r in rows if r["fits_flash"] is False]
        unk = [r["name"] for r in rows if r["fits_flash"] is None]
        if nofit:
            warnings.append(
                tr(
                    "no caben en {kb} KB y no compiten: {n}",
                    "do not fit in {kb} KB and do not compete: {n}",
                    kb=max_flash_kb,
                    n=nofit,
                )
            )
        if unk:
            warnings.append(
                tr(
                    "sin estimación de flash (no se sabe si caben): {n}",
                    "no flash estimate (unknown whether they fit): {n}",
                    n=unk,
                )
            )
    if sel["best"] != chosen:
        warnings.append(
            tr(
                "se eligió {c} y no {b} (mejor puntaje bruto) por la regla de 1-SE / presupuesto "
                "de flash",
                "{c} was chosen instead of {b} (best raw score) by the 1-SE rule / flash budget",
                c=chosen,
                b=sel["best"],
            )
        )
    warnings.append(
        tr(
            "sesgo de selección: El test ya se usó para comparar; si vas a iterar (cambiar "
            "modelos o hiperparámetros mirando este test) su puntaje deja de ser una estimación "
            "insesgada (Cawley y Talbot 2010). Reserva un conjunto final que no mires.",
            "selection bias: The test set was used to compare; if you keep iterating (changing "
            "models or hyperparameters while looking at this test) its score is no longer an "
            "unbiased estimate (Cawley & Talbot 2010). Keep a final set you never look at.",
        )
    )

    # -- 10) ensamblar
    leaderboard = sorted(
        rows,
        key=lambda r: (
            r["selection_score"] is None,
            -(1 if higher else -1) * (r["selection_score"] or 0.0),
            r["name"],
        ),
    )
    for i, r in enumerate(leaderboard):
        r["rank"] = i + 1
    reason = _reason(sel, by_name.get(chosen), selection, use_cv, max_flash_kb)
    sel_row = by_name.get(chosen)
    out: dict[str, Any] = {
        "task": task,
        "target": target,
        "classes": classes if cls else None,
        "rows": {s: int(len(y[s])) for s in y},
        "features_used": enc.feature_names,
        "selection": {
            "metric": selection,
            "higher_is_better": higher,
            "source": "cv" if use_cv else "val",
            "rule": "one_standard_error",
            "best_by_metric": sel["best"],
            "selected": chosen,
            "within_1se": sel["within_1se"],
            "max_flash_kb": max_flash_kb,
            "none_fit_flash": sel["none_fit"],
            "reason": reason,
            "export": {
                "job": sel_row["export_job"],
                "run_subdir": sel_row["run_subdir"],
                "model": chosen if sel_row["kind"] == "classical" else None,
            }
            if sel_row
            else None,
        },
        "leaderboard": leaderboard,
        "rank_by_metric": rank_by_metric,
        "pareto": front,
        "test_comparison": test_cmp,
        "cv": cv_meta,
        "models_skipped": skipped,
        "run_dirs": run_dirs,
        "out_dir": str(root) if keep else None,
        "training_warnings": training_warnings,
        "warnings": warnings,
        "reproducibility": {
            "seed": seed,
            "n_boot": n_boot,
            "cv_folds": cv_folds,
            "recipe_sha256": hashlib.sha256(
                json.dumps(recipe, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "libraries": _libs(),
        },
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    if cls and sel_row is not None:
        cm = _cm(yt, preds[chosen]["test"], k)
        out["selected_confusion_test"] = {"classes": classes, "matrix": cm.tolist()}
    out = _clean(out)
    from .report import benchmark_summary

    out["summary"] = out["summary_es"] = benchmark_summary(out)
    return out


def _libs() -> dict[str, str]:
    import pandas as pd
    import sklearn

    return {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def _reason(sel, row, metric, use_cv, max_flash_kb) -> str:
    if row is None:
        return tr("ningún modelo tiene puntaje", "no model has a score")
    src = (
        tr("validación cruzada en train", "cross-validation on train")
        if use_cv
        else tr("validación", "validation")
    )
    n = len(sel["within_1se"])
    txt = tr(
        "{m} elegido con {src} y la regla de 1 error estándar: {n} modelo(s) empatan con el "
        "mejor puntaje y se prefiere el exportable, pequeño y rápido.",
        "{m} chosen on {src} with the one-standard-error rule: {n} model(s) tie with the best "
        "score and the exportable, small, fast one is preferred.",
        m=row["name"],
        src=src,
        n=n,
    )
    if max_flash_kb is not None:
        txt += " " + tr(
            "Presupuesto de flash: {kb} KB.", "Flash budget: {kb} KB.", kb=f"{max_flash_kb:g}"
        )
    return txt
