"""Entrena varios modelos clásicos sobre los splits de una receta y compara honestamente."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tinyml_studio.clean import prepare_splits
from tinyml_studio.clean.prepare import resolve_target
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit

from . import models as M
from . import qual
from .features import FeatureEncoder, encoding_advice
from .metrics import classification_metrics, regression_metrics

NON_FEATURE_ROLES = {"target", "group", "time", "meta", "drop"}


def feature_columns(recipe: dict, df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Columnas con rol 'feature' en la receta (nunca objetivo, grupo, tiempo ni metadatos)."""
    roles = recipe.get("roles", {})
    target = resolve_target(recipe)
    cols = [c for c, r in roles.items() if r == "feature" and c in df.columns]
    cols = [c for c in cols if c != target and c not in exclude]
    if not cols:
        raise ValueError(
            tr(
                "la receta no tiene columnas con rol 'feature'",
                "the recipe has no columns with role 'feature'",
            )
        )
    return cols


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _quality(est, task, X, y, classes):
    if task == "classification":
        return qual.classification_block(y, {s: qual.sk_scores(est, X[s]) for s in y}, classes)
    return qual.regression_block(y, {s: est.predict(X[s]) for s in y})


def _score_key(task: str) -> str:
    return "f1_macro" if task == "classification" else "r2"


def _metrics(task, y, pred, classes):
    return (
        classification_metrics(y, pred, classes)
        if task == "classification"
        else (regression_metrics(y, pred))
    )


def train_models(
    recipe: dict[str, Any],
    model_names: list[str] | None = None,
    *,
    seed: int = 42,
    params: dict[str, dict] | None = None,
    exclude: tuple[str, ...] = (),
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    task = recipe["task"]
    if task not in ("classification", "regression"):
        raise ValueError(
            tr(
                "tarea no entrenable con modelos supervisados: {task}",
                "task cannot be trained with supervised models: {task}",
                task=task,
            )
        )
    emit(
        "prepare",
        3,
        tr(
            "Preparando datos (organizar, partir, limpiar)…",
            "Preparing data (organize, split, clean)…",
        ),
    )
    prep = prepare_splits(recipe, seed=seed)
    target = prep.target
    if target is None:
        raise ValueError(
            tr(
                "la receta no tiene columna objetivo (ni label_from_file)",
                "the recipe has no target column (nor label_from_file)",
            )
        )
    feats = feature_columns(recipe, prep.clean["train"], exclude)
    enc = FeatureEncoder.fit(prep.clean["train"], feats, target, task, recipe.get("encoding"))
    X = {s: enc.transform(df) for s, df in prep.clean.items()}
    if task == "classification":
        y = {s: enc.encode_target(df[target]) for s, df in prep.clean.items()}
    else:
        y = {s: pd.to_numeric(df[target]).to_numpy(dtype="float64") for s, df in prep.clean.items()}
    classes = enc.classes
    names = model_names or list(M.DEFAULT_SET[task])
    params = params or {}
    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}

    # línea base: clase mayoritaria / media de train
    if task == "classification":
        major = int(np.bincount(y["train"]).argmax())
        base_pred = {s: np.full(len(y[s]), major) for s in ("val", "test")}
    else:
        base_pred = {s: np.full(len(y[s]), y["train"].mean()) for s in ("val", "test")}
    baseline = {s: _metrics(task, y[s], base_pred[s], classes) for s in ("val", "test")}

    for i, name in enumerate(names):
        emit(
            "train",
            8 + 85 * i / len(names),
            tr(
                "Entrenando {name} ({i}/{n})…",
                "Training {name} ({i}/{n})…",
                name=name,
                i=i + 1,
                n=len(names),
            ),
        )
        est, eff = M.build(name, task, seed, params.get(name))
        Xtr, ytr = X["train"], y["train"]
        notes = []
        cap = eff.get("max_train_rows") if name in M.ROW_CAPPED else None
        if cap and len(Xtr) > cap:
            idx = np.random.default_rng(seed).choice(len(Xtr), cap, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
            notes.append(
                tr(
                    "{m} entrenada con una muestra aleatoria de {cap} de {n} filas",
                    "{m} trained on a random sample of {cap} of {n} rows",
                    m=name.upper(),
                    cap=cap,
                    n=len(X["train"]),
                )
            )
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = {s: est.predict(X[s]) for s in ("val", "test")}
        pred_us = (time.perf_counter() - t0) / (len(X["val"]) + len(X["test"])) * 1e6
        results[name] = {
            "params": {k: v for k, v in eff.items()},
            "fit_seconds": round(fit_s, 2),
            "predict_us_per_row_cpu_cloud": round(pred_us, 2),
            "complexity": M.complexity(name, est),
            "top_features": M.importances(name, est, enc.feature_names),
            "val": _metrics(task, y["val"], pred["val"], classes),
            "test": _metrics(task, y["test"], pred["test"], classes),
            "notes": notes,
        }
        fitted[name] = est
        try:
            results[name]["quality"] = _quality(est, task, X, y, classes)
        except Exception as exc:  # noqa: BLE001 - extra diagnostics must never break training
            results[name]["quality"] = {"error": str(exc)}
        emit(
            "model_done",
            8 + 85 * (i + 1) / len(names),
            tr("{name} listo", "{name} done", name=name),
            model=name,
        )

    key = _score_key(task)
    best = max(results, key=lambda n: results[n]["val"][key])  # se elige con VAL, nunca con test
    enc_warnings: list[str] = []
    for n in results:
        for w in encoding_advice(enc, n):
            if w not in enc_warnings:
                enc_warnings.append(w)
    out = {
        "task": task,
        "target": target,
        "selection": {"metric": key, "split": "val", "best": best},
        "classes": classes,
        "features_used": enc.feature_names,
        "features_skipped": enc.skipped,
        "features_excluded_by_user": list(exclude),
        "rows": {s: int(len(y[s])) for s in y},
        "baseline": {
            s: {k: v for k, v in m.items() if k in ("accuracy", "f1_macro", "r2", "mae")}
            for s, m in baseline.items()
        },
        "models": results,
        "reproducibility": {
            "seed": seed,
            "python": platform.python_version(),
            "libraries": _versions(),
            "recipe_sha256": hashlib.sha256(
                json.dumps(recipe, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "data_sha256": {Path(f).name: _sha(f) for f in recipe["files"]},
        },
        "warnings": _warnings(results, baseline, task, key, recipe.get("split")) + enc_warnings,
    }
    if out_dir is not None:
        _save(Path(out_dir), out, recipe, prep, enc, fitted)
    return out


def _versions() -> dict[str, str]:
    import sklearn

    return {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def _warnings(results, baseline, task, key, split=None) -> list[str]:
    w = []
    if split and split.get("strategy") == "temporal_blocks":
        w.append(
            tr(
                "la partición es por bloques temporales dentro de cada grupo: El test sale de las "
                "mismas sesiones que el train, así que NO mide la generalización a sesiones, "
                "hogares o dispositivos nuevos",
                "the split is by temporal blocks within each group: The test set comes from the "
                "same sessions as train, so it does NOT measure generalization to new sessions, "
                "households or devices",
            )
        )
    for n, r in results.items():
        gap = r["val"][key] - r["test"][key]
        if gap > 0.1:
            w.append(
                tr(
                    "{n}: {key} baja {gap:.2f} de val a test "
                    "(¿cambio de distribución o sobreajuste?)",
                    "{n}: {key} drops {gap:.2f} from val to test "
                    "(distribution shift or overfitting?)",
                    n=n,
                    key=key,
                    gap=gap,
                )
            )
    b = baseline["test"].get(key)
    if b is not None and all(r["test"][key] <= b + 0.01 for r in results.values()):
        w.append(
            tr(
                "ningún modelo supera claramente la línea base en test",
                "no model clearly beats the baseline on test",
            )
        )
    if task == "classification":
        small = [
            c
            for c, d in next(iter(results.values()))["test"]["per_class"].items()
            if d["support"] < 20
        ]
        if small:
            w.append(
                tr(
                    "clases con < 20 filas en test (métricas poco fiables): {small}",
                    "classes with < 20 rows in test (unreliable metrics): {small}",
                    small=", ".join(map(str, small)),
                )
            )
    return w


def _save(out_dir: Path, result, recipe, prep, enc, fitted: dict) -> None:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pipeline_fitted = {**recipe, "cleaning": prep.pipeline.to_list()}
    (out_dir / "recipe_fitted.json").write_text(
        json.dumps(pipeline_fitted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "encoder.json").write_text(
        json.dumps(enc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, model in fitted.items():  # pickle: abrir solo archivos propios
        joblib.dump(model, out_dir / f"model_{name}.joblib")
