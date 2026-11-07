"""Probar un modelo clásico ya entrenado con valores que escribe el usuario, y avisar si salen de rango.

Hace exactamente lo que hará el firmware: valores CRUDOS -> limpieza y codificación del entrenamiento
(``DeviceSpec``) -> modelo. El C exportado se verifica contra esta misma ruta en ``export_run``.

``input_ranges`` guarda en ``input_ranges.json`` el mínimo y el máximo que cada columna numérica tuvo en
el ENTRENAMIENTO: fuera de ese rango un modelo (sobre todo uno lineal) extrapola y el resultado es poco
fiable. Se usa aquí y en el código C exportado (``tm_out_of_range``).
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tinyml_studio.clean import Pipeline, prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.train.features import FeatureEncoder

from .spec import build_spec, emulate_features, raw_inputs


def _load(run_dir: str | Path, model: str | None):
    import joblib

    run = Path(run_dir)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    recipe = json.loads((run / "recipe_fitted.json").read_text(encoding="utf-8"))
    enc = FeatureEncoder.from_dict(json.loads((run / "encoder.json").read_text(encoding="utf-8")))
    name = model or results["selection"]["best"]
    path = run / f"model_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            tr(
                "no existe {f}: Entrena con ese modelo primero",
                "{f} does not exist: Train with that model first",
                f=path.name,
            )
        )
    est = joblib.load(path)
    spec = build_spec(Pipeline.from_list(recipe["cleaning"], fitted=True), enc)
    return run, results, recipe, enc, est, spec, name


def input_ranges(run_dir: str | Path, recipe: dict, results: dict, spec) -> dict[str, dict]:
    """``{columna: {min, max, median}}`` de las columnas numéricas crudas del entrenamiento."""
    run = Path(run_dir)
    cache = run / "input_ranges.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    prep = prepare_splits(recipe, seed=results["reproducibility"]["seed"])
    train = prep.raw["train"]
    out: dict[str, dict] = {}
    for c in spec.num_names:
        v = pd.to_numeric(train[c], errors="coerce").dropna()
        if len(v):
            out[c] = {"min": float(v.min()), "max": float(v.max()), "median": float(v.median())}
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def describe(run_dir: str | Path, model: str | None = None) -> dict[str, Any]:
    """Qué entradas espera el modelo (para dibujar el formulario)."""
    run, results, recipe, _enc, _est, spec, name = _load(run_dir, model)
    ranges = input_ranges(run, recipe, results, spec)
    return {
        "model": name,
        "task": results["task"],
        "target": recipe.get("target"),
        "classes": spec.classes,
        "numeric": [{"name": c, **ranges.get(c, {})} for c in spec.num_names],
        "categorical": [{"name": c, "values": spec.cat_values[c]} for c in spec.cat_names],
    }


def _clean_number(v: Any) -> float:
    if v is None or (isinstance(v, str) and not v.strip()):
        return math.nan
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return math.nan


def predict_values(
    run_dir: str | Path, values: dict[str, Any], model: str | None = None
) -> dict[str, Any]:
    """Predice con valores crudos. Falta o texto no numérico = NaN (el modelo usa su relleno)."""
    run, results, recipe, enc, est, spec, name = _load(run_dir, model)
    ranges = input_ranges(run, recipe, results, spec)
    row = {c: _clean_number(values.get(c)) for c in spec.num_names}
    for c in spec.cat_names:
        v = values.get(c)
        row[c] = None if v is None or str(v) == "" else str(v)
    num, cat = raw_inputs(spec, pd.DataFrame([row]))
    feats = emulate_features(spec, num, cat)
    out: dict[str, Any] = {"model": name, "task": results["task"]}
    if results["task"] == "classification":
        pred = int(est.predict(feats)[0])
        out["class_index"] = pred
        out["class"] = spec.classes[pred] if pred < len(spec.classes) else str(pred)
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(feats)[0]
            classes = [int(k) for k in getattr(est, "classes_", range(len(proba)))]
            out["probabilities"] = {
                spec.classes[k] if k < len(spec.classes) else str(k): float(p)
                for k, p in zip(classes, proba, strict=True)
            }
    else:
        out["value"] = float(est.predict(feats)[0])
    out["out_of_range"] = [
        {"column": c, "value": float(row[c]), "min": r["min"], "max": r["max"]}
        for c, r in ranges.items()
        if not math.isnan(row[c]) and (row[c] < r["min"] or row[c] > r["max"])
    ]
    out["missing"] = [c for c in spec.num_names if math.isnan(row[c])] + [
        c for c in spec.cat_names if row[c] is None
    ]
    out["unknown_category"] = [
        c for c in spec.cat_names if row[c] is not None and row[c] not in spec.cat_values[c]
    ]
    out["features"] = np.asarray(feats[0], dtype=float).tolist()
    return out
