"""De receta a splits limpios: organiza -> parte -> ajusta limpieza con train -> aplica."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tinyml_studio.augment.synth import synthesize
from tinyml_studio.dataset import apply_recipe
from tinyml_studio.split import make_split
from tinyml_studio.split.balance import balance_split

from .steps import Pipeline

SPLITS = ("train", "val", "test")


@dataclass
class Prepared:
    raw: dict[str, pd.DataFrame]  # tras organizar y partir, antes de limpiar
    clean: dict[str, pd.DataFrame]
    pipeline: Pipeline
    target: str | None


def resolve_target(recipe: dict[str, Any]) -> str | None:
    lff = recipe.get("label_from_file", {})
    return lff["column"] if lff.get("enabled") else recipe.get("target")


def cleaning_without_target(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """Pasos de limpieza sin el escalado de la columna OBJETIVO.

    Escalar la variable que se predice haría que el modelo (y el firmware) devolvieran valores
    estandarizados en vez de unidades reales (p. ej. -1.6 en lugar de 64 puntos). Una sugerencia de
    limpieza hecha antes de cambiar el objetivo, o armada a mano, podía incluirla.
    """
    target = resolve_target(recipe)
    out = []
    for step in recipe.get("cleaning", []):
        step = {**step}
        if target and step.get("kind") == "scale":
            params = {**step.get("params", {})}
            cols = [c for c in params.get("columns", []) if c != target]
            if not cols:
                continue
            params["columns"] = cols
            step["params"] = params
        out.append(step)
    return out


def prepare_splits(recipe: dict[str, Any], seed: int = 42) -> Prepared:
    df = apply_recipe(recipe)
    target = resolve_target(recipe)
    if target is not None:
        df = df[df[target].notna()].reset_index(drop=True)  # sin etiqueta no se puede entrenar
    idx = make_split(df, recipe["split"], target, seed)
    mode = recipe["split"].get("balance") or "off"
    if mode != "off" and target is not None and recipe.get("task") == "classification":
        idx, _ = balance_split(df, idx, target, mode, recipe["split"].get("time_column"))
    raw = {s: df.iloc[idx[s]].reset_index(drop=True) for s in SPLITS}
    aug = recipe.get("augment") or {}
    if aug.get("method") not in (None, "off") and target is not None:
        feats = [c for c, r in (recipe.get("roles") or {}).items() if r == "feature"]
        # filas NUEVAS solo en entrenamiento; validación y test quedan con datos reales
        raw["train"], _ = synthesize(raw["train"], feats, target, recipe.get("task", ""), aug, seed)
    pipe = Pipeline.from_list(cleaning_without_target(recipe))
    pipe.fit(raw["train"])  # SOLO train
    clean = {s: pipe.transform(raw[s], is_train=(s == "train")) for s in SPLITS}
    return Prepared(raw=raw, clean=clean, pipeline=pipe, target=target)
