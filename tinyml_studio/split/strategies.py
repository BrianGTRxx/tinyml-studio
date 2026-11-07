"""Estrategias de partición train/val/test: aleatoria, estratificada, por grupo y por bloques."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tinyml_studio.i18n import tr

from .blocks import split_temporal_blocks


def _cut(n: int, ratios) -> tuple[int, int]:
    a = int(round(n * ratios[0]))
    b = int(round(n * (ratios[0] + ratios[1])))
    return a, b


def split_random(n: int, ratios, seed: int) -> dict[str, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    a, b = _cut(n, ratios)
    return {"train": idx[:a], "val": idx[a:b], "test": idx[b:]}


def split_stratified(y: pd.Series, ratios, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    codes = pd.factorize(y)[0]
    for k in np.unique(codes):
        idx = rng.permutation(np.flatnonzero(codes == k))
        a, b = _cut(len(idx), ratios)
        out["train"].append(idx[:a])
        out["val"].append(idx[a:b])
        out["test"].append(idx[b:])
    return {k: np.sort(np.concatenate(v)) for k, v in out.items()}


def split_group(groups: pd.Series, ratios, seed: int) -> dict[str, np.ndarray]:
    """Grupos completos van a un solo split (sin fuga entre filas del mismo grupo)."""
    codes, uniques = pd.factorize(groups)
    if len(uniques) < 3:
        raise ValueError(
            tr(
                "partición por grupo necesita >= 3 grupos (hay {n})",
                "group split needs >= 3 groups (found {n})",
                n=len(uniques),
            )
        )
    order = np.random.default_rng(seed).permutation(len(uniques))
    sizes = np.bincount(codes, minlength=len(uniques))
    n, assign, acc = len(groups), {}, 0
    for g in order:
        frac = acc / n
        split = "train" if frac < ratios[0] else ("val" if frac < ratios[0] + ratios[1] else "test")
        assign[g] = split
        acc += sizes[g]
    # garantizar que val y test no queden vacíos
    for s in ("val", "test"):
        if s not in assign.values():
            donor = next(g for g in order[::-1] if assign[g] == "train")
            assign[donor] = s
    labels = np.array([assign[c] for c in codes])
    return {s: np.flatnonzero(labels == s) for s in ("train", "val", "test")}


def make_split(
    df: pd.DataFrame, spec: dict[str, Any], target: str | None, seed: int = 42
) -> dict[str, np.ndarray]:
    """Aplica la estrategia de la receta (``spec``) y devuelve posiciones por split."""
    ratios = tuple(spec.get("ratios", (0.8, 0.1, 0.1)))
    strat = spec["strategy"]
    if strat == "temporal_blocks":
        if not spec.get("time_column"):
            raise ValueError(
                tr("temporal_blocks requiere time_column", "temporal_blocks requires time_column")
            )
        return split_temporal_blocks(
            df,
            group_column=spec.get("group_column"),
            time_column=spec["time_column"],
            ratios=ratios,
        )
    if strat == "group":
        return split_group(df[spec["group_column"]], ratios, seed)
    if strat == "stratified":
        if target is None:
            raise ValueError(
                tr("stratified requiere columna objetivo", "stratified requires a target column")
            )
        return split_stratified(df[target], ratios, seed)
    if strat == "random":
        return split_random(len(df), ratios, seed)
    raise ValueError(
        tr(
            "estrategia de partición desconocida: {s}",
            "unknown split strategy: {s}",
            s=strat,
        )
    )
