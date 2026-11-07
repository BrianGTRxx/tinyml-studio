"""Equilibrado de clases por submuestreo: la clase grande se recorta al tamaño de la más pequeña.

Nunca al revés (no se inventan ni se repiten filas de la clase pequeña). Los recortes se reparten
de forma pareja a lo largo del tiempo (una de cada k filas), así la clase grande conserva toda su
variedad en lugar de quedarse solo con un tramo. Por defecto solo se recorta TRAIN: validación y
test conservan la proporción real, que es la que verá el aparato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODES = ("off", "train", "all")


def _even_pick(pos: np.ndarray, n: int) -> np.ndarray:
    """n posiciones repartidas de forma pareja entre ``pos`` (ya ordenadas)."""
    if n >= len(pos):
        return pos
    return pos[np.linspace(0, len(pos) - 1, n).round().astype(int)]


def balance_split(
    df: pd.DataFrame,
    idx: dict[str, np.ndarray],
    target: str,
    mode: str,
    time_column: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, tuple[int, int]]]]:
    """Devuelve (índices recortados, resumen ``{split: {clase: (antes, después)}}``)."""
    if mode not in MODES:
        raise ValueError(f"balance must be one of {MODES}")
    if mode == "off":
        return idx, {}
    out = dict(idx)
    summary: dict[str, dict[str, tuple[int, int]]] = {}
    for split in ("train", "val", "test") if mode == "all" else ("train",):
        pos = np.asarray(idx[split])
        if not len(pos):
            continue
        y = df[target].to_numpy()[pos]
        classes, counts = np.unique(y.astype(str), return_counts=True)
        if len(classes) < 2:
            continue
        n = int(counts.min())
        keep: list[np.ndarray] = []
        info: dict[str, tuple[int, int]] = {}
        for c, cnt in zip(classes, counts, strict=True):
            p = pos[y.astype(str) == c]
            if time_column and time_column in df.columns:
                p = p[np.argsort(df[time_column].to_numpy()[p], kind="stable")]
            kept = _even_pick(p, n)
            keep.append(kept)
            info[str(c)] = (int(cnt), int(len(kept)))
        out[split] = np.sort(np.concatenate(keep))
        summary[split] = info
    return out, summary
