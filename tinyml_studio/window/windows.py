"""Ventanas deslizantes de filas consecutivas (para 1D-CNN y modelos de series temporales).

Las ventanas se construyen DENTRO de cada split y de cada grupo (p. ej. archivo/sesión): una
ventana nunca cruza el límite entre train y val/test, ni entre dos grupos, ni un hueco de tiempo
grande (apagados, pausas), ni un cambio de etiqueta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from tinyml_studio.i18n import tr


@dataclass
class WindowConfig:
    window: int = 16  # filas por ventana
    stride: int = 4  # paso entre ventanas de entrenamiento (val/test usan stride = window)
    max_gap_factor: float = 5.0  # hueco > factor x mediana de dt => se corta la ventana

    def validate(self) -> None:
        if self.window < 2:
            raise ValueError(tr("window debe ser >= 2", "window must be >= 2"))
        if self.stride < 1:
            raise ValueError(tr("stride debe ser >= 1", "stride must be >= 1"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _seconds(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("int64").to_numpy().astype("float64") / 1e9
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    group: pd.Series,
    time: pd.Series | None,
    cfg: WindowConfig,
    stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Devuelve (Xw (m, W, F), yw (m,), idx (m, W) con posiciones de fila en X, resumen).

    ``X``, ``y``, ``group`` y ``time`` están alineados por fila. Sin ``time`` se usa el orden
    de las filas dentro de cada grupo.
    """
    cfg.validate()
    W, st = cfg.window, stride or cfg.stride
    n = len(X)
    t = _seconds(time) if time is not None else np.arange(n, dtype="float64")
    codes = pd.factorize(group)[0]
    order = np.lexsort((t, codes))  # por grupo y, dentro, por tiempo (estable)
    stats = {"rows": n, "segments": 0, "windows": 0, "dropped_mixed_label": 0, "rows_unused": 0}
    idx_all: list[np.ndarray] = []
    for g in np.unique(codes):
        rows = order[codes[order] == g]
        tt = t[rows]
        dt = np.diff(tt)
        pos = dt[dt > 0]
        med = float(np.median(pos)) if len(pos) else 0.0
        cut = np.flatnonzero((dt > cfg.max_gap_factor * med) | (dt < 0)) + 1 if med > 0 else []
        for seg in np.split(rows, cut):
            stats["segments"] += 1
            if len(seg) < W:
                stats["rows_unused"] += len(seg)
                continue
            starts = np.arange(0, len(seg) - W + 1, st)
            win = seg[starts[:, None] + np.arange(W)[None, :]]
            lab = y[win]
            same = (lab == lab[:, :1]).all(axis=1)
            stats["dropped_mixed_label"] += int((~same).sum())
            idx_all.append(win[same])
    if not idx_all:
        raise ValueError(
            tr(
                "no se pudo formar ninguna ventana de {W} filas consecutivas (¿hay muy pocas filas "
                "seguidas, huecos de tiempo o falta la columna de tiempo?)",
                "could not form any window of {W} consecutive rows (are there too few "
                "consecutive rows, time gaps, or a missing time column?)",
                W=W,
            )
        )
    idx = np.vstack(idx_all)
    stats["windows"] = int(len(idx))
    return X[idx].astype(np.float32), y[idx[:, -1]], idx, stats
