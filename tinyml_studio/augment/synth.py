"""Aumento de datos: crear filas NUEVAS a partir de las que ya hay (no confundir con imputar faltantes).

Dos técnicas simples y honestas para datos numéricos en tablas:

* ``noise``: copia una fila de entrenamiento y le suma un ruido gaussiano pequeño en cada columna numérica
  (``noise_frac`` veces la desviación típica de esa columna).
* ``interpolate``: elige una fila y uno de sus ``k`` vecinos más cercanos de la MISMA clase (en clasificación) o
  de todo el entrenamiento (en regresión) y crea una fila intermedia (idea de SMOTE; en regresión también
  se interpola el objetivo, como en SMOTER).

Reglas que no se rompen:
* Solo se aumenta el ENTRENAMIENTO. Validación y test quedan con datos reales.
* Las filas nuevas no aportan información que no estuviera ya en los datos: ayudan a que el modelo no se
  sobreajuste o a equilibrar clases, no a "descubrir" cosas nuevas.
* Las columnas de texto o de metadatos de la fila nueva se copian de la fila base; la clase se conserva.
* La validación cruzada dentro del entrenamiento sale optimista con filas sintéticas (hermanas de una fila real
  caen en pliegues distintos): hay que fiarse de validación y test.
"""
# ruff: noqa: E501

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tinyml_studio.i18n import tr

METHODS = ("off", "noise", "interpolate")
MODES = ("ratio", "balance")
POOL_MAX = 2000  # filas de referencia para buscar vecinos (limita el coste en datasets grandes)


def _numeric_features(df: pd.DataFrame, features: list[str], target: str | None) -> list[str]:
    return [
        c
        for c in features
        if c in df.columns and c != target and pd.api.types.is_numeric_dtype(df[c])
    ]


def synthesize(
    train: pd.DataFrame,
    features: list[str],
    target: str | None,
    task: str,
    cfg: dict[str, Any],
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Devuelve ``(entrenamiento aumentado, resumen)``. Sin cambios si el método es ``off``."""
    method = str(cfg.get("method") or "off")
    summary: dict[str, Any] = {"method": method, "added": 0, "per_class": {}}
    if (
        method == "off"
        or target is None
        or task not in ("classification", "regression")
        or train.empty
    ):
        return train, summary
    if method not in METHODS:
        raise ValueError(
            tr("método de aumento desconocido: {m}", "unknown augmentation method: {m}", m=method)
        )
    mode = str(cfg.get("mode") or "ratio")
    if mode not in MODES:
        raise ValueError(
            tr("modo de aumento desconocido: {m}", "unknown augmentation mode: {m}", m=mode)
        )
    cols = _numeric_features(train, features, target)
    if not cols:
        raise ValueError(
            tr(
                "el aumento necesita columnas numéricas de entrada",
                "augmentation needs numeric input columns",
            )
        )
    rng = np.random.default_rng(seed)
    ratio = max(0.0, float(cfg.get("ratio", 1.0)))
    noise_frac = max(0.0, float(cfg.get("noise_frac", 0.05)))
    k = max(1, int(cfg.get("k", 5)))
    cap = max(1, int(cfg.get("max_synthetic", 60000)))
    df = train.reset_index(drop=True)
    X = df[cols].to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xf = np.where(np.isnan(X), med, X)  # solo para medir distancias
    std = np.nanstd(X, axis=0)
    std = np.where((std > 0) & np.isfinite(std), std, 1.0)
    Z = Xf / std

    # cuántas filas nuevas por grupo
    groups: dict[Any, np.ndarray] = {}
    if task == "classification":
        for c, idx in df.groupby(target, sort=True).indices.items():
            groups[c] = np.asarray(idx)
    else:
        groups["__all__"] = np.arange(len(df))
    if mode == "balance" and task == "classification":
        top = max(len(v) for v in groups.values())
        plan = {c: top - len(v) for c, v in groups.items()}
    else:
        plan = {c: int(round(len(v) * ratio)) for c, v in groups.items()}
    total = sum(plan.values())
    if total > cap:  # recorte proporcional al tope
        scale = cap / total
        plan = {c: int(n * scale) for c, n in plan.items()}

    rows: list[pd.DataFrame] = []
    for c, n_new in plan.items():
        if n_new <= 0:
            continue
        members = groups[c]
        base = rng.choice(members, size=n_new, replace=True)
        new = df.iloc[base].copy().reset_index(drop=True)
        newX = X[base].copy()
        if method == "noise":
            noise = rng.normal(0.0, 1.0, size=newX.shape) * (noise_frac * std)
            newX = newX + noise
        else:  # interpolate
            if len(members) >= 2:  # con una sola fila no hay con quién interpolar: queda como copia
                pool = (
                    members
                    if len(members) <= POOL_MAX
                    else rng.choice(members, POOL_MAX, replace=False)
                )
                lam = rng.uniform(0.0, 1.0, size=(n_new, 1))
                neigh = np.empty(n_new, dtype=int)
                kk = min(k, len(pool) - 1)
                for s in range(0, n_new, 512):
                    ch = base[s : s + 512]
                    d = ((Z[ch][:, None, :] - Z[pool][None, :, :]) ** 2).sum(axis=2)
                    d[pool[None, :] == ch[:, None]] = np.inf
                    near = np.argpartition(d, kk - 1, axis=1)[:, :kk]
                    pick = near[np.arange(len(ch)), rng.integers(0, kk, size=len(ch))]
                    neigh[s : s + len(ch)] = pool[pick]
                nb = X[neigh]
                mixed = X[base] + lam * (nb - X[base])
                # si algún valor faltaba en una de las dos filas, se queda el que exista
                mixed = np.where(np.isnan(X[base]), nb, mixed)
                mixed = np.where(np.isnan(nb), X[base], mixed)
                newX = mixed
                if task == "regression":
                    y0 = df[target].to_numpy(dtype=float)[base]
                    y1 = df[target].to_numpy(dtype=float)[neigh]
                    new[target] = y0 + lam[:, 0] * (y1 - y0)
        for j, col in enumerate(cols):
            vals = newX[:, j]
            if pd.api.types.is_integer_dtype(df[col]):
                vals = np.round(vals)
            new[col] = vals
        rows.append(new)
        summary["per_class"][str(c)] = int(n_new)
    if not rows:
        return train, summary
    out = pd.concat([df, *rows], ignore_index=True)
    summary["added"] = int(len(out) - len(df))
    return out, summary
