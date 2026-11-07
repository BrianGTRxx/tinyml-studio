"""Comparación antes/después: tabla de estadísticos y gráficas Plotly (histogramas agregados)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tinyml_studio.i18n import tr


def _f(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def unit_view(prep) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(antes, después) de train en las MISMAS unidades: se omite el escalado.

    Comparar antes/después con el escalado incluido no tiene sentido (los ejes cambian de
    unidades); aquí se ven los efectos de centinelas, absurdos, imputación y atípicos.
    """
    from .steps import Pipeline

    pipe = Pipeline([s for s in prep.pipeline.steps if s.kind != "scale"])
    pipe.fitted = True  # los pasos ya vienen ajustados con train
    return prep.raw["train"], pipe.transform(prep.raw["train"], is_train=True)


def numeric_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        c
        for c in columns
        if c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def compare_stats(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    for c in columns:
        if c not in before.columns or c not in after.columns:
            continue
        b, a = _f(before[c]), _f(after[c])
        rows.append(
            {
                "column": c,
                "rows_before": int(len(b)),
                "rows_after": int(len(a)),
                "nulls_before": int(b.isna().sum()),
                "nulls_after": int(a.isna().sum()),
                "min_before": _r(b.min()),
                "min_after": _r(a.min()),
                "max_before": _r(b.max()),
                "max_after": _r(a.max()),
                "mean_before": _r(b.mean()),
                "mean_after": _r(a.mean()),
                "std_before": _r(b.std()),
                "std_after": _r(a.std()),
            }
        )
    return rows


def _r(x):
    return None if x is None or not np.isfinite(x) else float(x)


def histogram_data(before: pd.Series, after: pd.Series, bins: int = 40) -> dict:
    """Histogramas con los mismos cortes (ya agregados: no se envían miles de puntos crudos).

    El rango sale de los datos *después* de limpiar (ampliado 25 %); lo que quede fuera
    en "antes" se cuenta aparte (``outside_before``), así un absurdo no aplasta la gráfica.
    """
    b, a = _f(before).dropna(), _f(after).dropna()
    ref = a if len(a) else b
    if ref.empty:
        return {"edges": [], "before": [], "after": [], "outside_before": 0}
    lo, hi = float(ref.min()), float(ref.max())
    pad = (hi - lo) * 0.25 or 1.0
    lo, hi = lo - pad, hi + pad
    edges = np.linspace(lo, hi, bins + 1)
    return {
        "edges": edges.tolist(),
        "before": np.histogram(b, edges)[0].tolist(),
        "after": np.histogram(a, edges)[0].tolist(),
        "outside_before": int(((b < lo) | (b > hi)).sum()),
    }


def figure(before: pd.DataFrame, after: pd.DataFrame, columns: list[str], cols_per_row: int = 3):
    """Figura Plotly: un histograma antes/después por columna."""
    from plotly.subplots import make_subplots

    cols = [c for c in columns if c in before.columns and c in after.columns]
    rows = max(1, -(-len(cols) // cols_per_row))
    titles = []
    data = {c: histogram_data(before[c], after[c]) for c in cols}
    for c in cols:
        out = data[c]["outside_before"]
        titles.append(
            f"{c}"
            + (
                tr(" ({n} fuera de rango antes)", " ({n} out of range before)", n=out)
                if out
                else ""
            )
        )
    fig = make_subplots(rows=rows, cols=cols_per_row, subplot_titles=titles)
    for i, c in enumerate(cols):
        d, r, k = data[c], i // cols_per_row + 1, i % cols_per_row + 1
        mids = [(x + y) / 2 for x, y in zip(d["edges"][:-1], d["edges"][1:], strict=False)]
        fig.add_bar(
            x=mids,
            y=d["before"],
            name=tr("antes", "before"),
            marker_color="#c0504d",
            opacity=0.6,
            showlegend=i == 0,
            row=r,
            col=k,
        )
        fig.add_bar(
            x=mids,
            y=d["after"],
            name=tr("después", "after"),
            marker_color="#4f81bd",
            opacity=0.6,
            showlegend=i == 0,
            row=r,
            col=k,
        )
    fig.update_layout(
        barmode="overlay",
        height=300 * rows,
        title=tr("Antes / después de la limpieza (train)", "Before / after cleaning (train)"),
    )
    return fig
