"""Perfilado de un dataset: forma, tipos, nulos, balance de clases, estadísticos.

La salida es un dict JSON-serializable (la API y el CLI la devuelven tal cual).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from tinyml_studio.i18n import tr
from tinyml_studio.io.schema import candidate_features, detect_schema


def _num(x: Any) -> float | None:
    """float JSON-seguro (NaN/inf -> None)."""
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def profile_dataframe(df: pd.DataFrame, *, label_column: str | None = None) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    columns: list[dict[str, Any]] = []
    for c in df.columns:
        s = df[c]
        nulls = int(s.isna().sum())
        info: dict[str, Any] = {
            "name": str(c),
            "dtype": str(s.dtype),
            "nulls": nulls,
            "nulls_pct": round(100 * nulls / n_rows, 3) if n_rows else 0.0,
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_bool_dtype(s):
            pass
        elif pd.api.types.is_numeric_dtype(s):
            f = pd.to_numeric(s, errors="coerce").astype("float64")
            info["constant"] = bool(f.nunique(dropna=True) <= 1)
            info.update(
                min=_num(f.min()),
                max=_num(f.max()),
                mean=_num(f.mean()),
                std=_num(f.std()),
                median=_num(f.median()),
                q01=_num(f.quantile(0.01)),
                q99=_num(f.quantile(0.99)),
                inf=int((~f.isna() & ~f.map(math.isfinite)).sum()),
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            info.update(min=str(s.min()), max=str(s.max()))
        else:
            top = s.value_counts(dropna=True).head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        columns.append(info)

    out: dict[str, Any] = {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "schema": detect_schema(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "candidate_features": candidate_features(df),
        "columns": columns,
    }
    if label_column is not None:
        if label_column not in df.columns:
            raise KeyError(
                tr(
                    "columna de etiqueta inexistente: {c}",
                    "label column does not exist: {c}",
                    c=label_column,
                )
            )
        counts = df[label_column].value_counts(dropna=False)
        out["class_balance"] = {
            "label_column": label_column,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "imbalance_ratio": _num(counts.max() / counts.min()) if counts.min() else None,
        }
    return out
