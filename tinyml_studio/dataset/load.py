"""Carga y coerción de una tabla: arregla lo que suele venir mal en CSV de origen dispar."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tinyml_studio.io import read_influx_csv

NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "#n/a", "missing"}
_BOOL_TRUE = {"true", "yes", "y", "si", "sí", "t", "verdadero"}
_BOOL_FALSE = {"false", "no", "n", "f", "falso"}
_DATE_HINTS = ("date", "fecha", "time", "tiempo", "timestamp", "datetime", "hora")


@dataclass
class LoadedTable:
    path: str
    df: pd.DataFrame
    rows: int
    encoding: str
    separator: str
    annotation_rows: int
    coercions: list[dict] = field(default_factory=list)


def _is_texty(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def coerce_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Aplica coerciones deterministas y devuelve (df, notas). Cada nota es serializable."""
    df = df.copy()
    notes: list[dict] = []
    for col in df.columns:
        s = df[col]
        if not _is_texty(s):
            continue
        s = s.astype("string").str.strip()
        na_mask = s.str.lower().isin(NA_TOKENS) & s.notna()
        if na_mask.any():
            notes.append({"column": str(col), "action": "na_tokens", "count": int(na_mask.sum())})
            s = s.mask(na_mask)
        nn = s.dropna()
        if nn.empty:
            df[col] = s
            continue

        # numérico (admite coma decimal "1,5")
        cand = nn.str.replace(",", ".", regex=False)
        num = pd.to_numeric(cand, errors="coerce")
        if num.notna().mean() >= 0.98:
            comma = bool(nn.str.contains(",", regex=False).any())
            full = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
            df[col] = full
            notes.append({"column": str(col), "action": "numeric", "decimal_comma": comma})
            continue

        # booleano en texto
        low = set(nn.str.lower().unique())
        if len(low) == 2 and low <= (_BOOL_TRUE | _BOOL_FALSE) and low & _BOOL_TRUE:
            df[col] = s.str.lower().isin(_BOOL_TRUE).where(s.notna()).astype("boolean")
            notes.append({"column": str(col), "action": "boolean"})
            continue

        # fecha/hora
        sample = nn.head(200)
        looks = sample.str.contains(r"\d", regex=True).all() and (
            sample.str.contains(r"[-/:]", regex=True).mean() > 0.9
        )
        if looks or any(h in str(col).lower() for h in _DATE_HINTS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(s, errors="coerce", utc=True)
            if dt.notna().sum() / len(nn) >= 0.9:
                df[col] = dt
                notes.append({"column": str(col), "action": "datetime"})
                continue
        df[col] = s
    return df, notes


def load_table(path: str | Path) -> LoadedTable:
    res = read_influx_csv(path)
    df, notes = coerce_frame(res.df)
    return LoadedTable(
        path=str(path),
        df=df,
        rows=len(df),
        encoding=res.encoding,
        separator=res.separator,
        annotation_rows=res.annotation_rows,
        coercions=notes,
    )
