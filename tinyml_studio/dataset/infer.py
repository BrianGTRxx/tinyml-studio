"""Inferencia del tipo semántico de una columna, con motivo (para explicarlo al usuario)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tinyml_studio.i18n import tr

from .names import name_tokens

ID_TOKENS = {"id", "uuid", "guid", "index", "idx", "key", "codigo", "code", "num", "number"}
KINDS = (
    "empty",
    "constant",
    "binary",
    "datetime",
    "id",
    "numeric_continuous",
    "numeric_discrete",
    "categorical",
    "text",
)


@dataclass
class KindInfo:
    kind: str
    reason: str
    high_cardinality: bool = False


def _integer_valued(f: pd.Series) -> bool:
    v = f.dropna()
    return bool(len(v)) and bool(np.all(np.isfinite(v))) and bool((v == np.round(v)).all())


def infer_kind(s: pd.Series, name: str) -> KindInfo:
    n = int(s.notna().sum())
    if n == 0:
        return KindInfo("empty", tr("todos los valores son nulos", "all values are null"))
    nunique = int(s.nunique(dropna=True))
    if nunique <= 1:
        v0 = s.dropna().iloc[0]
        v0 = v0.item() if hasattr(v0, "item") else v0
        return KindInfo("constant", tr("un único valor ({v})", "a single value ({v})", v=repr(v0)))
    toks = name_tokens(name)
    id_name = bool(toks & ID_TOKENS)

    if pd.api.types.is_bool_dtype(s):
        return KindInfo("binary", tr("booleano", "boolean"))
    if pd.api.types.is_datetime64_any_dtype(s):
        return KindInfo("datetime", tr("fecha/hora", "date/time"))

    if pd.api.types.is_numeric_dtype(s):
        f = pd.to_numeric(s, errors="coerce").astype("float64")
        if nunique == 2 and set(f.dropna().unique()) <= {0.0, 1.0}:
            return KindInfo("binary", tr("numérica con solo 0/1", "numeric with only 0/1"))
        if _integer_valued(f):
            ratio = nunique / n
            v = f.dropna()
            monotonic = bool(v.is_monotonic_increasing or v.is_monotonic_decreasing)
            if n >= 20 and ratio >= 0.98 and (monotonic or id_name):
                return KindInfo(
                    "id",
                    tr(
                        "entero casi único y monótono: Parece contador/identificador",
                        "nearly unique, monotonic integer: Looks like a counter/identifier",
                    ),
                )
            if id_name and ratio >= 0.9:
                return KindInfo(
                    "id",
                    tr(
                        "nombre de identificador y valores casi únicos",
                        "identifier-like name and nearly unique values",
                    ),
                )
            if nunique <= 20 or (nunique <= 50 and ratio < 0.05):
                return KindInfo(
                    "numeric_discrete",
                    tr(
                        "entera con {n} valores distintos",
                        "integer with {n} distinct values",
                        n=nunique,
                    ),
                )
        return KindInfo(
            "numeric_continuous",
            tr("numérica con {n} valores distintos", "numeric with {n} distinct values", n=nunique),
        )

    # texto / categórica
    v = s.dropna().astype(str)
    ratio = nunique / n
    mean_words = float(v.str.split().str.len().mean())
    if nunique == 2:
        return KindInfo("binary", tr("dos categorías", "two categories"))
    if (id_name and ratio >= 0.5) or (ratio >= 0.95 and n >= 20 and mean_words < 4):
        return KindInfo(
            "id",
            tr(
                "texto casi único por fila: Identificador", "nearly unique text per row: Identifier"
            ),
        )
    if mean_words >= 5 or v.str.len().mean() > 60:
        return KindInfo(
            "text",
            tr("texto libre (~{w:.0f} palabras)", "free text (~{w:.0f} words)", w=mean_words),
        )
    return KindInfo(
        "categorical",
        tr("{n} categorías", "{n} categories", n=nunique),
        high_cardinality=nunique > 50,
    )


NUMERIC_KINDS = {"numeric_continuous", "numeric_discrete", "binary"}


def is_numeric_kind(kind: str) -> bool:
    return kind in NUMERIC_KINDS


def is_per_file_counter(df: pd.DataFrame, col: str, group: str = "source") -> bool:
    """Entero que casi siempre crece (o decrece) fila a fila DENTRO de cada archivo.

    Tolera huecos, repetidos y algún reinicio: >= 95 % de los saltos con el mismo signo
    y >= 90 % de valores distintos.
    """
    seen = 0
    for _, g in df.groupby(group, sort=False):
        v = pd.to_numeric(g[col], errors="coerce").dropna()
        if len(v) < 20:
            continue
        if not _integer_valued(v) or v.nunique() / len(v) < 0.9:
            return False
        d = v.diff().dropna()
        if max((d > 0).mean(), (d < 0).mean()) < 0.95:
            return False
        seen += 1
    return seen > 0
