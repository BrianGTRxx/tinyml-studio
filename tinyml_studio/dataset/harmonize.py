"""Armonización de columnas entre varios archivos: ¿qué columnas son "la misma"?

Combina nombre normalizado, similitud de texto, inclusión de tokens, compatibilidad de
tipo y solapamiento de rangos. Solo SUGIERE; el usuario puede editar el mapa (receta).
"""

from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tinyml_studio.i18n import tr

from .load import LoadedTable
from .names import name_tokens, normalize_name

FUZZY_MIN = 0.8


@dataclass
class ColumnGroup:
    canonical: str
    members: dict[str, str] = field(default_factory=dict)  # ruta de archivo -> nombre original
    score: float = 1.0
    reason: str = field(
        default_factory=lambda: tr(
            "idéntico tras normalizar el nombre", "identical after normalizing the name"
        )
    )


@dataclass
class _ColMeta:
    table: str
    orig: str
    norm: str
    numeric: bool
    datetime: bool
    lo: float | None
    hi: float | None


def _meta(t: LoadedTable) -> list[_ColMeta]:
    out = []
    for c in t.df.columns:
        s = t.df[c]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        lo = hi = None
        if numeric and s.notna().sum() >= 5:
            f = pd.to_numeric(s, errors="coerce").astype("float64")
            lo, hi = float(f.quantile(0.01)), float(f.quantile(0.99))
        out.append(
            _ColMeta(
                t.path,
                str(c),
                normalize_name(c),
                numeric,
                pd.api.types.is_datetime64_any_dtype(s),
                lo,
                hi,
            )
        )
    return out


def _range_compatible(a: _ColMeta, b: _ColMeta) -> bool:
    if None in (a.lo, a.hi, b.lo, b.hi):
        return True
    span = max(a.hi - a.lo, b.hi - b.lo)
    if span == 0:
        return True
    gap = max(a.lo, b.lo) - min(a.hi, b.hi)  # >0 si los rangos no se solapan
    return gap <= 5 * span


def _score(a: _ColMeta, b: _ColMeta) -> tuple[float, str]:
    if a.numeric != b.numeric or a.datetime != b.datetime:
        return 0.0, ""
    if a.norm == b.norm:
        # mismo nombre = misma columna, aunque los rangos difieran (p. ej. cada archivo es un
        # aparato distinto con otro nivel de potencia): el rango solo desempata nombres parecidos
        return 1.0, tr("idéntico tras normalizar el nombre", "identical after normalizing the name")
    if not _range_compatible(a, b):
        return 0.0, ""
    ta, tb = name_tokens(a.orig), name_tokens(b.orig)
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if small and small < big and all(len(t) >= 3 for t in small):
        return 0.8 + 0.1 * len(small) / len(big), tr(
            "un nombre contiene al otro", "one name contains the other"
        )
    r = difflib.SequenceMatcher(None, a.norm, b.norm).ratio()
    return (
        (r, tr("nombres parecidos ({r:.2f})", "similar names ({r:.2f})", r=r))
        if r >= FUZZY_MIN
        else (0.0, "")
    )


def harmonize(tables: list[LoadedTable]) -> list[ColumnGroup]:
    groups: list[tuple[ColumnGroup, list[_ColMeta]]] = []
    for t in tables:
        for m in _meta(t):
            best, best_s, best_r = None, 0.0, ""
            for g, metas in groups:
                if t.path in g.members:
                    continue
                s, r = max((_score(m, o) for o in metas), key=lambda x: x[0])
                if s > best_s:
                    best, best_s, best_r = (g, metas), s, r
            if best is None or best_s == 0.0:
                groups.append((ColumnGroup(m.norm, {t.path: m.orig}), [m]))
            else:
                g, metas = best
                g.members[t.path] = m.orig
                metas.append(m)
                if best_s < g.score or g.score == 1.0 and best_s < 1.0:
                    g.score, g.reason = best_s, best_r
    out = [g for g, _ in groups]
    # nombre canónico: el más frecuente entre los miembros (empate: el más corto)
    used: set[str] = set()
    for g in out:
        norms = Counter(normalize_name(n) for n in g.members.values())
        g.canonical = sorted(norms, key=lambda k: (-norms[k], len(k), k))[0]
        base, i = g.canonical, 2
        while g.canonical in used:
            g.canonical, i = f"{base}_{i}", i + 1
        used.add(g.canonical)
    return out


def groups_to_rename(groups: list[ColumnGroup]) -> dict[str, dict[str, str]]:
    rename: dict[str, dict[str, str]] = {}
    for g in groups:
        for path, orig in g.members.items():
            rename.setdefault(path, {})[orig] = g.canonical
    return rename


def source_name(path: str) -> str:
    return Path(path).name


def unify(
    tables: list[LoadedTable], rename: dict[str, dict[str, str]], source_column: str = "source"
) -> pd.DataFrame:
    """Une las tablas con nombres canónicos y añade la columna ``source`` (nombre del archivo)."""
    frames = []
    for t in tables:
        mapping = rename.get(t.path, {})
        df = t.df.rename(columns=lambda c, m=mapping: m.get(str(c), normalize_name(c)))
        if df.columns.duplicated().any():
            raise ValueError(
                tr(
                    "{f}: El mapa de columnas produce nombres repetidos",
                    "{f}: The column mapping produces duplicate names",
                    f=source_name(t.path),
                )
            )
        df.insert(0, source_column, source_name(t.path))
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)
