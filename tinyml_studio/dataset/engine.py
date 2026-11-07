"""Orquestador: ``inspect_files`` entiende uno o varios CSV; ``apply_recipe`` los reproduce."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tinyml_studio.i18n import tr
from tinyml_studio.profile import profile_dataframe

from . import quality
from .compat import compatibility
from .harmonize import ColumnGroup, groups_to_rename, harmonize, source_name, unify
from .infer import KindInfo, infer_kind, is_per_file_counter
from .load import LoadedTable, load_table
from .names import label_from_filename
from .report import summary, summary_es
from .roles import assign_roles, choose_task, suggest_split, target_candidates

RECIPE_VERSION = 1
SUFFIXES = {".csv", ".tsv", ".txt"}
LABEL_COLUMN = "label"


def expand_paths(paths: list[str | Path]) -> list[str]:
    out: list[str] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(str(f) for f in p.iterdir() if f.suffix.lower() in SUFFIXES)
        else:
            out.append(str(p))
    if not out:
        raise ValueError(tr("no se encontraron archivos CSV", "no CSV files found"))
    return out


def _kinds(df: pd.DataFrame) -> dict[str, KindInfo]:
    kinds = {c: infer_kind(df[c], c) for c in df.columns if c != "source"}
    for c, k in kinds.items():
        if k.kind in ("numeric_continuous", "numeric_discrete") and is_per_file_counter(df, c):
            kinds[c] = KindInfo(
                "id",
                tr(
                    "contador entero monótono dentro de cada archivo",
                    "integer counter that increases monotonically within each file",
                ),
            )
    return kinds


def _coverage(groups: list[ColumnGroup], n_files: int) -> dict[str, float]:
    return {g.canonical: len(g.members) / n_files for g in groups}


def inspect_files(paths: list[str | Path]) -> dict[str, Any]:
    files = expand_paths(paths)
    tables = [load_table(f) for f in files]
    groups = harmonize(tables)
    rename = groups_to_rename(groups)
    df = unify(tables, rename)
    n_files = len(tables)
    kinds = _kinds(df)
    cov = _coverage(groups, n_files)

    # ¿cada archivo es una clase? (solo si no hay objetivo claro)
    labels = {source_name(t.path): label_from_filename(t.path) for t in tables}
    cands = target_candidates(df, kinds, cov)
    task = choose_task(cands, kinds, n_files, len(set(labels.values())))
    # nulos medidos solo en los archivos donde la columna existe
    present: dict[str, pd.Series] = {}
    missing_in: dict[str, list[str]] = {}
    for g in groups:
        names = {source_name(p) for p in g.members}
        present[g.canonical] = df["source"].isin(names)
        missing_in[g.canonical] = sorted({source_name(t.path) for t in tables} - names)
    present_null = {
        c: float(df.loc[m, c].isna().mean()) if m.any() else 1.0 for c, m in present.items()
    }
    file_ids = frozenset(quality.source_identifiers(df, kinds))
    roles = assign_roles(
        df,
        {**kinds, "source": KindInfo("categorical", "")},
        task["target"],
        task["label_from_file"],
        n_files > 1,
        present_null,
        missing_in,
        file_ids,
    )
    split = suggest_split(roles, task, n_files > 1)

    prof = {c["name"]: c for c in profile_dataframe(df)["columns"]}
    by_canon = {g.canonical: g for g in groups}
    columns = []
    for c in df.columns:
        if c == "source":
            continue
        g = by_canon.get(c)
        columns.append(
            {
                "name": c,
                "kind": kinds[c].kind,
                "kind_reason": kinds[c].reason,
                "role": roles[c]["role"],
                "role_reason": roles[c]["reason"],
                "coverage": round(cov.get(c, 1.0), 3),
                "present_in": {source_name(p): n for p, n in g.members.items()} if g else {},
                "stats": {
                    k: v
                    for k, v in prof[c].items()
                    if k not in ("name", "dtype", "constant", "inf")
                },
                "dtype": prof[c]["dtype"],
            }
        )

    alerts: list[dict] = []
    for t in tables:
        for n in t.coercions:
            if n["action"] in ("numeric", "datetime", "boolean") or n["count"] > 0:
                alerts.append(
                    quality._a(
                        "info", "coercion", _coercion_msg(t, n), n["column"], source_name(t.path)
                    )
                )
    for g in groups:
        if len(g.members) < n_files:
            miss = missing_in[g.canonical]
            shown = ", ".join(miss[:3]) + (
                tr(" y {n} más", " and {n} more", n=len(miss) - 3) if len(miss) > 3 else ""
            )
            sev = "medium" if roles[g.canonical]["role"] in ("feature", "target") else "info"
            alerts.append(
                quality._a(
                    sev,
                    "missing_column",
                    tr(
                        "'{c}' falta en: {shown}",
                        "'{c}' is missing in: {shown}",
                        c=g.canonical,
                        shown=shown,
                    ),
                    g.canonical,
                    files=miss,
                )
            )
        elif g.score < 1.0:
            alerts.append(
                quality._a(
                    "info",
                    "renamed",
                    tr(
                        "'{c}' agrupa {m} ({r}); revisa el mapa",
                        "'{c}' groups {m} ({r}); review the mapping",
                        c=g.canonical,
                        m=sorted(set(g.members.values())),
                        r=g.reason,
                    ),
                    g.canonical,
                )
            )
    alerts += quality.check_nulls(df.drop(columns="source"), present)
    alerts += quality.check_absurd_values(df, kinds)
    alerts += quality.check_sentinels(df, kinds)
    alerts += quality.check_duplicates(df)
    if task["target"]:
        alerts += quality.check_target(df, task["target"], task["task"])
    alerts += quality.check_leakage(
        df, kinds, task["target"], task["task"], any(r["role"] == "time" for r in roles.values())
    )
    alerts += quality.check_drift(df, kinds)
    alerts += quality.check_file_mismatch(df, kinds)
    for c in sorted(file_ids):
        alerts.append(
            quality._a(
                "high" if task["label_from_file"] else "medium",
                "file_identifier",
                tr(
                    "'{c}' identifica el archivo de origen: ",
                    "'{c}' identifies the source file: ",
                    c=c,
                )
                + (
                    tr(
                        "se excluye de las features (la etiqueta sale del archivo)",
                        "it is excluded from the features (the label comes from the file)",
                    )
                    if task["label_from_file"]
                    else tr("revisa si filtra información", "check whether it leaks information")
                ),
                c,
            )
        )
    order = {"high": 0, "medium": 1, "info": 2}
    alerts.sort(key=lambda a: order[a["severity"]])

    from tinyml_studio.clean.suggest import suggest_cleaning

    cleaning = suggest_cleaning([c for c in columns if c["role"] != "meta"], alerts, n_files > 1)
    recipe = {
        "version": RECIPE_VERSION,
        "files": files,
        "rename": rename,
        "drop": [c for c, r in roles.items() if r["role"] == "drop" and c != "source"],
        "roles": {c: r["role"] for c, r in roles.items()},
        "task": task["task"],
        "target": task["target"],
        "label_from_file": {
            "enabled": task["label_from_file"],
            "column": LABEL_COLUMN,
            "mapping": labels,
        },
        "split": split,
        "encoding": {"default": "onehot", "overrides": {}, "max_categories": 20},
        "cleaning": cleaning,
    }
    report = {
        "version": RECIPE_VERSION,
        "files": [
            {
                "file": source_name(t.path),
                "rows": t.rows,
                "cols": int(t.df.shape[1]),
                "encoding": t.encoding,
                "separator": t.separator,
                "influx_annotation_rows": t.annotation_rows,
            }
            for t in tables
        ],
        "rows_total": int(len(df)),
        "columns": columns,
        "column_groups": [
            {
                "canonical": g.canonical,
                "members": {source_name(p): n for p, n in g.members.items()},
                "score": round(g.score, 2),
                "reason": g.reason,
            }
            for g in groups
            if len(g.members) > 1
        ],
        "compatibility": compatibility(files, groups),
        "target_candidates": cands[:5],
        "task": task,
        "split": split,
        "alerts": alerts,
        "recipe": recipe,
    }
    report["summary_es"] = summary_es(report)
    report["summary"] = summary(report)
    return report


def _coercion_msg(t: LoadedTable, n: dict) -> str:
    a = n["action"]
    f = source_name(t.path)
    if a == "numeric":
        extra = tr(" (coma decimal)", " (decimal comma)") if n.get("decimal_comma") else ""
        return tr(
            "'{c}' venía como texto y se convirtió a número{extra} en {f}",
            "'{c}' was read as text and converted to a number{extra} in {f}",
            c=n["column"],
            extra=extra,
            f=f,
        )
    if a == "datetime":
        return tr(
            "'{c}' se interpretó como fecha/hora en {f}",
            "'{c}' was interpreted as date/time in {f}",
            c=n["column"],
            f=f,
        )
    if a == "boolean":
        return tr(
            "'{c}' se interpretó como booleano en {f}",
            "'{c}' was interpreted as boolean in {f}",
            c=n["column"],
            f=f,
        )
    return tr(
        "{k} valores tipo 'NA' en '{c}' ({f}) se tomaron como nulos",
        "{k} 'NA'-like values in '{c}' ({f}) were treated as nulls",
        k=n["count"],
        c=n["column"],
        f=f,
    )


def apply_recipe(recipe: dict[str, Any]) -> pd.DataFrame:
    """Reproduce los datos organizados a partir de una receta (editable a mano)."""
    if recipe.get("version") != RECIPE_VERSION:
        raise ValueError(
            tr(
                "versión de receta no soportada: {v}",
                "unsupported recipe version: {v}",
                v=recipe.get("version"),
            )
        )
    from tinyml_studio.train.features import normalize_encoding

    normalize_encoding(recipe.get("encoding"))  # valida; ausente = onehot
    tables = [load_table(f) for f in recipe["files"]]
    df = unify(tables, recipe["rename"])
    lff = recipe.get("label_from_file", {})
    if lff.get("enabled"):
        df[lff["column"]] = df["source"].map(lff["mapping"])
        if df[lff["column"]].isna().any():
            raise ValueError(
                tr(
                    "label_from_file.mapping no cubre todos los archivos",
                    "label_from_file.mapping does not cover all files",
                )
            )
    roles = recipe.get("roles", {})
    drop = [c for c in recipe.get("drop", []) if c in df.columns]
    drop += [c for c, r in roles.items() if r == "drop" and c in df.columns and c not in drop]
    return df.drop(columns=drop)
