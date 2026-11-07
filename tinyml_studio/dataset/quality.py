"""Alertas de calidad y riesgo de fuga de datos. Cada alerta es un dict serializable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tinyml_studio.i18n import tr

from .infer import KindInfo

SENTINELS = (-999.0, -9999.0, -99999.0, 9999.0, 99999.0, 999999.0)


def _a(sev: str, code: str, msg: str, column: str | None = None, file: str | None = None, **d):
    return {"severity": sev, "code": code, "column": column, "file": file, "message": msg, **d}


def _f(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _numeric_cols(df, kinds):
    return [
        c
        for c in df.columns
        if kinds.get(c) and kinds[c].kind in ("numeric_continuous", "numeric_discrete")
    ]


def _nulls_msg(c: str, p: float) -> str:
    return tr("'{c}' tiene {p:.0%} de nulos", "'{c}' has {p:.0%} nulls", c=c, p=p)


def check_nulls(df: pd.DataFrame, present: dict[str, pd.Series] | None = None) -> list[dict]:
    """Nulos reales: se miden solo en los archivos donde la columna existe."""
    out = []
    for c in df.columns:
        rows = present[c] if present and c in present else slice(None)
        p = float(df.loc[rows, c].isna().mean()) if len(df.loc[rows]) else 0.0
        if p > 0.5:
            out.append(_a("high", "nulls", _nulls_msg(c, p), c, pct=round(p, 3)))
        elif p > 0.2:
            out.append(_a("medium", "nulls", _nulls_msg(c, p), c, pct=round(p, 3)))
    return out


def check_absurd_values(df: pd.DataFrame, kinds: dict[str, KindInfo]) -> list[dict]:
    """Valores muy lejos del rango típico (q01..q99 ampliado 20 veces), por archivo."""
    out = []
    for c in _numeric_cols(df, kinds):
        for src, g in df.groupby("source", sort=False):
            f = _f(g[c]).dropna()
            if len(f) < 30:
                continue
            q01, q99 = f.quantile(0.01), f.quantile(0.99)
            scale = q99 - q01
            if scale <= 0:
                continue
            bad = f[(f > q99 + 20 * scale) | (f < q01 - 20 * scale)]
            if 0 < len(bad) <= max(5, 0.01 * len(f)):
                worst = bad.iloc[int(np.argmax(np.abs(bad.to_numpy() - f.median())))]
                out.append(
                    _a(
                        "medium",
                        "absurd_values",
                        tr(
                            "{n} valor(es) absurdo(s) en '{c}' ({src}); p. ej. {worst:g} "
                            "cuando lo típico está entre {q01:g} y {q99:g}",
                            "{n} absurd value(s) in '{c}' ({src}); e.g. {worst:g} "
                            "when the typical range is {q01:g} to {q99:g}",
                            n=len(bad),
                            c=c,
                            src=src,
                            worst=worst,
                            q01=q01,
                            q99=q99,
                        ),
                        c,
                        src,
                        count=int(len(bad)),
                    )
                )
    return out


def check_sentinels(df: pd.DataFrame, kinds: dict[str, KindInfo]) -> list[dict]:
    out = []
    for c in _numeric_cols(df, kinds):
        f = _f(df[c])
        lo, hi = f.min(), f.max()
        for v in SENTINELS:
            cnt = int((f == v).sum())
            if cnt and v in (lo, hi) and cnt / max(f.notna().sum(), 1) < 0.5:
                out.append(
                    _a(
                        "high",
                        "sentinel",
                        tr(
                            "'{c}' contiene {cnt} veces {v:g}: Parece un código de dato faltante",
                            "'{c}' contains {v:g} {cnt} times: It looks like a missing-data code",
                            c=c,
                            cnt=cnt,
                            v=v,
                        ),
                        c,
                        count=cnt,
                        value=v,
                    )
                )
    return out


def check_duplicates(df: pd.DataFrame) -> list[dict]:
    out = []
    for src, g in df.groupby("source", sort=False):
        p = float(g.drop(columns="source").duplicated().mean()) if len(g) else 0.0
        if p > 0.01:
            out.append(
                _a(
                    "high" if p > 0.2 else "medium",
                    "duplicates",
                    tr(
                        "{p:.0%} de filas duplicadas en {src}",
                        "{p:.0%} duplicate rows in {src}",
                        p=p,
                        src=src,
                    ),
                    None,
                    src,
                    pct=round(p, 3),
                )
            )
    return out


def check_target(df: pd.DataFrame, target: str, task: str) -> list[dict]:
    out = []
    vc = df[target].value_counts()
    if task == "classification":
        if len(vc) > 50:
            out.append(
                _a(
                    "medium",
                    "many_classes",
                    tr(
                        "'{t}' tiene {n} clases: ¿es en realidad regresión?",
                        "'{t}' has {n} classes: Is it actually regression?",
                        t=target,
                        n=len(vc),
                    ),
                    target,
                )
            )
        if len(vc) >= 2:
            ratio = float(vc.max() / vc.min())
            if ratio > 3:
                out.append(
                    _a(
                        "high" if ratio > 10 else "medium",
                        "imbalance",
                        tr(
                            "desbalance de clases {ratio:.1f}:1 (más grande '{big}', "
                            "más pequeña '{small}')",
                            "class imbalance {ratio:.1f}:1 (largest '{big}', smallest '{small}')",
                            ratio=ratio,
                            big=vc.idxmax(),
                            small=vc.idxmin(),
                        ),
                        target,
                        ratio=round(ratio, 1),
                    )
                )
            for k, n in vc[vc < 10].items():
                out.append(
                    _a(
                        "high",
                        "tiny_class",
                        tr(
                            "la clase '{k}' tiene solo {n} fila(s)",
                            "class '{k}' has only {n} row(s)",
                            k=k,
                            n=n,
                        ),
                        target,
                        count=int(n),
                    )
                )
    return out


def check_leakage(
    df: pd.DataFrame,
    kinds: dict[str, KindInfo],
    target: str | None,
    task: str,
    time_sorted_risk: bool,
) -> list[dict]:
    out = []
    feats = [
        c
        for c in df.columns
        if c not in ("source", target)
        and kinds.get(c)
        and kinds[c].kind in ("numeric_continuous", "numeric_discrete")
    ]
    # 1) filas consecutivas casi iguales -> partir al azar filtra información
    flagged: list[str] = []
    for src, g in df.groupby("source", sort=False):
        if len(g) < 50:
            continue
        ac = []
        for c in feats[:12]:
            f = _f(g[c])
            if f.std() > 0:
                v = f.autocorr(lag=1)
                if pd.notna(v):
                    ac.append(abs(v))
        if ac and float(np.median(ac)) > 0.9:
            flagged.append(str(src))
    if flagged:
        more = tr(" y {n} más", " and {n} more", n=len(flagged) - 3) if len(flagged) > 3 else ""
        out.append(
            _a(
                "medium",
                "autocorrelation",
                tr(
                    "filas consecutivas muy parecidas (autocorrelación > 0.9) en "
                    "{files}{more}: No partas al azar; usa bloques temporales o por grupo",
                    "consecutive rows are very similar (autocorrelation > 0.9) in "
                    "{files}{more}: Do not split randomly; use temporal or group blocks",
                    files=", ".join(flagged[:3]),
                    more=more,
                ),
                None,
                files=flagged,
            )
        )
    # 2) objetivo ordenado en bloques
    if target is not None:
        for src, g in df.groupby("source", sort=False):
            t = g[target].dropna()
            if len(t) >= 50 and t.nunique() > 1:
                runs = int((t != t.shift()).sum())
                if runs / len(t) < 0.02:
                    out.append(
                        _a(
                            "medium",
                            "sorted_target",
                            tr(
                                "en {src} el objetivo viene ordenado en bloques "
                                "({runs} cambios en {n} filas): No barajes antes de partir",
                                "in {src} the target is sorted in blocks "
                                "({runs} changes in {n} rows): Do not shuffle before splitting",
                                src=src,
                                runs=runs,
                                n=len(t),
                            ),
                            target,
                            src,
                        )
                    )
    # 3) columnas casi deterministas del objetivo
    if target is not None and task == "classification":
        for c in df.columns:
            k = kinds.get(c)
            if (
                c in ("source", target)
                or not k
                or k.kind not in ("binary", "categorical", "numeric_discrete")
            ):
                continue
            if df[c].nunique() < 2 or df[c].nunique() > 0.5 * len(df):
                continue
            ct = df.groupby(c, observed=True)[target].agg(lambda s: s.value_counts().iloc[0])
            purity = float(ct.sum() / df[[c, target]].dropna().shape[0])
            base = float(df[target].value_counts(normalize=True).iloc[0])
            if purity >= 0.99 and purity - base > 0.05:
                out.append(
                    _a(
                        "high",
                        "target_leak",
                        tr(
                            "'{c}' casi determina '{t}' ({purity:.1%}): Posible fuga o "
                            "columna derivada del objetivo",
                            "'{c}' almost determines '{t}' ({purity:.1%}): Possible leakage "
                            "or a column derived from the target",
                            c=c,
                            t=target,
                            purity=purity,
                        ),
                        c,
                        purity=round(purity, 4),
                    )
                )
    if time_sorted_risk:
        out.append(
            _a(
                "info",
                "time_column",
                tr(
                    "hay columna de tiempo: Ordena por tiempo y parte por bloques contiguos",
                    "there is a time column: Sort by time and split into contiguous blocks",
                ),
            )
        )
    return out


def check_drift(df: pd.DataFrame, kinds: dict[str, KindInfo]) -> list[dict]:
    out = []
    if df["source"].nunique() < 2:
        return out
    for c in _numeric_cols(df, kinds):
        f = _f(df[c])
        scale = f.quantile(0.75) - f.quantile(0.25)
        if not scale or np.isnan(scale):
            scale = f.std()
        if not scale or np.isnan(scale):
            continue
        med = f.groupby(df["source"]).median().dropna()
        if len(med) >= 2 and (med.max() - med.min()) / scale > 3:
            out.append(
                _a(
                    "info",
                    "drift",
                    tr(
                        "'{c}' cambia mucho de un archivo a otro (medianas entre {lo:g} y {hi:g})",
                        "'{c}' changes a lot from one file to another (medians between "
                        "{lo:g} and {hi:g})",
                        c=c,
                        lo=med.min(),
                        hi=med.max(),
                    ),
                    c,
                )
            )
    return out


def check_file_mismatch(df: pd.DataFrame, kinds: dict[str, KindInfo]) -> list[dict]:
    """Archivos que parecen medidos de forma distinta: escala muy diferente o signo opuesto.

    Típico de capturas con otra calibración o con otra convención de signo (p. ej. potencia
    reactiva). Mezclar esos archivos hace que el modelo aprenda "cómo se grabó" y no "qué es".
    """
    out: list[dict] = []
    if df["source"].nunique() < 2:
        return out
    for c in _numeric_cols(df, kinds):
        # archivo -> (mediana, magnitud, fracción de valores positivos)
        per: dict[str, tuple[float, float, float]] = {}
        for src, g in df.groupby("source", sort=False):
            f = _f(g[c]).dropna()
            if len(f) < 30 or f.nunique() < 3:
                continue
            q01, q99, med = f.quantile(0.01), f.quantile(0.99), f.median()
            mag = float(max(abs(q01), abs(q99), abs(med)))
            nz = f[f.abs() > 0.02 * max(mag, 1e-12)]
            pos = float((nz > 0).mean()) if len(nz) >= 10 else 0.5
            per[str(src)] = (float(med), mag, pos)
        if len(per) < 2:
            continue
        mags = {k: v[1] for k, v in per.items() if v[1] > 0}
        if len(mags) >= 2:
            hi_f, lo_f = max(mags, key=mags.get), min(mags, key=mags.get)
            ratio = mags[hi_f] / mags[lo_f]
            if ratio >= 20:
                out.append(
                    _a(
                        "high",
                        "file_mismatch",
                        tr(
                            "'{c}' tiene una escala muy distinta entre archivos: Unas {r:.0f} "
                            "veces mayor en '{a}' (típico {va:g}) que en '{b}' (típico {vb:g}). "
                            "Puede ser otra calibración u otras unidades: No mezcles estos "
                            "archivos sin revisarlo.",
                            "'{c}' has a very different scale between files: About {r:.0f} "
                            "times larger in '{a}' (typical {va:g}) than in '{b}' (typical "
                            "{vb:g}). It may be another calibration or other units: Do not mix "
                            "these files without checking.",
                            c=c,
                            r=ratio,
                            a=hi_f,
                            b=lo_f,
                            va=mags[hi_f],
                            vb=mags[lo_f],
                        ),
                        c,
                        files=[hi_f, lo_f],
                        kind="scale",
                    )
                )
        plus = [k for k, v in per.items() if v[2] >= 0.9 and abs(v[0]) > 0]
        minus = [k for k, v in per.items() if v[2] <= 0.1 and abs(v[0]) > 0]
        if plus and minus:
            out.append(
                _a(
                    "high",
                    "file_mismatch",
                    tr(
                        "'{c}' tiene el signo opuesto entre archivos: Positiva en '{a}' y "
                        "negativa en '{b}'. Puede ser otra convención de signo: No mezcles "
                        "estos archivos sin revisarlo.",
                        "'{c}' has the opposite sign between files: Positive in '{a}' and "
                        "negative in '{b}'. It may be another sign convention: Do not mix "
                        "these files without checking.",
                        c=c,
                        a=plus[0],
                        b=minus[0],
                    ),
                    c,
                    files=[plus[0], minus[0]],
                    kind="sign",
                )
            )
    return out


def source_identifiers(df: pd.DataFrame, kinds: dict[str, KindInfo]) -> list[str]:
    """Columnas no continuas cuyo valor casi determina el archivo de origen (>= 3 archivos).

    Si la etiqueta sale del archivo, estas columnas filtran la respuesta (p. ej. un
    ``fabricante`` que solo aparece en un archivo).
    """
    if df["source"].nunique() < 3:
        return []
    out = []
    for c, k in kinds.items():
        if k.kind not in ("binary", "categorical", "numeric_discrete"):
            continue
        sub = df[[c, "source"]].dropna()
        if len(sub) < 30 or sub[c].nunique() < 2:
            continue
        top = sub.groupby(c, observed=True)["source"].agg(lambda s: s.value_counts().iloc[0])
        if top.sum() / len(sub) >= 0.99 and sub["source"].nunique() >= 2:
            out.append(c)
    return out
