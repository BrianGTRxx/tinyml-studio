"""Validación por grupos (sesiones/archivos): ¿el modelo funciona con una sesión que NO vio?

La partición habitual deja el test dentro de las mismas sesiones que el entrenamiento, así que
mide "más tarde en la misma grabación". Aquí se deja fuera una sesión (archivo) completa, se
entrena con las demás y se mide en la que quedó fuera. Solo se puede medir una clase si existe
en al menos DOS sesiones: con una sola, al dejarla fuera la clase desaparece del entrenamiento.
Eso se dice sin rodeos en vez de dar una cifra inventada.

El preprocesado (limpieza + codificación) se ajusta en cada pliegue SOLO con las sesiones de
entrenamiento.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tinyml_studio.clean.prepare import cleaning_without_target, resolve_target
from tinyml_studio.clean.steps import Pipeline
from tinyml_studio.dataset import apply_recipe
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit

from . import models as M
from .features import FeatureEncoder
from .run import feature_columns

MAX_TRAIN_ROWS = 40_000  # tope por pliegue: la validación debe ser rápida
UNSTABLE_SPREAD = (
    0.1  # si según la semilla la exactitud de una sesión varía >= 10 puntos: inestable
)
DUPLICATE_OVERLAP = 0.5  # una sesión con >= 50 % de filas idénticas a otras se considera copia


def session_columns(df: pd.DataFrame, group: str, candidates: list[str]) -> list[str]:
    """Columnas discretas con UN valor en (casi) cada sesión y valores distintos entre sesiones.

    Describen la sesión de captura (equipo, lugar, canal), no el fenómeno: el modelo puede aprender
    "qué sesión es" en lugar de "qué aparato es". Solo se evalúan columnas no continuas.
    """
    if group not in df.columns or df[group].nunique() < 2:
        return []
    out = []
    for c in candidates:
        if c not in df.columns or c == group:
            continue
        s = df[[c, group]].dropna()
        if len(s) < 30:
            continue
        if pd.api.types.is_float_dtype(s[c]) and s[c].nunique() > 20:
            continue  # continua: la separa la validación por grupos, no un criterio de nombre
        per = s.groupby(group, observed=True)[c].nunique()
        constant = float((per == 1).mean())
        values = s.groupby(group, observed=True)[c].agg(lambda x: x.mode().iloc[0])
        if constant >= 0.9 and values.nunique() >= 2:
            out.append(c)
    return out


def _fold(
    df: pd.DataFrame,
    group: str,
    left_out: str,
    recipe: dict,
    feats: list[str],
    target: str,
    model: str,
    seeds: tuple[int, ...],
    params: dict | None,
) -> dict[str, Any] | None:
    tr_df = df[df[group] != left_out].reset_index(drop=True)
    te_df = df[df[group] == left_out].reset_index(drop=True)
    seen = set(tr_df[target].astype(str))
    te_df = te_df[te_df[target].astype(str).isin(seen)].reset_index(drop=True)
    if te_df.empty:
        return None  # sus clases no existen en ninguna otra sesión: no medible
    # ¿la sesión dejada fuera es (casi) una copia de otra? Entonces no mide generalización
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    overlap = 0.0
    if num:
        seen_rows = set(pd.util.hash_pandas_object(tr_df[num].round(6), index=False))
        hashes = pd.util.hash_pandas_object(te_df[num].round(6), index=False)
        overlap = float(hashes.isin(seen_rows).mean())
    accs = []
    for s in seeds:  # varias semillas: un solo árbol puede ser muy inestable entre sesiones
        tr_s = tr_df
        if len(tr_s) > MAX_TRAIN_ROWS:
            tr_s = tr_s.sample(MAX_TRAIN_ROWS, random_state=s).reset_index(drop=True)
        pipe = Pipeline.from_list(cleaning_without_target(recipe))
        pipe.fit(tr_s)
        trc = pipe.transform(tr_s, is_train=True)
        tec = pipe.transform(te_df, is_train=False)
        enc = FeatureEncoder.fit(trc, feats, target, "classification", recipe.get("encoding"))
        Xtr, ytr = enc.transform(trc), enc.encode_target(trc[target])
        Xte, yte = enc.transform(tec), enc.encode_target(tec[target])
        est, _eff = M.build(model, "classification", s, params)
        est.fit(Xtr, ytr)
        accs.append(float((est.predict(Xte) == yte).mean()))
    return {
        "group": str(left_out),
        "classes": sorted({str(c) for c in te_df[target]}),
        "rows": int(len(te_df)),
        "accuracy": round(float(np.mean(accs)), 4),
        "accuracy_min": round(min(accs), 4),
        "accuracy_max": round(max(accs), 4),
        "overlap": round(overlap, 3),  # fracción de sus filas que ya estaban en el entrenamiento
    }


def _run(df, group, recipe, feats, target, model, seed, params, label, n_seeds=3):
    groups = sorted(df[group].astype(str).unique())
    folds, skipped = [], []
    for i, g in enumerate(groups):
        emit(
            "group_fold",
            10 + 80 * i / max(1, len(groups)),
            tr("{label}: Dejando fuera «{g}»…", "{label}: Leaving out «{g}»…", label=label, g=g),
        )
        f = _fold(
            df.assign(**{group: df[group].astype(str)}),
            group,
            g,
            recipe,
            feats,
            target,
            model,
            tuple(seed + k for k in range(n_seeds)),
            params,
        )
        (folds if f else skipped).append(f or g)
    clean = [f for f in folds if f["overlap"] < DUPLICATE_OVERLAP]
    contaminated = [f["group"] for f in folds if f["overlap"] >= DUPLICATE_OVERLAP]
    accs = [f["accuracy"] for f in clean]
    # medible = la clase salió en algún pliegue cuya sesión NO es una copia de otra
    measurable = sorted({c for f in clean for c in f["classes"]})
    all_classes = sorted(df[target].astype(str).unique())
    return {
        "folds": folds,
        "skipped_groups": skipped,
        "duplicate_groups": contaminated,
        "unstable_groups": [
            f["group"] for f in clean if f["accuracy_max"] - f["accuracy_min"] >= UNSTABLE_SPREAD
        ],
        "mean_accuracy": round(float(np.mean(accs)), 4) if accs else None,
        "min_accuracy": round(float(np.min(accs)), 4) if accs else None,
        "classes_measurable": measurable,
        "classes_not_measurable": sorted(set(all_classes) - set(measurable)),
    }


def group_validation(
    recipe: dict[str, Any],
    *,
    model: str = "random_forest",
    group: str | None = None,
    seed: int = 42,
    params: dict | None = None,
    exclude: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validación dejando una sesión fuera. Solo clasificación."""
    if recipe.get("task") != "classification":
        raise ValueError(
            tr(
                "la validación por sesiones solo está disponible para clasificación",
                "validation by sessions is only available for classification",
            )
        )
    target = resolve_target(recipe)
    if target is None:
        raise ValueError(tr("la receta no tiene objetivo", "the recipe has no target"))
    group = group or (recipe.get("split") or {}).get("group") or "source"
    emit("prepare", 4, tr("Preparando datos…", "Preparing data…"))
    if (recipe.get("roles") or {}).get(group) == "drop":  # se necesita aunque no sea variable
        recipe = {**recipe, "roles": {**recipe["roles"], group: "meta"}}
    df = apply_recipe(recipe)
    df = df[df[target].notna()].reset_index(drop=True)
    if group not in df.columns:
        raise ValueError(
            tr(
                "no existe la columna de sesión «{g}»",
                "the session column «{g}» does not exist",
                g=group,
            )
        )
    feats = feature_columns(recipe, df, exclude)
    if df[group].nunique() < 2:  # una sola sesión: no hay con qué comparar
        return {
            "model": model,
            "group_column": group,
            "n_groups": int(df[group].nunique()),
            "n_classes": int(df[target].nunique()),
            "session_columns": [],
            "with_all_features": {
                "folds": [],
                "skipped_groups": [],
                "duplicate_groups": [],
                "mean_accuracy": None,
                "min_accuracy": None,
                "classes_measurable": [],
                "classes_not_measurable": sorted(df[target].astype(str).unique()),
            },
            "without_session_columns": None,
            "verdict": "not_measurable",
            "warnings": [
                tr(
                    "Hay una sola sesión (archivo): No se puede medir con una sesión nueva. "
                    "Carga varias sesiones de cada clase.",
                    "There is only one session (file): How it works on a new session cannot be "
                    "measured. Load several sessions of each class.",
                )
            ],
        }
    sess_cols = session_columns(df, group, feats)
    full = _run(df, group, recipe, feats, target, model, seed, params, tr("con todo", "with all"))
    out: dict[str, Any] = {
        "model": model,
        "group_column": group,
        "n_groups": int(df[group].nunique()),
        "n_classes": int(df[target].nunique()),
        "session_columns": sess_cols,
        "with_all_features": full,
        "without_session_columns": None,
        "warnings": [],
    }
    if sess_cols:
        feats2 = [c for c in feats if c not in sess_cols]
        if feats2:
            out["without_session_columns"] = _run(
                df,
                group,
                recipe,
                feats2,
                target,
                model,
                seed,
                params,
                tr("sin sesión", "without session"),
            )
    n_meas, n_all = len(full["classes_measurable"]), out["n_classes"]
    if n_meas == 0:
        out["verdict"] = "not_measurable"
        out["warnings"].append(
            tr(
                "Ninguna clase aparece en 2 sesiones o más: NO se puede medir el modelo "
                "con una sesión nueva. Captura cada aparato en varias sesiones distintas.",
                "No class appears in 2 or more sessions: How the model behaves on a NEW session "
                "CANNOT be measured. Capture each appliance in several different sessions.",
            )
        )
    elif n_meas < n_all:
        out["verdict"] = "partial"
        out["warnings"].append(
            tr(
                "Solo {m} de {n} clases tienen 2 sesiones o más: La cifra vale para ellas; "
                "las demás no se pueden medir con los datos actuales.",
                "Only {m} of {n} classes appear in 2 or more sessions: The figure holds for those "
                "classes; the rest cannot be measured with the current data.",
                m=n_meas,
                n=n_all,
            )
        )
    else:
        out["verdict"] = "measurable"
    if full["unstable_groups"]:
        out["warnings"].append(
            tr(
                "Resultado INESTABLE en {g}: Según la semilla la exactitud cambia 10 puntos o más. "
                "El modelo usa algo que cambia entre sesiones (p. ej. una columna de sesión); "
                "compara con la ejecución sin las columnas de sesión.",
                "UNSTABLE result on {g}: With other seeds the accuracy changes by 10 points or "
                "more. The model relies on something that varies between sessions (usually a "
                "session column or a calibration value); compare with the run without the session "
                "columns.",
                g=", ".join(full["unstable_groups"]),
            )
        )
    if full["duplicate_groups"]:
        out["warnings"].append(
            tr(
                "Sesiones (casi) copias de otra: {g}. Su cifra NO mide generalización y no "
                "cuenta en el resultado; revisa que no subiste el mismo archivo dos veces.",
                "Sessions that are (almost) copies of another: {g}. Their figure does NOT measure "
                "generalization and is left out of the result; check that you did not upload the "
                "same file twice.",
                g=", ".join(full["duplicate_groups"]),
            )
        )
    if sess_cols:
        out["warnings"].append(
            tr(
                "Columnas que describen la sesión (un valor por sesión): {c}. El modelo puede "
                "aprender la sesión en vez del fenómeno; considera marcarlas como metadato.",
                "Columns that describe the session (one value per session): {c}. The model may "
                "learn the session instead of the phenomenon; consider marking them as metadata.",
                c=", ".join(sess_cols),
            )
        )
    return out


def summary(res: dict[str, Any]) -> str:
    """Resumen de texto de la validación por sesiones."""
    full = res["with_all_features"]
    lines = [
        tr(
            "Validación dejando una sesión fuera ({m}; {g} sesiones, {c} clases)",
            "Leave-one-session-out validation ({m}; {g} sessions, {c} classes)",
            m=res["model"],
            g=res["n_groups"],
            c=res["n_classes"],
        )
    ]
    if full["mean_accuracy"] is None:
        lines.append(
            tr(
                "  Exactitud en sesión nueva: No medible",
                "  Accuracy on a new session: Not measurable",
            )
        )
    else:
        lines.append(
            tr(
                "  Exactitud en una sesión que no vio: Media {a:.1%} (mínima {b:.1%})",
                "  Accuracy on an unseen session: Mean {a:.1%} (minimum {b:.1%})",
                a=full["mean_accuracy"],
                b=full["min_accuracy"],
            )
        )
    for f in full["folds"]:
        dup = tr(" [copia]", " [copy]") if f["overlap"] >= DUPLICATE_OVERLAP else ""
        lines.append(f"    {f['group']}: {f['accuracy']:.1%} ({', '.join(f['classes'])}){dup}")
    if full["skipped_groups"]:
        lines.append(
            tr(
                "  Sin medir (su clase solo está en esa sesión): ",
                "  Not measured (its class is only in that session): ",
            )
            + ", ".join(full["skipped_groups"])
        )
    alt = res.get("without_session_columns")
    if alt and alt["mean_accuracy"] is not None:
        lines.append(
            tr(
                "  Sin las columnas de sesión ({c}): Media {a:.1%}",
                "  Without the session columns ({c}): Mean {a:.1%}",
                c=", ".join(res["session_columns"]),
                a=alt["mean_accuracy"],
            )
        )
    lines += [f"  ! {w}" for w in res["warnings"]]
    return "\n".join(lines)
