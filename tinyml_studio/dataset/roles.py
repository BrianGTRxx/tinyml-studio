"""Sugerencia de roles de columna, tarea y estrategia de partición (todo editable)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tinyml_studio.i18n import tr

from .infer import KindInfo
from .names import name_tokens

TARGET_STRONG = {
    "target",
    "label",
    "labels",
    "class",
    "clase",
    "etiqueta",
    "y",
    "output",
    "outcome",
    "diagnosis",
    "diagnostico",
    "species",
    "especie",
    "survived",
    "churn",
    "fraud",
    "resultado",
    "result",
    "actividad",
    "activity",
    "categoria",
    "category",
}
TARGET_WEAK = {"type", "tipo", "status", "estado", "quality", "calidad", "default", "grade"}
TARGET_REGRESSION = {"price", "precio", "saleprice", "medv", "salary", "salario", "cost", "costo"}
GROUP_TOKENS = {
    "session",
    "sesion",
    "subject",
    "sujeto",
    "patient",
    "paciente",
    "user",
    "usuario",
    "device",
    "dispositivo",
    "group",
    "grupo",
    "episode",
    "episodio",
    "run",
    "block",
    "bloque",
}
# Nombres típicos de telemetría del propio dispositivo/transporte (no del fenómeno medido):
# suelen "identificar la sesión" y filtrar información al modelo. Solo por nombre; editable.
TELEMETRY_TOKENS = {
    "uptime",
    "rssi",
    "wifi",
    "ssid",
    "mac",
    "ip",
    "hostname",
    "firmware",
    "heap",
    "ram",
    "memoria",
    "memory",
    "bateria",
    "battery",
    "secuencia",
    "sequence",
    "seq",
    "counter",
    "contador",
    "timestamp",
}
MAX_CLASSES = 30
TARGET_THRESHOLD = 0.5


def target_candidates(
    df: pd.DataFrame, kinds: dict[str, KindInfo], coverage: dict[str, float]
) -> list[dict[str, Any]]:
    cols = [c for c in df.columns if c != "source"]
    out = []
    for i, c in enumerate(cols):
        k = kinds[c].kind
        if k in ("empty", "constant", "id", "datetime", "text"):
            continue
        toks = name_tokens(c)
        score, why = 0.0, []
        if toks & TARGET_STRONG:
            score += 0.5
            why.append(tr("el nombre sugiere una etiqueta", "the name suggests a label"))
        elif toks & TARGET_WEAK:
            score += 0.3
            why.append(tr("el nombre podría ser una etiqueta", "the name could be a label"))
        elif toks & TARGET_REGRESSION:
            score += 0.5
            why.append(
                tr("el nombre sugiere un valor a predecir", "the name suggests a value to predict")
            )
        nun = int(df[c].nunique(dropna=True))
        if k in ("binary", "categorical", "numeric_discrete") and 2 <= nun <= MAX_CLASSES:
            score += 0.3
            why.append(tr("{n} clases", "{n} classes", n=nun))
        elif k == "numeric_continuous":
            score += 0.1
        if i == len(cols) - 1:
            score += 0.15
            why.append(tr("es la última columna", "it is the last column"))
        if df[c].isna().mean() > 0.2:
            score -= 0.3
            why.append(tr("muchos nulos", "many nulls"))
        if coverage.get(c, 1.0) < 1.0:
            score -= 0.2
            why.append(tr("no está en todos los archivos", "it is not in all files"))
        if score > 0:
            out.append({"column": c, "score": round(score, 2), "reason": "; ".join(why)})
    return sorted(out, key=lambda d: -d["score"])


def choose_task(
    cands: list[dict], kinds: dict[str, KindInfo], n_files: int, label_candidates: int
) -> dict[str, Any]:
    if cands and cands[0]["score"] >= TARGET_THRESHOLD:
        c = cands[0]["column"]
        k = kinds[c].kind
        task = "regression" if k == "numeric_continuous" else "classification"
        return {
            "task": task,
            "target": c,
            "label_from_file": False,
            "reason": tr(
                "columna objetivo probable: {c} ({r})",
                "likely target column: {c} ({r})",
                c=c,
                r=cands[0]["reason"],
            ),
            "needs_user_confirmation": cands[0]["score"] < 0.8,
        }
    if n_files >= 2 and label_candidates >= 2:
        return {
            "task": "classification",
            "target": None,
            "label_from_file": True,
            "reason": tr(
                "no hay columna objetivo clara; cada archivo parece ser una clase "
                "(la etiqueta saldría del nombre del archivo)",
                "there is no clear target column; each file seems to be a class "
                "(the label would come from the file name)",
            ),
            "needs_user_confirmation": True,
        }
    return {
        "task": "unsupervised",
        "target": None,
        "label_from_file": False,
        "reason": tr(
            "no hay una columna objetivo clara: Elige una, o usa detección de anomalías",
            "there is no clear target column: Pick one, or use anomaly detection",
        ),
        "needs_user_confirmation": True,
    }


def assign_roles(
    df: pd.DataFrame,
    kinds: dict[str, KindInfo],
    target: str | None,
    label_from_file: bool,
    multi_file: bool,
    present_null: dict[str, float],
    missing_in: dict[str, list[str]],
    file_identifiers: frozenset[str] = frozenset(),
) -> dict[str, dict[str, str]]:
    roles: dict[str, dict[str, str]] = {}
    for c in df.columns:
        k = kinds[c].kind if c in kinds else "categorical"
        if c == "source":
            if label_from_file:
                roles[c] = {
                    "role": "group",
                    "reason": tr(
                        "cada archivo es una clase; sirve para particionar sin fuga",
                        "each file is a class; it is used to split without leakage",
                    ),
                }
            else:
                why = tr(
                    "archivo de origen (informativo, no es feature)",
                    "source file (informational, not a feature)",
                )
                roles[c] = {
                    "role": "meta" if multi_file else "drop",
                    "reason": why if multi_file else tr("un solo archivo", "a single file"),
                }
        elif c == target:
            roles[c] = {
                "role": "target",
                "reason": tr("columna objetivo sugerida", "suggested target column"),
            }
        elif k in ("empty", "constant"):
            roles[c] = {"role": "drop", "reason": kinds[c].reason}
        elif k == "id":
            roles[c] = {"role": "drop", "reason": kinds[c].reason}
        elif k == "datetime":
            roles[c] = {
                "role": "time",
                "reason": tr(
                    "fecha/hora: Ordena y parte por bloques",
                    "date/time: Sort and split into blocks",
                ),
            }
        elif k == "text":
            roles[c] = {
                "role": "drop",
                "reason": tr(
                    "texto libre (aún no soportado como feature)",
                    "free text (not yet supported as a feature)",
                ),
            }
        elif present_null.get(c, 0.0) > 0.6:
            roles[c] = {
                "role": "drop",
                "reason": tr("{p:.0%} de nulos", "{p:.0%} nulls", p=present_null[c]),
            }
        elif label_from_file and c in file_identifiers:
            roles[c] = {
                "role": "meta",
                "reason": tr(
                    "su valor identifica el archivo de origen y la "
                    "etiqueta sale del archivo: Sería una fuga",
                    "its value identifies the source file and the "
                    "label comes from the file: It would be leakage",
                ),
            }
        elif name_tokens(c) & TELEMETRY_TOKENS:
            hit = sorted(name_tokens(c) & TELEMETRY_TOKENS)[0]
            roles[c] = {
                "role": "meta",
                "reason": tr(
                    "el nombre ('{hit}') sugiere telemetría del "
                    "dispositivo: Puede identificar la sesión y filtrar información; "
                    "cámbialo a feature si es una medida real",
                    "the name ('{hit}') suggests device telemetry: "
                    "it may identify the session and leak information; "
                    "change it to a feature if it is a real measurement",
                    hit=hit,
                ),
            }
        elif (
            k in ("categorical", "numeric_discrete")
            and name_tokens(c) & GROUP_TOKENS
            and 2 <= df[c].nunique() <= 50
        ):
            roles[c] = {
                "role": "group",
                "reason": tr(
                    "el nombre sugiere sesión/sujeto/dispositivo",
                    "the name suggests session/subject/device",
                ),
            }
        else:
            extra = tr(" (hay que codificarla)", " (must be encoded)") if k == "categorical" else ""
            if missing_in.get(c):
                m = missing_in[c]
                extra += tr(
                    "; falta en {n} archivo(s): Imputar o usar solo los que la tienen",
                    "; missing in {n} file(s): Impute or use only those that have it",
                    n=len(m),
                )
            roles[c] = {"role": "feature", "reason": f"{kinds[c].kind}{extra}"}
    return roles


def suggest_split(roles: dict[str, dict[str, str]], task: dict, multi_file: bool) -> dict[str, Any]:
    time_col = next((c for c, r in roles.items() if r["role"] == "time"), None)
    group_col = next((c for c, r in roles.items() if r["role"] == "group"), None)
    ratios = [0.8, 0.1, 0.1]
    if task.get("label_from_file") and not time_col:
        # cada archivo es una clase: partir por archivo dejaría clases enteras fuera del train
        return {
            "strategy": "stratified",
            "group_column": None,
            "time_column": None,
            "ratios": ratios,
            "reason": tr(
                "cada archivo es una clase y no hay tiempo: Estratificada por etiqueta "
                "(partir por archivo dejaría clases fuera). Ojo: Filas consecutivas parecidas "
                "pueden filtrar información; si hay orden temporal usa bloques",
                "each file is a class and there is no time: Stratified by label "
                "(splitting by file would leave classes out). Note: Similar consecutive rows "
                "may leak information; if there is a temporal order, use blocks",
            ),
        }
    if time_col and group_col:
        return {
            "strategy": "temporal_blocks",
            "group_column": group_col,
            "time_column": time_col,
            "ratios": ratios,
            "reason": tr(
                "hay tiempo y grupo: Bloques temporales por grupo",
                "there is time and a group: Temporal blocks per group",
            ),
        }
    if group_col:
        return {
            "strategy": "group",
            "group_column": group_col,
            "time_column": None,
            "ratios": ratios,
            "reason": tr(
                "particionar por grupo evita fuga entre filas parecidas",
                "splitting by group avoids leakage between similar rows",
            ),
        }
    if time_col:
        return {
            "strategy": "temporal_blocks",
            "group_column": None,
            "time_column": time_col,
            "ratios": ratios,
            "reason": tr(
                "hay columna de tiempo: Bloques temporales",
                "there is a time column: Temporal blocks",
            ),
        }
    if task["task"] == "classification":
        return {
            "strategy": "stratified",
            "group_column": None,
            "time_column": None,
            "ratios": ratios,
            "reason": tr(
                "clasificación sin tiempo ni grupo: Estratificada",
                "classification without time or group: Stratified",
            ),
        }
    return {
        "strategy": "random",
        "group_column": None,
        "time_column": None,
        "ratios": ratios,
        "reason": tr("sin tiempo ni grupo: Aleatoria", "no time or group: Random"),
    }
