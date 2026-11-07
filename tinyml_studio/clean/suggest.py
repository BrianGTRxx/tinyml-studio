"""Limpieza guiada por las alertas del motor de datasets: alerta -> paso concreto + motivo."""

from __future__ import annotations

from tinyml_studio.i18n import tr

NUMERIC = {"numeric_continuous", "numeric_discrete"}


def suggest_cleaning(columns: list[dict], alerts: list[dict], multi_file: bool) -> list[dict]:
    """Devuelve pasos ``{"kind","params","why"}`` (sin ajustar), en orden de ejecución.

    Orden: centinelas -> absurdos -> duplicados -> imputación -> escalado. El usuario puede
    quitar, cambiar o añadir pasos en la receta; nada se aplica sin que quede registrado.
    """
    feats = {c["name"]: c for c in columns if c["role"] == "feature"}
    num_feats = [n for n, c in feats.items() if c["kind"] in NUMERIC]
    steps: list[dict] = []
    touched: set[str] = set()

    sentinel_cols: dict[float, list[str]] = {}
    for a in alerts:
        if a["code"] == "sentinel" and a["column"] in feats:
            sentinel_cols.setdefault(a["value"], []).append(a["column"])
    for value, cols in sentinel_cols.items():
        touched |= set(cols)
        steps.append(
            {
                "kind": "replace_values",
                "params": {"columns": sorted(cols), "values": [value]},
                "why": tr(
                    "alerta 'sentinel': {v:g} parece un código de dato faltante",
                    "'sentinel' alert: {v:g} looks like a missing-data code",
                    v=value,
                ),
            }
        )

    absurd = sorted(
        {a["column"] for a in alerts if a["code"] == "absurd_values" and a["column"] in feats}
    )
    if absurd:
        touched |= set(absurd)
        steps.append(
            {
                "kind": "absurd_to_nan",
                "params": {
                    "columns": absurd,
                    "factor": 20.0,
                    "by": "source" if multi_file else None,
                },
                "why": tr(
                    "alerta 'absurd_values': Valores a más de 20× el rango q01–q99 "
                    "(límites calculados solo con train, por archivo)",
                    "'absurd_values' alert: Values more than 20× the q01–q99 range "
                    "(bounds computed from train only, per file)",
                ),
            }
        )

    if any(a["code"] == "duplicates" for a in alerts):
        steps.append(
            {
                "kind": "drop_duplicates",
                "params": {"ignore": ["source"]},
                "why": tr(
                    "alerta 'duplicates': Filas repetidas (se quitan solo del entrenamiento)",
                    "'duplicates' alert: Repeated rows (removed from training only)",
                ),
            }
        )

    for a in alerts:
        if a["code"] in ("nulls", "missing_column") and a["column"] in feats:
            touched.add(a["column"])
    for n, c in feats.items():
        if c["stats"].get("nulls", 0) > 0:
            touched.add(n)
    num_imp = sorted(n for n in touched if n in num_feats)
    cat_imp = sorted(n for n in touched if n in feats and n not in num_feats)
    if num_imp:
        steps.append(
            {
                "kind": "impute",
                "params": {"columns": num_imp, "strategy": "median"},
                "why": tr(
                    "columnas con nulos, absurdos o ausentes en algún archivo: Mediana de train "
                    "(los absurdos pasan a nulo antes)",
                    "columns with nulls, absurd values or missing in some file: Train median "
                    "(absurd values become null first)",
                ),
            }
        )
    if cat_imp:
        steps.append(
            {
                "kind": "impute",
                "params": {"columns": cat_imp, "strategy": "mode"},
                "why": tr(
                    "categóricas con nulos: Moda de train",
                    "categorical columns with nulls: Train mode",
                ),
            }
        )
    if num_feats:
        steps.append(
            {
                "kind": "scale",
                "params": {"columns": sorted(num_feats), "method": "standard"},
                "why": tr(
                    "escalado estándar ajustado solo con train (redes/SVM lo necesitan; "
                    "se exporta para el firmware)",
                    "standard scaling fitted on train only (networks/SVMs need it; "
                    "it is exported for the firmware)",
                ),
            }
        )
    return steps
