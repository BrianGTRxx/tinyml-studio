"""Resumen legible del informe del motor (``summary_es`` y ``summary`` en el idioma activo)."""

from __future__ import annotations

from tinyml_studio.i18n import tr

TASKS = {
    "classification": "clasificación",
    "regression": "regresión",
    "unsupervised": "sin etiqueta (anomalías / exploración)",
}
TASKS_EN = {
    "classification": "classification",
    "regression": "regression",
    "unsupervised": "unlabeled (anomalies / exploration)",
}
MAX_SHOWN = 25
SEV = {"high": "ALTA", "medium": "media", "info": "info"}
SEV_EN = {"high": "HIGH", "medium": "medium", "info": "info"}


def summary(r: dict) -> str:
    """Resumen en el idioma activo (``tr``)."""
    return _summary(r)


def summary_es(r: dict) -> str:
    return _summary(r)


def _summary(r: dict) -> str:
    L: list[str] = []
    L.append(
        tr(
            "Archivos: {n} · filas totales: {t}",
            "Files: {n} · total rows: {t}",
            n=len(r["files"]),
            t=r["rows_total"],
        )
    )
    for f in r["files"]:
        L.append(
            tr(
                "  - {f}: {r} filas × {c} columnas",
                "  - {f}: {r} rows × {c} columns",
                f=f["file"],
                r=f["rows"],
                c=f["cols"],
            )
        )
    cols = r["columns"]
    n_by_role: dict[str, int] = {}
    for c in cols:
        n_by_role[c["role"]] = n_by_role.get(c["role"], 0) + 1
    L.append(
        tr("Columnas por rol: ", "Columns by role: ")
        + ", ".join(f"{k}={v}" for k, v in sorted(n_by_role.items()))
    )
    t = r["task"]
    L.append("")
    L.append(
        tr(
            "Tarea sugerida: {t}",
            "Suggested task: {t}",
            t=tr(TASKS[t["task"]], TASKS_EN[t["task"]]),
        )
    )
    L.append(f"  {t['reason']}")
    if t["needs_user_confirmation"]:
        L.append(
            tr(
                "  ⚠ Confírmalo o corrígelo en la receta.",
                "  ⚠ Confirm or correct it in the recipe.",
            )
        )
    if r["target_candidates"]:
        L.append(
            tr("  Candidatas a objetivo: ", "  Target candidates: ")
            + "; ".join(f"{c['column']} ({c['score']})" for c in r["target_candidates"])
        )
    s = r["split"]
    L.append(
        tr(
            "Partición sugerida: {a} — {b}",
            "Suggested split: {a} — {b}",
            a=s["strategy"],
            b=s["reason"],
        )
    )
    if r["column_groups"]:
        L.append("")
        L.append(
            tr(
                "Columnas equivalentes entre archivos (revisa):",
                "Equivalent columns across files (review):",
            )
        )
        for g in r["column_groups"]:
            names = sorted(set(g["members"].values()))
            if len(names) > 1:
                L.append(f"  - {g['canonical']} ← {', '.join(names)} ({g['reason']})")
    drops = [c for c in cols if c["role"] == "drop"]
    if drops:
        L.append("")
        L.append(tr("Se descartarían:", "Would be dropped:"))
        for c in drops:
            L.append(f"  - {c['name']}: {c['role_reason']}")
    important = [a for a in r["alerts"] if a["severity"] != "info"]
    infos = [a for a in r["alerts"] if a["severity"] == "info"]
    if important:
        L.append("")
        L.append(tr("Alertas:", "Alerts:"))
        for a in important[:MAX_SHOWN]:
            L.append(f"  [{tr(SEV[a['severity']], SEV_EN[a['severity']])}] {a['message']}")
        if len(important) > MAX_SHOWN:
            L.append(
                tr(
                    "  … y {n} más (ver --json)",
                    "  … and {n} more (see --json)",
                    n=len(important) - MAX_SHOWN,
                )
            )
    if infos:
        by: dict[str, int] = {}
        for a in infos:
            by[a["code"]] = by.get(a["code"], 0) + 1
        L.append("")
        L.append(
            tr("Avisos informativos: ", "Informational notices: ")
            + ", ".join(f"{k}={v}" for k, v in sorted(by.items()))
            + tr(" (detalle en --json)", " (details in --json)")
        )
    return "\n".join(L)
