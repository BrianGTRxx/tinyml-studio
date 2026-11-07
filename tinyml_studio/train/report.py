"""Resumen legible de una corrida (``*_summary_es`` y ``*_summary`` en el idioma activo)."""

from __future__ import annotations

from tinyml_studio.i18n import tr


def _task(cls: bool) -> str:
    return tr("clasificación", "classification") if cls else tr("regresión", "regression")


def summary(r: dict) -> str:
    """Resumen en el idioma activo (``tr``)."""
    cls = r["task"] == "classification"
    key = "f1_macro" if cls else "r2"
    L = [
        tr(
            "Tarea: {t} · objetivo: {o}",
            "Task: {t} · target: {o}",
            t=_task(cls),
            o=r["target"],
        ),
        tr("Filas: ", "Rows: ") + ", ".join(f"{k}={v}" for k, v in r["rows"].items()),
        tr("Features ({n}): ", "Features ({n}): ", n=len(r["features_used"]))
        + ", ".join(r["features_used"]),
    ]
    if r["features_skipped"]:
        L.append(
            tr("Features omitidas: ", "Skipped features: ")
            + "; ".join(f"{k} ({v})" for k, v in r["features_skipped"].items())
        )
    if r["features_excluded_by_user"]:
        L.append(
            tr("Excluidas por el usuario: ", "Excluded by the user: ")
            + ", ".join(r["features_excluded_by_user"])
        )
    b = r["baseline"]
    L.append(
        tr(
            "Línea base ({key}): Val={v:.3f} test={t:.3f}",
            "Baseline ({key}): Val={v:.3f} test={t:.3f}",
            key=key,
            v=b["val"][key],
            t=b["test"][key],
        )
    )
    L.append("")
    extra = "acc" if cls else "mae"
    L.append(
        f"{tr('modelo', 'model'):<14}{'val ' + key:>12}{'test ' + key:>12}{'test ' + extra:>12}"
        f"{tr('ajuste s', 'fit s'):>10}{tr('tamaño', 'size'):>22}"
    )
    for n, m in r["models"].items():
        mark = " ←" if n == r["selection"]["best"] else ""
        te = m["test"]
        size = ", ".join(f"{k}={v}" for k, v in m["complexity"].items())
        third = te["accuracy" if cls else "mae"]
        L.append(
            f"{n:<14}{m['val'][key]:>12.3f}{te[key]:>12.3f}{third:>12.3f}"
            f"{m['fit_seconds']:>10.1f}{size:>22}{mark}"
        )
    L.append(
        "\n"
        + tr(
            "Mejor por {m} en {s} (el test no se usó para elegir): {b}",
            "Best by {m} on {s} (test was not used to choose): {b}",
            m=r["selection"]["metric"],
            s=r["selection"]["split"],
            b=r["selection"]["best"],
        )
    )
    best = r["models"][r["selection"]["best"]]
    if best["top_features"]:
        L.append(
            tr("Features más influyentes: ", "Most influential features: ")
            + ", ".join(f"{f['feature']} ({f['importance']})" for f in best["top_features"][:5])
        )
    for n, m in r["models"].items():
        for note in m["notes"]:
            L.append(tr("Nota {n}: {note}", "Note {n}: {note}", n=n, note=note))
    if r["warnings"]:
        L.append("")
        L.append(tr("Advertencias:", "Warnings:"))
        L += [f"  - {w}" for w in r["warnings"]]
    return "\n".join(L)


def summary_es(r: dict) -> str:
    return summary(r)


def _warn_lines(r: dict) -> list[str]:
    return [tr("Advertencia: {w}", "Warning: {w}", w=w) for w in r["warnings"]]


def keras_summary(r: dict) -> str:
    cls = r["task"] == "classification"
    key = "f1_macro" if cls else "r2"
    h = r["hyperparameters"]
    L = [
        tr(
            "MLP Keras {h} · {p} parámetros · {e} épocas "
            "(máx. {em}, parada temprana con val_loss) · {s} s",
            "Keras MLP {h} · {p} parameters · {e} epochs "
            "(max. {em}, early stopping on val_loss) · {s} s",
            h=h["hidden"],
            p=r["parameters"],
            e=h["epochs_run"],
            em=h["epochs_max"],
            s=r["fit_seconds"],
        ),
        tr(
            "Tarea: {t} · objetivo: {o}",
            "Task: {t} · target: {o}",
            t=_task(cls),
            o=r["target"],
        ),
        tr("Filas: ", "Rows: ") + ", ".join(f"{k}={v}" for k, v in r["rows"].items()),
        tr(
            "Línea base ({key}): Test={t:.3f}",
            "Baseline ({key}): Test={t:.3f}",
            key=key,
            t=r["baseline"]["test"][key],
        ),
        f"{key}: val={r['val'][key]:.3f} test={r['test'][key]:.3f}",
    ]
    L += _warn_lines(r)
    return "\n".join(L)






def cnn1d_summary(r: dict) -> str:
    a, h = r["architecture"], r["hyperparameters"]
    w = r["window"]
    L = [
        tr(
            "1D-CNN Keras · ventana {w} filas (paso {st} en train) · "
            "filtros {f} kernel {k} cabeza {hd} · "
            "{p} parámetros · {e} épocas · {s} s",
            "Keras 1D-CNN · window {w} rows (stride {st} in train) · "
            "filters {f} kernel {k} head {hd} · "
            "{p} parameters · {e} epochs · {s} s",
            w=w["window"],
            st=w["stride"],
            f=a["filters"],
            k=a["kernel"],
            hd=a["head"],
            p=r["parameters"],
            e=h["epochs_run"],
            s=r["fit_seconds"],
        ),
        tr("Ventanas: ", "Windows: ") + ", ".join(f"{k}={v}" for k, v in r["windows"].items()),
        tr(
            "F1 macro: Val={v:.3f} test={t:.3f} (exactitud test {a:.3f})",
            "Macro F1: Val={v:.3f} test={t:.3f} (test accuracy {a:.3f})",
            v=r["val"]["f1_macro"],
            t=r["test"]["f1_macro"],
            a=r["test"]["accuracy"],
        ),
        tr(
            "Referencias en test: Clase mayoritaria {m:.3f} · "
            "random forest con solo la última fila {rf:.3f}",
            "Test references: Majority class {m:.3f} · "
            "random forest with only the last row {rf:.3f}",
            m=r["baseline"]["test"]["f1_macro"],
            rf=r["reference_rf_last_row"]["test"]["f1_macro"],
        ),
    ]
    L += _warn_lines(r)
    return "\n".join(L)


def keras_summary_es(r: dict) -> str:
    return keras_summary(r)






def cnn1d_summary_es(r: dict) -> str:
    return cnn1d_summary(r)


def benchmark_summary(r: dict) -> str:
    """Resumen del benchmark (leaderboard, selección, empates, advertencias) en el idioma activo."""
    sel = r["selection"]
    m = sel["metric"]
    L = [
        tr("Benchmark · tarea: {t} · objetivo: {o}", "Benchmark · task: {t} · target: {o}",
           t=r["task"], o=r["target"]),
        tr("Filas: ", "Rows: ") + ", ".join(f"{k}={v}" for k, v in r["rows"].items()),
        tr(
            "Selección por {m} en {s} (regla de 1 error estándar); el test NO elige.",
            "Selection by {m} on {s} (one-standard-error rule); test does NOT choose.",
            m=m,
            s=sel["source"],
        ),
        "",
    ]  # fmt: skip
    for row in r["leaderboard"]:
        v, t = row["metrics"]["val"].get(m), row["metrics"]["test"].get(m)
        ci = row.get("test_ci") or {}
        flags = [
            f
            for f, on in (
                ("*", row.get("selected")),
                ("pareto", row.get("pareto")),
                ("C/TFLite", row.get("exportable")),
                (tr("no cabe", "no fit"), row.get("fits_flash") is False),
                (tr("empate", "tied"), row.get("statistically_tied_with_best")
                 and not row.get("selected")),
            )
            if on
        ]  # fmt: skip
        sz = row.get("size_bytes")
        cv = row.get("cv")
        L.append(
            f"{row['rank']:>2}. {row['name']:<28} val={_f(v)} test={_f(t)}"
            f" IC95%[{_f(ci.get('lo'))},{_f(ci.get('hi'))}]"
            + (f" CV={_f(cv['mean'])}±{_f(cv['std'])}" if cv else "")
            + (f" {sz / 1024:.1f}KB" if sz is not None else "")
            + (f" [{', '.join(flags)}]" if flags else "")
        )
    L += ["", tr("Recomendado: ", "Recommended: ") + str(sel["selected"]), sel["reason"]]
    if r.get("models_skipped"):
        L.append(
            tr("Omitidos: ", "Skipped: ")
            + "; ".join(f"{s['name']} ({s['reason']})" for s in r["models_skipped"])
        )
    L += [tr("Advertencia: {w}", "Warning: {w}", w=w) for w in r["warnings"]]
    return "\n".join(L)


def _f(x) -> str:
    return "—" if x is None else f"{x:.3f}"
