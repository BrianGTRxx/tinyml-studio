"""Rechazo de "desconocido": un umbral de confianza para no contestar cuando el modelo duda.

Un clasificador siempre elige una de sus clases, aunque lo que ve no se parezca a ninguna (un
aparato nuevo, un sensor suelto). Aquí se elige, en VALIDACIÓN, el umbral de probabilidad máxima más
bajo con el que lo que se acepta acierta al menos ``target_accuracy``. El TEST se mide una sola vez.

Además se mide qué tan bien detecta lo desconocido de verdad: por cada clase (hasta ``MAX_HOLDOUT``)
se reentrena el modelo SIN esa clase, se calibra el umbral con las demás y se cuenta cuántas filas
de la clase ausente se rechazan. Eso es lo único que prueba que "desconocido" funciona.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio.clean.prepare import prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit

from . import models as M
from .features import FeatureEncoder
from .run import feature_columns

MAX_HOLDOUT = 6
MIN_ACCEPTED = 20
MIN_KNOWN = 0.9  # al buscar desconocidos, no se renuncia a más del 10 % de lo conocido


def pick_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> dict[str, float]:
    """Umbral más bajo (más cobertura) cuyo aceptado acierta >= target; si ninguno, el más alto."""
    order = np.argsort(conf)
    conf_s, ok_s = conf[order], correct[order].astype(float)
    # precisión de "todo lo que tiene confianza >= conf_s[i]"
    tail_ok = np.cumsum(ok_s[::-1])[::-1]
    tail_n = np.arange(len(conf_s), 0, -1)
    prec = tail_ok / tail_n
    good = np.where((prec >= target) & (tail_n >= min(MIN_ACCEPTED, len(conf_s))))[0]
    i = int(good[0]) if len(good) else len(conf_s) - 1
    thr = float(conf_s[i])
    keep = conf >= thr
    return {
        "threshold": round(thr, 4),
        "coverage": round(float(keep.mean()), 4),
        "accuracy_accepted": round(float(correct[keep].mean()), 4) if keep.any() else float("nan"),
        "reached_target": bool(len(good)),
    }


def _apply(conf, correct, thr) -> dict[str, float | None]:
    keep = conf >= thr
    return {
        "coverage": round(float(keep.mean()), 4),
        "accuracy_accepted": round(float(correct[keep].mean()), 4) if keep.any() else None,
    }


def calibrate_rejection(
    run_dir: str | Path,
    model: str | None = None,
    *,
    target_accuracy: float = 0.95,
    holdout: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    import joblib

    run = Path(run_dir)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    recipe = json.loads((run / "recipe_fitted.json").read_text(encoding="utf-8"))
    if results["task"] != "classification":
        raise ValueError(
            tr(
                "el rechazo de desconocido solo aplica a clasificación",
                "unknown rejection only applies to classification",
            )
        )
    enc = FeatureEncoder.from_dict(json.loads((run / "encoder.json").read_text(encoding="utf-8")))
    name = model or results["selection"]["best"]
    path = run / f"model_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(tr("no existe {f}", "{f} does not exist", f=path.name))
    est = joblib.load(path)
    if not hasattr(est, "predict_proba"):
        raise ValueError(
            tr(
                "'{m}' no da probabilidades: Usa un árbol, bosque, regresión logística, MLP, "
                "boosting o Naive Bayes",
                "'{m}' gives no probabilities: Use a tree, forest, logistic regression, MLP, "
                "boosting or Naive Bayes",
                m=name,
            )
        )
    prep = prepare_splits(recipe, seed=results["reproducibility"]["seed"])
    X = {s: enc.transform(df) for s, df in prep.clean.items()}
    y = {s: enc.encode_target(df[prep.target]) for s, df in prep.clean.items()}

    def conf_ok(e, split, classes=None):
        p = e.predict_proba(X[split])
        pred = np.asarray(e.classes_)[p.argmax(1)]
        return p.max(1), pred == y[split]

    emit("reject", 10, tr("Calibrando el umbral…", "Calibrating the threshold…"))
    cv, ok_v = conf_ok(est, "val")
    pick = pick_threshold(cv, ok_v, target_accuracy)
    ct, ok_t = conf_ok(est, "test")
    curve = []
    for thr in np.unique(np.round(np.quantile(cv, np.linspace(0, 0.95, 20)), 4)):
        curve.append({"threshold": float(thr), **_apply(cv, ok_v, thr)})
    res: dict[str, Any] = {
        "model": name,
        "target_accuracy": target_accuracy,
        "threshold": pick["threshold"],
        "reached_target": pick["reached_target"],
        "val": {"coverage": pick["coverage"], "accuracy_accepted": pick["accuracy_accepted"]},
        "test": _apply(ct, ok_t, pick["threshold"]),  # medido una sola vez
        "curve": curve,
        "unknown_detection": None,
        "family": "rejection",
    }
    if holdout:
        res["unknown_detection"] = _holdout(
            recipe, prep, enc, name, M, target_accuracy, seed, results["task"]
        )
    res["summary"] = summary(res)
    emit("done", 100, tr("Umbral calibrado", "Threshold calibrated"))
    return res


def _holdout(recipe, prep, enc, name, M_, target, seed, task) -> dict[str, Any] | None:
    classes = list(range(len(enc.classes)))
    if len(classes) < 3:
        return None  # con 2 clases, quitar una deja un problema de una sola clase
    feats = feature_columns(recipe, prep.clean["train"])
    sizes = [
        (int((prep.clean["train"][prep.target].astype(str) == c).sum()), i)
        for i, c in enumerate(enc.classes)
    ]
    chosen = [i for _, i in sorted(sizes, reverse=True)[:MAX_HOLDOUT]]
    rows = []
    known_conf: list[np.ndarray] = []
    absent_conf: list[np.ndarray] = []
    for n, k in enumerate(chosen):
        emit(
            "reject",
            20 + 70 * n / len(chosen),
            tr("Sin la clase {c}…", "Without class {c}…", c=enc.classes[k]),
        )
        train = prep.clean["train"]
        keep_tr = train[prep.target].astype(str) != enc.classes[k]
        val = prep.clean["val"]
        keep_va = val[prep.target].astype(str) != enc.classes[k]
        enc2 = FeatureEncoder.fit(train[keep_tr], feats, prep.target, task, recipe.get("encoding"))
        Xtr, ytr = enc2.transform(train[keep_tr]), enc2.encode_target(train[keep_tr][prep.target])
        est, _ = M_.build(name, task, seed, None)
        est.fit(Xtr, ytr)
        va = val[keep_va]
        p = est.predict_proba(enc2.transform(va))
        pred = np.asarray(est.classes_)[p.argmax(1)]
        pick = pick_threshold(p.max(1), pred == enc2.encode_target(va[prep.target]), target)
        absent = val[~keep_va]  # solo validación: el test no participa en elegir umbrales
        if absent.empty:
            continue
        p_abs = est.predict_proba(enc2.transform(absent))
        known_conf.append(p.max(1))
        absent_conf.append(p_abs.max(1))
        rejected = float((p_abs.max(1) < pick["threshold"]).mean())
        rows.append(
            {
                "class": enc.classes[k],
                "rows": int(len(absent)),
                "rejected": round(rejected, 4),
                "known_coverage": pick["coverage"],
            }
        )
    if not rows:
        return None
    # umbral único que rechaza lo máximo de lo desconocido sin dejar de responder >= MIN_KNOWN
    # de lo conocido (promedio sobre las clases retiradas; es un proxy del modelo final)
    grid = np.round(np.linspace(0.3, 0.99, 70), 3)
    best = None
    for thr in grid:
        cov = float(np.mean([(c >= thr).mean() for c in known_conf]))
        rej = float(np.mean([(c < thr).mean() for c in absent_conf]))
        if cov >= MIN_KNOWN and (best is None or rej > best["rejected"]):
            best = {
                "threshold": float(thr),
                "rejected": round(rej, 4),
                "known_coverage": round(cov, 4),
            }
    return {
        "suggested": best,
        "classes": rows,
        "mean_rejected": round(float(np.mean([r["rejected"] for r in rows])), 4),
        "mean_known_coverage": round(float(np.mean([r["known_coverage"] for r in rows])), 4),
    }


def summary(res: dict[str, Any]) -> str:
    v, t = res["val"], res["test"]
    lines = [
        tr(
            "Umbral de confianza para {m}: {th} (objetivo: {a:.0%} de aciertos en lo aceptado).",
            "Confidence threshold for {m}: {th} (goal: {a:.0%} correct among accepted).",
            m=res["model"],
            th=res["threshold"],
            a=res["target_accuracy"],
        ),
        tr(
            "Validación: Responde {c:.1%} de las veces y acierta {a:.1%}. Test (una vez): "
            "responde {tc:.1%} y acierta {ta}.",
            "Validation: Answers {c:.1%} of the time and is right {a:.1%}. Test (once): "
            "answers {tc:.1%} and is right {ta}.",
            c=v["coverage"],
            a=v["accuracy_accepted"] or 0,
            tc=t["coverage"],
            ta="—" if t["accuracy_accepted"] is None else f"{t['accuracy_accepted']:.1%}",
        ),
    ]
    if not res["reached_target"]:
        lines.append(
            tr(
                "Con este modelo ningún umbral alcanza el objetivo: El modelo está seguro y se "
                "equivoca. No lo uses para rechazar sin mejorar el modelo.",
                "No threshold reaches the goal with this model: It is confident and wrong. Do not "
                "use it for rejection before improving the model.",
            )
        )
    u = res["unknown_detection"]
    if u:
        lines.append(
            tr(
                "Detección de lo desconocido (se reentrena sin una clase): Rechaza en promedio "
                "{r:.1%} de la clase ausente y sigue respondiendo {k:.1%} de lo conocido.",
                "Unknown detection (retrained without one class): Rejects on average {r:.1%} of "
                "the missing class and still answers {k:.1%} of the known data.",
                r=u["mean_rejected"],
                k=u["mean_known_coverage"],
            )
        )
        sg = u.get("suggested")
        if sg:
            lines.append(
                tr(
                    "Para detectar desconocidos conviene un umbral más alto: {t} rechaza "
                    "{r:.1%} de lo desconocido y deja responder {k:.1%} de lo conocido "
                    "(estimado con modelos reentrenados sin una clase, solo validación; lo "
                    "conocido sale de las mismas sesiones: Con sesiones nuevas la confianza "
                    "baja y se rechazará más).",
                    "To catch unknowns a higher threshold works better: {t} rejects "
                    "{r:.1%} of unknowns and still answers {k:.1%} of known data "
                    "(estimated with models retrained without a class, validation only; the known "
                    "data comes from the same sessions: On new sessions confidence drops and more "
                    "is rejected).",
                    t=sg["threshold"],
                    r=sg["rejected"],
                    k=sg["known_coverage"],
                )
            )
    else:
        lines.append(
            tr(
                "No se midió la detección de lo desconocido (hacen falta 3 o más clases).",
                "Unknown detection was not measured (needs 3 or more classes).",
            )
        )
    return "\n".join(lines)
