"""Búsqueda de hiperparámetros con presupuesto de la placa (flash y RAM).

Busca al azar (semilla fija, reproducible) en una rejilla pequeña por modelo. Cada candidato se
entrena sobre TRAIN, se puntúa en VALIDACIÓN y se mide con el mismo emisor de C que la exportación
(flash de tablas + RAM de trabajo). Los que no caben se descartan; entre los que caben gana el mejor
puntaje (empate: el más pequeño). El TEST se mide una sola vez, solo para el ganador, y NO participa
en la elección.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from tinyml_studio.clean.prepare import prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit

from . import models as M
from .bench import score_fn
from .features import FeatureEncoder
from .run import feature_columns

MAX_TRIALS = 200
GRIDS: dict[str, dict[str, list]] = {
    "decision_tree": {
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 16],
        "min_samples_leaf": [1, 2, 5, 10, 20],
    },
    "random_forest": {
        "n_estimators": [5, 10, 20, 30, 50, 80],
        "max_depth": [3, 4, 6, 8, 10, 12],
        "min_samples_leaf": [1, 3, 5, 10],
    },
    "extra_trees": {
        "n_estimators": [5, 10, 20, 30, 50, 80],
        "max_depth": [3, 4, 6, 8, 10, 12],
        "min_samples_leaf": [1, 3, 5, 10],
    },
    "gradient_boosting": {
        "max_iter": [20, 50, 100, 200],
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "mlp": {
        "hidden_layer_sizes": [[8], [16], [32], [16, 8], [32, 16], [64, 32]],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    },
}


def tunable() -> list[str]:
    return list(GRIDS)


def _sample(grid: dict[str, list], n: int, rng: np.random.Generator) -> list[dict]:
    total = int(np.prod([len(v) for v in grid.values()]))
    keys = list(grid)
    picks = rng.permutation(total)[: min(n, total)]
    out = []
    for flat in picks:
        p, r = {}, int(flat)
        for k in keys:
            r, i = divmod(r, len(grid[k]))
            p[k] = grid[k][i]
        out.append(p)
    return out


def _cost(name, est, spec, task, n_classes) -> tuple[int, int] | None:
    """(bytes de flash, bytes de RAM) con el emisor de C; None si el modelo no tiene emisor."""
    from tinyml_studio.export.cgen import EMITTERS

    if name not in EMITTERS:
        return None
    _, data_bytes, ram_model, _ = EMITTERS[name](est, name, spec, task, n_classes)
    return int(data_bytes), int(ram_model + 4 * spec.n_features)


def _kb(v: int | None) -> str:
    return "—" if v is None else f"{v / 1024:.1f} KB"


def tune(
    recipe: dict[str, Any],
    model: str = "random_forest",
    *,
    flash_kb: float | None = None,
    ram_kb: float | None = None,
    n_trials: int = 24,
    seed: int = 42,
    exclude: tuple[str, ...] = (),
    max_seconds: float | None = None,
) -> dict[str, Any]:
    from tinyml_studio.export.spec import build_spec

    task = recipe["task"]
    if task not in ("classification", "regression"):
        raise ValueError(tr("tarea no entrenable: {t}", "task cannot be trained: {t}", t=task))
    if model not in GRIDS:
        raise ValueError(
            tr(
                "no hay rejilla de búsqueda para '{m}'. Disponibles: {a}",
                "no search grid for '{m}'. Available: {a}",
                m=model,
                a=", ".join(GRIDS),
            )
        )
    if model not in M.available(task):
        raise ValueError(
            tr(
                "'{m}' no sirve para la tarea {t}",
                "'{m}' does not work for task {t}",
                m=model,
                t=task,
            )
        )
    n_trials = max(1, min(int(n_trials), MAX_TRIALS))
    emit("prepare", 3, tr("Preparando datos…", "Preparing data…"))
    prep = prepare_splits(recipe, seed=seed)
    target = prep.target
    feats = feature_columns(recipe, prep.clean["train"], exclude)
    enc = FeatureEncoder.fit(prep.clean["train"], feats, target, task, recipe.get("encoding"))
    X = {s: enc.transform(df) for s, df in prep.clean.items()}
    cls = task == "classification"
    y = {
        s: enc.encode_target(df[target]) if cls else np.asarray(df[target], dtype="float64")
        for s, df in prep.clean.items()
    }
    n_cls = len(enc.classes) if cls else 0
    metric = "f1_macro" if cls else "r2"
    spec = build_spec(prep.pipeline, enc)
    rng = np.random.default_rng(seed)
    cands = _sample(GRIDS[model], n_trials, rng)
    defaults = {k: v for k, v in M.DEFAULTS[model].items() if k in GRIDS[model]}
    if defaults and defaults not in cands:
        cands.insert(0, defaults)  # el punto de partida de la app siempre se evalúa

    t0 = time.time()
    rows: list[dict] = []
    ests: dict[int, Any] = {}
    for i, p in enumerate(cands):
        if max_seconds is not None and time.time() - t0 > max_seconds and rows:
            break
        emit(
            "tune",
            5 + 85 * i / max(len(cands), 1),
            tr("Probando {i}/{n}…", "Trying {i}/{n}…", i=i + 1, n=len(cands)),
        )
        try:
            est, _eff = M.build(model, task, seed, p)
            est.fit(X["train"], y["train"])
            score = score_fn(task, metric, y["val"], est.predict(X["val"]), n_cls)
            cost = _cost(model, est, spec, task, n_cls)
        except Exception as exc:  # noqa: BLE001 - un candidato roto no tumba la búsqueda
            rows.append({"params": p, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        flash, ram = cost if cost else (None, None)
        fits = (
            (flash_kb is None or (flash is not None and flash <= flash_kb * 1024))
            and (ram_kb is None or (ram is not None and ram <= ram_kb * 1024))
            and (cost is not None or (flash_kb is None and ram_kb is None))
        )
        rows.append(
            {
                "params": p,
                "val_score": None if score != score else round(float(score), 4),
                "flash_bytes": flash,
                "ram_bytes": ram,
                "fits": bool(fits),
                "complexity": M.complexity(model, est),
                "is_default": p == defaults,
            }
        )
        ests[len(rows) - 1] = est
    ok = [(i, r) for i, r in enumerate(rows) if "error" not in r and r["val_score"] is not None]
    if not ok:
        raise ValueError(tr("ningún candidato se pudo evaluar", "no candidate could be evaluated"))
    fitting = [(i, r) for i, r in ok if r["fits"]]
    pool = fitting or ok
    best_i, best = max(
        pool, key=lambda ir: (ir[1]["val_score"], -(ir[1]["flash_bytes"] or 0), -ir[0])
    )
    est = ests[best_i]
    test_score = score_fn(task, metric, y["test"], est.predict(X["test"]), n_cls)
    base = next((r for _, r in ok if r["is_default"]), None)
    best["chosen"] = True
    res = {
        "model": model,
        "task": task,
        "metric": metric,
        "budget": {"flash_kb": flash_kb, "ram_kb": ram_kb},
        "trials": rows,
        "n_evaluated": len(ok),
        "n_fitting": len(fitting),
        "best": {
            "params": best["params"],
            "val_score": best["val_score"],
            "flash_bytes": best["flash_bytes"],
            "ram_bytes": best["ram_bytes"],
            "test_score": None if test_score != test_score else round(float(test_score), 4),
            "fits_budget": bool(fitting),
        },
        "default": None
        if base is None
        else {
            "val_score": base["val_score"],
            "flash_bytes": base["flash_bytes"],
            "fits_budget": base["fits"],
        },
        "seconds": round(time.time() - t0, 1),
        "family": "tuning",
    }
    res["summary"] = summary(res)
    emit("done", 100, tr("Búsqueda terminada", "Search finished"))
    return res


def summary(res: dict[str, Any]) -> str:
    b, d = res["best"], res["default"]
    lines = [
        tr(
            "Búsqueda para {m}: {n} candidatos evaluados, {f} caben en el presupuesto.",
            "Search for {m}: {n} candidates evaluated, {f} fit the budget.",
            m=res["model"],
            n=res["n_evaluated"],
            f=res["n_fitting"],
        )
    ]
    if not b["fits_budget"] and (res["budget"]["flash_kb"] or res["budget"]["ram_kb"]):
        lines.append(
            tr(
                "NINGÚN candidato cabe: Se muestra el mejor sin presupuesto. Sube el presupuesto o "
                "prueba otro modelo.",
                "NO candidate fits: Showing the best ignoring the budget. Raise the budget or try "
                "another model.",
            )
        )
    lines.append(
        tr(
            "Mejor: {p} · {mt} en validación {v} · flash {fl} · RAM {r} · en test {t} "
            "(medido una sola vez).",
            "Best: {p} · {mt} on validation {v} · flash {fl} · RAM {r} · on test {t} "
            "(measured once).",
            p=b["params"],
            mt=res["metric"],
            v=b["val_score"],
            fl=_kb(b["flash_bytes"]),
            r=_kb(b["ram_bytes"]),
            t=b["test_score"],
        )
    )
    if d:
        lines.append(
            tr(
                "Parámetros por defecto: {mt} {v}, flash {fl}, {fit}.",
                "Default parameters: {mt} {v}, flash {fl}, {fit}.",
                mt=res["metric"],
                v=d["val_score"],
                fl=_kb(d["flash_bytes"]),
                fit=tr("caben", "fit") if d["fits_budget"] else tr("no caben", "do not fit"),
            )
        )
    return "\n".join(lines)
