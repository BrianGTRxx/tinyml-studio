"""Importancia de variables por permutación (sobre VALIDACIÓN, nunca test).

Se baraja una columna ORIGINAL a la vez (las categóricas con one-hot cuentan como una sola) y se
mide cuánto cae la métrica. Una columna con caída ~0 no le sirve al modelo: se puede quitar del
firmware (menos sensores, menos RAM). La caída es relativa al modelo y a los datos de validación;
si la validación sale de las mismas sesiones que el entrenamiento, una columna que identifica la
sesión puede salir "importante" sin generalizar (ver la validación por sesiones).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio.clean.prepare import prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit

from .bench import score_fn
from .features import FeatureEncoder

MAX_ROWS = 5000
USELESS = 0.002  # caída mínima absoluta que se considera señal
WEAK_FRACTION = 0.1  # y menos de este tanto de la caída de la mejor columna = aporta muy poco


def permutation_importance(
    run_dir: str | Path,
    model: str | None = None,
    *,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    import joblib

    run = Path(run_dir)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    recipe = json.loads((run / "recipe_fitted.json").read_text(encoding="utf-8"))
    enc = FeatureEncoder.from_dict(json.loads((run / "encoder.json").read_text(encoding="utf-8")))
    task = results["task"]
    name = model or results["selection"]["best"]
    path = run / f"model_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            tr("no existe {f}", "{f} does not exist", f=path.name),
        )
    est = joblib.load(path)
    prep = prepare_splits(recipe, seed=results["reproducibility"]["seed"])
    val = prep.clean["val"]
    if len(val) > MAX_ROWS:
        val = val.sample(MAX_ROWS, random_state=seed)
    val = val.reset_index(drop=True)
    cls = task == "classification"
    y = enc.encode_target(val[prep.target]) if cls else np.asarray(val[prep.target], dtype=float)
    k = len(enc.classes) if cls else 0
    metric = "f1_macro" if cls else "r2"
    base = score_fn(task, metric, y, est.predict(enc.transform(val)), k)
    rng = np.random.default_rng(seed)
    cols = [*enc.numeric, *enc.categorical]
    rows = []
    for i, c in enumerate(cols):
        emit("importance", 5 + 90 * i / max(len(cols), 1), c)
        drops = []
        for _ in range(max(n_repeats, 1)):
            shuffled = val.copy()
            shuffled[c] = rng.permutation(val[c].to_numpy())
            drops.append(base - score_fn(task, metric, y, est.predict(enc.transform(shuffled)), k))
        rows.append(
            {
                "feature": c,
                "drop": round(float(np.mean(drops)), 4),
                "std": round(float(np.std(drops)), 4),
            }
        )
    rows.sort(key=lambda r: -r["drop"])
    limit = max(USELESS, WEAK_FRACTION * rows[0]["drop"]) if rows else USELESS
    for r in rows:  # un bosque también parte por ruido: se compara con la mejor columna
        r["useless"] = bool(r["drop"] <= limit)
    useless = [r["feature"] for r in rows if r["useless"]]
    res = {
        "model": name,
        "task": task,
        "metric": metric,
        "base_score": round(float(base), 4),
        "rows_used": int(len(val)),
        "features": rows,
        "useless": useless,
        "family": "importance",
    }
    res["summary"] = summary(res)
    emit("done", 100, tr("Importancia calculada", "Importance computed"))
    return res


def summary(res: dict[str, Any]) -> str:
    limit = max(USELESS, WEAK_FRACTION * res["features"][0]["drop"]) if res["features"] else USELESS
    top = ", ".join(f"{r['feature']} (-{r['drop']:.3f})" for r in res["features"][:5])
    lines = [
        tr(
            "Importancia por permutación de {m} ({mt} base {b}, {n} filas de validación).",
            "Permutation importance of {m} ({mt} base {b}, {n} validation rows).",
            m=res["model"],
            mt=res["metric"],
            b=res["base_score"],
            n=res["rows_used"],
        ),
        tr("Más importantes: {t}.", "Most important: {t}.", t=top),
    ]
    if res["useless"]:
        lines.append(
            tr(
                "Aportan muy poco (caída ≤ {u}): {c}. Quitarlas ahorraría sensores y RAM; vuelve a "
                "entrenar sin ellas para confirmarlo.",
                "Add very little (drop ≤ {u}): {c}. Removing them would save sensors and RAM; "
                "retrain without them to confirm.",
                u=round(limit, 3),
                c=", ".join(res["useless"]),
            )
        )
    return "\n".join(lines)
