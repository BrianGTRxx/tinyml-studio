"""Latencia ESTIMADA de una predicción en cada placa (no es una medición).

Se cuentan las operaciones del modelo (MAC, comparaciones, exponenciales) y se multiplican por
ciclos por operación SUPUESTOS según el tipo de CPU (con FPU, sin FPU o Linux) y se divide
por el reloj de la ficha de la placa. Los ciclos por operación son supuestos nuestros, no datos del
fabricante, y no se midieron en hardware: por eso siempre se da un RANGO (ideal .. x3 por caché
de flash, interrupciones y compilador sin optimizar) y, si la ficha no confirma la FPU, se cubren
los dos casos. Para el valor real hay que cronometrar ``tm_predict`` en la placa.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio.boards.loader import load_boards
from tinyml_studio.clean import Pipeline
from tinyml_studio.i18n import tr
from tinyml_studio.train.features import FeatureEncoder

from .spec import build_spec

# ciclos por operación SUPUESTOS (ver docstring)
CYCLES = {
    "fpu": {"mac": 4.0, "cmp": 8.0, "exp": 60.0},
    "soft": {"mac": 80.0, "cmp": 40.0, "exp": 1500.0},
    "linux": {"mac": 1.5, "cmp": 4.0, "exp": 30.0},
}
SLOWDOWN = 3.0  # cota alta: caché de flash (XIP), interrupciones, compilación sin optimizar
PRE_MAC_PER_FEATURE = 3  # limpieza + escalado por variable
LINEAR = ("logreg", "linear", "linear_svm", "lda", "lasso", "elasticnet")


def count_ops(name: str, est, n_features: int) -> dict[str, int]:
    """Operaciones de UNA predicción (cota alta: se toma el camino más largo en árboles)."""
    ops = {"mac": PRE_MAC_PER_FEATURE * n_features, "cmp": 0, "exp": 0}
    if name == "decision_tree":
        ops["cmp"] += int(est.get_depth())
    elif name in ("random_forest", "extra_trees"):
        ops["cmp"] += int(sum(t.get_depth() for t in est.estimators_))
        ops["mac"] += len(est.estimators_) * int(
            np.size(est.classes_) if hasattr(est, "classes_") else 1
        )
    elif name == "gradient_boosting":
        preds = [p for stage in est._predictors for p in stage]
        ops["cmp"] += int(sum(np.ceil(np.log2(len(p.nodes) + 1)) for p in preds))
    elif name in LINEAR:
        ops["mac"] += int(np.size(est.coef_))
    elif name in ("svm", "svr"):
        sv = int(len(est.support_))
        ops["mac"] += sv * n_features + sv * max(int(np.size(getattr(est, "classes_", [0]))) - 1, 1)
        ops["exp"] += sv
    elif name == "gaussian_nb":
        ops["mac"] += 3 * int(est.theta_.size)
    elif name == "mlp":
        ops["mac"] += int(sum(w.size for w in est.coefs_))
        neurons = int(sum(b.size for b in est.intercepts_))
        if getattr(est, "activation", "relu") in ("tanh", "logistic"):
            ops["exp"] += neurons
        else:
            ops["cmp"] += neurons
    else:
        raise ValueError(tr("modelo sin estimación: {n}", "no estimate for model: {n}", n=name))
    return ops


def _classes(board: dict) -> list[tuple[str, str]]:
    """[(clave de ciclos, etiqueta)] a cubrir para esta placa."""
    if board.get("ai_tier") == "linux_runtime" or str(board.get("arch", "")).startswith("aarch64"):
        return [("linux", "linux")]
    fpu = board.get("fpu")
    if fpu is True:
        return [("fpu", "fpu")]
    if fpu is False:
        return [("soft", "soft")]
    return [("fpu", "fpu"), ("soft", "soft")]  # la ficha no confirma la FPU: se cubren ambos


def estimate_for_board(ops: dict[str, int], board: dict) -> dict[str, Any]:
    mhz = float(board.get("clock_mhz") or 0)
    if mhz <= 0:
        return {"board": board["id"], "name": board.get("name", board["id"]), "ms_lo": None}
    scen = _classes(board)
    cyc = [sum(ops[k] * CYCLES[c][k] for k in ("mac", "cmp", "exp")) for c, _ in scen]
    lo, hi = min(cyc), max(cyc) * SLOWDOWN
    return {
        "board": board["id"],
        "name": board.get("name", board["id"]),
        "clock_mhz": mhz,
        "fpu_known": board.get("fpu") is not None or len(scen) == 1,
        "cpu_class": "+".join(label for _, label in scen),
        "cycles_lo": int(lo),
        "cycles_hi": int(hi),
        "ms_lo": round(lo / (mhz * 1e3), 4),
        "ms_hi": round(hi / (mhz * 1e3), 4),
    }


def estimate_latency(
    run_dir: str | Path,
    model: str | None = None,
    board: str | None = None,
    target_ms: float | None = None,
) -> dict[str, Any]:
    import joblib

    run = Path(run_dir)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    recipe = json.loads((run / "recipe_fitted.json").read_text(encoding="utf-8"))
    enc = FeatureEncoder.from_dict(json.loads((run / "encoder.json").read_text(encoding="utf-8")))
    name = model or results["selection"]["best"]
    path = run / f"model_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(tr("no existe {f}", "{f} does not exist", f=path.name))
    est = joblib.load(path)
    spec = build_spec(Pipeline.from_list(recipe["cleaning"], fitted=True), enc)
    ops = count_ops(name, est, spec.n_features)
    boards = [
        b
        for b in load_boards()["boards"]
        if b.get("ai_tier") != "none" and (board in (None, "all") or b["id"] == board)
    ]
    if not boards:
        raise ValueError(tr("placa desconocida: {b}", "unknown board: {b}", b=board))
    rows = [r for r in (estimate_for_board(ops, b) for b in boards) if r["ms_lo"] is not None]
    rows.sort(key=lambda r: r["ms_hi"])
    for r in rows:
        r["meets_target"] = None if target_ms is None else bool(r["ms_hi"] <= target_ms)
    res = {
        "model": name,
        "ops": ops,
        "boards": rows,
        "target_ms": target_ms,
        "assumed_cycles": CYCLES,
        "slowdown": SLOWDOWN,
        "family": "latency",
        "disclaimer": tr(
            "ESTIMACIÓN, no medición: Ciclos por operación supuestos (no del fabricante), rango "
            "ideal..x{s}. Cronometra tm_predict en la placa para el valor real.",
            "ESTIMATE, not a measurement: Assumed cycles per operation (not from the vendor), "
            "range ideal..x{s}. Time tm_predict on the board for the real value.",
            s=int(SLOWDOWN),
        ),
    }
    res["summary"] = summary(res)
    return res


def summary(res: dict[str, Any]) -> str:
    o = res["ops"]
    lines = [
        tr(
            "Latencia estimada por predicción de {m} ({mac} MAC, {cmp} comparaciones, {exp} exp).",
            "Estimated latency per prediction of {m} ({mac} MAC, {cmp} comparisons, {exp} exp).",
            m=res["model"],
            mac=o["mac"],
            cmp=o["cmp"],
            exp=o["exp"],
        )
    ]
    for r in res["boards"][:6]:
        flag = "" if r["fpu_known"] else tr(" (FPU sin confirmar)", " (FPU unconfirmed)")
        lines.append(f"  {r['name']}: {r['ms_lo']:.3f} – {r['ms_hi']:.3f} ms{flag}")
    lines.append(res["disclaimer"])
    return "\n".join(lines)
