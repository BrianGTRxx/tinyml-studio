"""Búsqueda de hiperparámetros con presupuesto de placa."""

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.dataset import inspect_files
from tinyml_studio.train.tune import tune

RNG = np.random.default_rng(5)


def _recipe(tmp_path):
    n = 600
    cls = RNG.integers(0, 3, n)
    df = pd.DataFrame(
        {
            "x1": RNG.normal(cls * 2.0, 1.2),
            "x2": RNG.normal(-cls * 1.5, 1.2),
            "x3": RNG.normal(0, 1, n),
            "clase": np.array(["a", "b", "c"])[cls],
        }
    )
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    rec = inspect_files([p])["recipe"]
    rec["task"] = "classification"
    rec["target"] = "clase"
    rec["label_from_file"] = {"enabled": False}
    rec["roles"] = {"x1": "feature", "x2": "feature", "x3": "feature", "clase": "target"}
    return rec


def test_budget_rejects_big_candidates_and_picks_a_fitting_one(tmp_path):
    rec = _recipe(tmp_path)
    free = tune(rec, "random_forest", n_trials=12, seed=1)
    biggest = max(r["flash_bytes"] for r in free["trials"] if "flash_bytes" in r)
    budget_kb = biggest / 1024 / 2
    res = tune(rec, "random_forest", flash_kb=budget_kb, n_trials=12, seed=1)
    assert res["best"]["fits_budget"] and res["best"]["flash_bytes"] <= budget_kb * 1024
    assert res["n_fitting"] < res["n_evaluated"]
    assert all(not r["fits"] for r in res["trials"] if r.get("flash_bytes", 0) > budget_kb * 1024)
    assert res["best"]["test_score"] is not None  # el test se mide solo para el ganador


def test_impossible_budget_is_reported(tmp_path):
    res = tune(_recipe(tmp_path), "decision_tree", flash_kb=0.01, n_trials=6)
    assert res["best"]["fits_budget"] is False and res["n_fitting"] == 0
    assert "NINGÚN" in res["summary"] or "NO candidate" in res["summary"]


def test_reproducible_and_default_always_evaluated(tmp_path):
    rec = _recipe(tmp_path)
    a = tune(rec, "decision_tree", n_trials=5, seed=7)
    b = tune(rec, "decision_tree", n_trials=5, seed=7)
    assert a["best"] == b["best"] and a["default"] is not None


def test_unknown_model_rejected(tmp_path):
    with pytest.raises(ValueError):
        tune(_recipe(tmp_path), "knn")
