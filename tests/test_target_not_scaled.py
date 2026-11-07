"""El objetivo de una regresión no se escala: el modelo y el firmware devuelven unidades reales."""

import numpy as np
import pandas as pd

from tinyml_studio.clean.prepare import cleaning_without_target, prepare_splits
from tinyml_studio.dataset import inspect_files
from tinyml_studio.train.run import train_models


def _recipe(tmp_path):
    rng = np.random.default_rng(2)
    n = 500
    x = rng.normal(size=n)
    pd.DataFrame(
        {"x": x, "z": rng.normal(size=n), "nota": 60 + 5 * x + rng.normal(0, 1, n)}
    ).to_csv(tmp_path / "d.csv", index=False)
    rec = inspect_files([tmp_path / "d.csv"])["recipe"]
    rec["task"], rec["target"] = "regression", "nota"
    rec["label_from_file"] = {"enabled": False}
    rec["roles"] = {"x": "feature", "z": "feature", "nota": "target"}
    # limpieza armada (o sugerida antes de cambiar el objetivo) que escala también el objetivo
    rec["cleaning"] = [
        {"kind": "scale", "params": {"columns": ["x", "z", "nota"], "method": "standard"}}
    ]
    return rec


def test_target_is_removed_from_scaling_and_stays_in_real_units(tmp_path):
    rec = _recipe(tmp_path)
    assert cleaning_without_target(rec)[0]["params"]["columns"] == ["x", "z"]
    prep = prepare_splits(rec, seed=0)
    assert 55 < prep.clean["train"]["nota"].mean() < 65  # unidades reales, no ~0
    assert abs(prep.clean["train"]["x"].mean()) < 0.2  # lo demás sí se escala
    res = train_models(rec, ["linear"], seed=0)
    assert res["models"]["linear"]["test"]["mae"] < 3  # error en puntos, no en desviaciones


def test_scale_step_with_only_the_target_disappears(tmp_path):
    rec = _recipe(tmp_path)
    rec["cleaning"] = [{"kind": "scale", "params": {"columns": ["nota"], "method": "standard"}}]
    assert cleaning_without_target(rec) == []
