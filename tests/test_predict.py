"""Probar un modelo entrenado con valores escritos a mano, con avisos de rango."""

import numpy as np
import pandas as pd

from tinyml_studio.dataset import inspect_files
from tinyml_studio.export.predict import describe, predict_values
from tinyml_studio.train import train_models

RNG = np.random.default_rng(5)


def _train(tmp_path, task="classification"):
    n = 600
    k = n // 3
    base = np.repeat([0.0, 5.0, 10.0], k)
    df = pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1.0, n),
            "f2": RNG.normal(0, 1, n) + base / 2,
            "tipo": RNG.choice(["u", "v"], n),
        }
    )
    if task == "classification":
        df["label"] = np.repeat(["a", "b", "c"], k)
    else:
        df["y"] = 2 * df["f1"] + df["f2"] + RNG.normal(0, 0.1, n)
    df.to_csv(tmp_path / "d.csv", index=False)
    rep = inspect_files([tmp_path / "d.csv"])
    recipe = rep["recipe"]
    if task == "regression":
        recipe["task"] = "regression"
        recipe["target"] = "y"
        recipe["roles"]["y"] = "target"
    train_models(
        recipe,
        ["decision_tree"] if task == "classification" else ["linear"],
        seed=0,
        out_dir=tmp_path / "run",
    )
    return tmp_path / "run"


def test_describe_lists_inputs_and_training_ranges(tmp_path):
    run = _train(tmp_path)
    d = describe(run)
    names = [c["name"] for c in d["numeric"]]
    assert names == ["f1", "f2"] and d["classes"] == ["a", "b", "c"]
    assert d["categorical"][0]["name"] == "tipo" and d["categorical"][0]["values"] == ["u", "v"]
    f1 = d["numeric"][0]
    assert f1["min"] < 0 < 12 < f1["max"] + 5 and f1["min"] <= f1["median"] <= f1["max"]
    assert (run / "input_ranges.json").exists()  # queda guardado para el C exportado


def test_predict_classifies_and_warns_out_of_range(tmp_path):
    run = _train(tmp_path)
    low = predict_values(run, {"f1": 0.2, "f2": 0.1, "tipo": "u"})
    high = predict_values(run, {"f1": 10.1, "f2": 5.0, "tipo": "v"})
    assert low["class"] == "a" and high["class"] == "c"
    assert abs(sum(low["probabilities"].values()) - 1.0) < 1e-9
    assert low["out_of_range"] == []
    far = predict_values(run, {"f1": 500, "f2": 0, "tipo": "u"})
    assert [w["column"] for w in far["out_of_range"]] == ["f1"]


def test_missing_and_unknown_values_are_reported(tmp_path):
    run = _train(tmp_path)
    r = predict_values(run, {"f1": "", "f2": "1,5", "tipo": "zzz"})  # coma decimal admitida
    assert "f1" in r["missing"] and r["unknown_category"] == ["tipo"]
    assert r["class"] in {"a", "b", "c"}


def test_regression_returns_a_value(tmp_path):
    run = _train(tmp_path, "regression")
    r = predict_values(run, {"f1": 3.0, "f2": 1.0, "tipo": "u"})
    assert abs(r["value"] - 7.0) < 0.5 and "class" not in r
