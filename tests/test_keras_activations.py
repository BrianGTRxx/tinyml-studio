import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

pytest.importorskip("tensorflow")

from tinyml_studio.api.handlers import KerasParams  # noqa: E402
from tinyml_studio.dataset import inspect_files  # noqa: E402
from tinyml_studio.export.tflite import export_tflite  # noqa: E402
from tinyml_studio.train.keras_mlp import train_keras_mlp  # noqa: E402

RNG = np.random.default_rng(7)


def _train(tmp, **kw):
    n, k = 900, 300
    base = np.repeat([0.0, 6.0, 12.0], k)
    pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1.0, n),
            "f2": RNG.normal(0, 1, n) + base / 2,
            "label": np.repeat(["a", "b", "c"], k),
        }
    ).to_csv(tmp / "d.csv", index=False)
    recipe = inspect_files([tmp / "d.csv"])["recipe"]
    kw = {"hidden": (8,), "epochs": 60, "batch_size": 32, "seed": 0, **kw}
    res = train_keras_mlp(recipe, out_dir=tmp / "run", **kw)
    return res, tmp / "run"


def test_default_architecture_unchanged():
    from tinyml_studio.train.keras_mlp import build_mlp

    m = build_mlp(4, 3, "classification", (8, 4))
    assert [type(layer).__name__ for layer in m.layers] == ["Dense"] * 3
    assert [layer.activation.__name__ for layer in m.layers[:2]] == ["relu", "relu"]


@pytest.mark.parametrize("act,op", [("tanh", "TANH"), ("sigmoid", "LOGISTIC")])
def test_activation_dropout_export(tmp_path, act, op):
    res, run = _train(tmp_path, activation=act, dropout=0.2, l2=1e-4)
    assert res["test"]["f1_macro"] > 0.85
    assert res["hyperparameters"]["activation"] == act and res["hyperparameters"]["dropout"] == 0.2
    rep = export_tflite(run, tmp_path / "x")
    assert op in rep["ops"] and "DROPOUT" not in "".join(rep["ops"])  # inactivo en inferencia
    assert rep["comparison"]["float_vs_keras_outputs"]["max_abs"] < 1e-4
    assert rep["metrics_test"]["tflite_int8"]["f1_macro"] > 0.8
    add = {"TANH": "AddTanh", "LOGISTIC": "AddLogistic"}[op]
    assert add in rep["tflm_resolver"]
    if rep["metrics_test"].get("tflm_int8"):  # runtime real de TFLM si está instalado
        assert rep["comparison"]["agreement_tflm_vs_tflite_int8"] >= 0.99


def test_invalid_params():
    assert KerasParams(recipe={}, activation="tanh", dropout=0.5).dropout == 0.5
    for bad in ({"activation": "gelu"}, {"dropout": 0.6}, {"l2": -1}):
        with pytest.raises(ValidationError):
            KerasParams(recipe={}, **bad)
