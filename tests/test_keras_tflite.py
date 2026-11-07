import json
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from tinyml_studio.dataset import inspect_files  # noqa: E402
from tinyml_studio.export.tflite import c_array, export_tflite  # noqa: E402
from tinyml_studio.train.keras_mlp import train_keras_mlp  # noqa: E402

HAS_TFLM = pytest.importorskip("tflite_micro", reason="sin tflite-micro") is not None
RNG = np.random.default_rng(5)


def _clf_csv(path, n=900):
    k = n // 3
    base = np.repeat([0.0, 6.0, 12.0], k)
    pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1.0, n),
            "f2": RNG.normal(0, 1, n) + base / 2,
            "tipo": RNG.choice(["u", "v"], n),
            "label": np.repeat(["a", "b", "c"], k),
        }
    ).to_csv(path, index=False)


def _train(tmp, builder=_clf_csv, **kw):
    builder(tmp / "d.csv")
    recipe = inspect_files([tmp / "d.csv"])["recipe"]
    kw = {"hidden": (8,), "epochs": 60, "batch_size": 32, "seed": 0, **kw}
    train_keras_mlp(recipe, out_dir=tmp / "run", **kw)
    return tmp / "run"


def test_c_array_roundtrip():
    data = bytes(range(40))
    src = c_array(data, "x")
    nums = [int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})", src)]
    assert bytes(nums) == data and "x_len = 40" in src and "aligned(16)" in src


def test_train_keras_results(tmp_path):
    run = _train(tmp_path)
    res = json.loads((run / "results.json").read_text())
    assert res["kind"] == "keras_mlp" and res["test"]["f1_macro"] > 0.85
    assert res["baseline"]["test"]["f1_macro"] < 0.5
    assert res["hyperparameters"]["epochs_run"] <= 60
    assert {"model.keras", "encoder.json", "recipe_fitted.json"} <= {p.name for p in run.iterdir()}


def test_export_float_io_verification(tmp_path):
    run = _train(tmp_path)
    rep = export_tflite(run, tmp_path / "x", flash_kb=64, ram_kb=32)
    out = tmp_path / "x"
    c = rep["comparison"]
    assert c["float_vs_keras_outputs"]["max_abs"] < 1e-4  # float tflite == Keras
    assert c["agreement_int8_vs_keras"] >= 0.95
    assert rep["metrics_test"]["tflite_int8"]["f1_macro"] > 0.8
    assert abs(rep["int8_f1_drop"]) < 0.1
    # (con modelos de juguete el int8 puede pesar más que el float: parámetros de cuantización y
    # ops Quantize/Dequantize; en el dataset real sí es menor, ver test_real_dataset_and_cli)
    assert rep["sizes_bytes"]["tflite_int8"] > 0
    assert "FULLY_CONNECTED" in rep["ops"] and "AddFullyConnected" in rep["tflm_resolver"]
    assert rep["arena"]["measured"] and 0 < rep["arena"]["arena_bytes"] < 20000
    assert rep["metrics_test"]["tflm_int8"]["f1_macro"] > 0.8
    assert c["agreement_tflm_vs_tflite_int8"] >= 0.99  # el runtime real de TFLM coincide
    assert rep["budget"]["flash_data"]["verdict"] == "cabe"
    # los bytes de model_data.h son exactamente el .tflite int8
    nums = [int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})", (out / "model_data.h").read_text())]
    assert bytes(nums) == (out / "model_int8.tflite").read_bytes()
    cfg = (out / "model_config.h").read_text()
    assert "TM_INT8_IO 0" in cfg and "TM_ARENA_SIZE" in cfg and "TM_N_OUTPUTS 3" in cfg
    ino = (out / "inference_example.ino").read_text()
    assert "ml.resolver.AddFullyConnected();" in ino and "tm_prepare(num, cat, x);" in ino
    meta = json.loads((out / "model_meta.json").read_text())
    assert meta["classes"] == ["a", "b", "c"] and meta["inputs_num"] == ["f1", "f2"]
    rows = (out / "test_vectors.csv").read_text().strip().splitlines()
    assert rows[0].endswith("expected_int8,expected_keras") and len(rows) > 50


def test_export_int8_io(tmp_path):
    run = _train(tmp_path)
    rep = export_tflite(run, tmp_path / "x", int8_io=True)
    assert rep["io"] == "int8"
    assert rep["metrics_test"]["tflite_int8"]["f1_macro"] > 0.8
    assert rep["metrics_test"]["tflm_int8"]["f1_macro"] > 0.8
    cfg = (tmp_path / "x" / "model_config.h").read_text()
    assert "TM_INT8_IO 1" in cfg and "TM_INPUT_SCALE" in cfg
    assert "int8 I/O" in (tmp_path / "x" / "inference_example.ino").read_text()


def test_export_regression(tmp_path):
    def csv(path, n=800):
        x = RNG.normal(size=n)
        pd.DataFrame(
            {"x1": x, "x2": RNG.normal(size=n), "SalePrice": 3 * x + 100 + RNG.normal(0, 0.1, n)}
        ).to_csv(path, index=False)

    run = _train(tmp_path, csv, epochs=60)
    rep = export_tflite(run, tmp_path / "x")
    assert rep["metrics_test"]["keras"]["r2"] > 0.8
    assert rep["metrics_test"]["tflite_int8"]["r2"] > 0.7
    assert rep["comparison"]["float_vs_keras_outputs"]["max_abs"] < 1e-3


def test_rejects_classical_run(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "results.json").write_text(json.dumps({"task": "classification"}))
    with pytest.raises(ValueError):
        export_tflite(tmp_path / "run", tmp_path / "x")


