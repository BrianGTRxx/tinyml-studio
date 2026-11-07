"""Rechazo de desconocido: umbral calibrado en validación y detección medida de verdad."""

import numpy as np
import pytest
from test_tune import _recipe

from tinyml_studio.train.reject import calibrate_rejection, pick_threshold
from tinyml_studio.train.run import train_models


def test_pick_threshold_meets_target_with_max_coverage():
    conf = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99] * 10)
    correct = conf >= 0.7  # lo dudoso falla
    r = pick_threshold(conf, correct, 0.99)
    assert r["reached_target"] and 0.69 < r["threshold"] <= 0.7
    assert r["accuracy_accepted"] == 1.0 and r["coverage"] == pytest.approx(5 / 7, abs=0.01)


def test_unreachable_target_is_reported():
    conf = np.linspace(0.4, 1, 100)
    r = pick_threshold(conf, np.zeros(100, bool), 0.9)
    assert r["reached_target"] is False


def test_calibration_and_holdout_on_three_classes(tmp_path):
    rec = _recipe(tmp_path)
    train_models(rec, ["random_forest"], seed=0, out_dir=tmp_path / "run")
    res = calibrate_rejection(tmp_path / "run", "random_forest", target_accuracy=0.9)
    assert res["val"]["accuracy_accepted"] >= 0.9 or not res["reached_target"]
    assert 0 < res["val"]["coverage"] <= 1 and res["test"]["coverage"] > 0
    u = res["unknown_detection"]
    assert u and len(u["classes"]) == 3 and 0 <= u["mean_rejected"] <= 1
    assert "desconocido" in res["summary"].lower() or "unknown" in res["summary"].lower()


def test_model_without_probabilities_is_refused(tmp_path):
    rec = _recipe(tmp_path)
    train_models(rec, ["linear_svm"], seed=0, out_dir=tmp_path / "run")
    with pytest.raises(ValueError):
        calibrate_rejection(tmp_path / "run", "linear_svm")


def test_onnx_export_answers_unknown_below_threshold(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")
    import importlib.util

    from tinyml_studio.export.onnx_export import export_onnx

    rec = _recipe(tmp_path)
    train_models(rec, ["random_forest"], seed=0, out_dir=tmp_path / "run")
    out = tmp_path / "onnx"
    export_onnx(tmp_path / "run", out, "random_forest", reject_below=0.9999)
    monkeypatch.chdir(out)
    spec = importlib.util.spec_from_file_location("infer_onnx", out / "infer_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    label, _ = mod.predict({"x1": 0.5, "x2": -0.7, "x3": 0.1})  # una mezcla dudosa
    assert label == "desconocido"
    label2, _ = mod.predict({"x1": 0.5, "x2": -0.7, "x3": 0.1}, reject_below=0.0001)
    assert label2 in ("a", "b", "c")
