"""Detección de anomalías: umbral sobre lo normal, detección medida solo para evaluar."""

import pytest
from test_tune import _recipe

from tinyml_studio.train.anomaly import anomaly_detection


@pytest.mark.parametrize("model", ["isolation_forest", "one_class_svm"])
def test_detects_other_classes_with_few_false_alarms(tmp_path, model):
    rec = _recipe(tmp_path)
    res = anomaly_detection(
        rec, model=model, normal_classes=["a"], false_alarm=0.05, out_dir=tmp_path / "o"
    )
    d = res["detection"]
    assert d["auc"] > 0.8 and d["rate"] > 0.3
    assert set(d["per_class"]) == {"b", "c"}
    assert res["test_false_alarm"] < 0.2
    assert (tmp_path / "o" / "anomaly_model.joblib").exists()
    assert "AUC" in res["summary"]


def test_without_anomalous_examples_only_false_alarms(tmp_path):
    res = anomaly_detection(_recipe(tmp_path), false_alarm=0.02)
    assert res["detection"] is None and res["test_false_alarm"] is not None
    assert "no se puede medir" in res["summary"] or "cannot be measured" in res["summary"]


def test_bad_inputs(tmp_path):
    rec = _recipe(tmp_path)
    with pytest.raises(ValueError):
        anomaly_detection(rec, normal_classes=["zzz"])
    with pytest.raises(ValueError):
        anomaly_detection(rec, false_alarm=0.9)
    with pytest.raises(ValueError):
        anomaly_detection(rec, model="nope")
