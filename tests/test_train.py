import json

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.dataset import inspect_files
from tinyml_studio.train import train_models
from tinyml_studio.train.features import FeatureEncoder
from tinyml_studio.train.metrics import classification_metrics, regression_metrics

RNG = np.random.default_rng(7)


def test_classification_metrics_known_values():
    y = np.array([0, 0, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 0])
    m = classification_metrics(y, p, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["b"]["recall"] == 1.0
    assert m["per_class"]["c"]["f1"] == 0.0
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_metrics_ignore_absent_classes_and_regression():
    m = classification_metrics(np.array([0, 0]), np.array([0, 0]), ["a", "b"])
    assert m["f1_macro"] == 1.0  # 'b' no tiene muestras: no penaliza
    r = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert r["mae"] == pytest.approx(1 / 3) and r["r2"] == pytest.approx(0.5)


def test_encoder_fit_train_only_roundtrip_unseen():
    tr = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan, 4.0], "c": ["a", "b", "a", "b"], "y": ["p", "q", "p", "q"]}
    )
    te = pd.DataFrame({"x": [np.nan, 9.0], "c": ["zzz", None], "y": ["p", "q"]})
    enc = FeatureEncoder.fit(tr, ["x", "c"], "y", "classification")
    assert enc.feature_names == ["x", "c=a", "c=b"] and enc.classes == ["p", "q"]
    X = enc.transform(te)
    assert X[0, 0] == 2.0  # mediana de train
    assert X[0, 1:].tolist() == [0.0, 0.0] and X[1, 1:].tolist() == [0.0, 0.0]  # desconocida
    enc2 = FeatureEncoder.from_dict(json.loads(json.dumps(enc.to_dict())))
    np.testing.assert_array_equal(enc.transform(te), enc2.transform(te))
    with pytest.raises(ValueError):
        enc.encode_target(pd.Series(["p", "nueva"]))


def test_encoder_skips_high_cardinality_and_all_null():
    tr = pd.DataFrame(
        {
            "hc": [f"v{i}" for i in range(50)],
            "n": [np.nan] * 50,
            "x": range(50),
            "y": ["a", "b"] * 25,
        }
    )
    enc = FeatureEncoder.fit(tr, ["hc", "n", "x"], "y", "classification")
    assert set(enc.skipped) == {"hc", "n"} and enc.numeric == ["x"]


def _separable_csv(path, n=450):
    k = n // 3
    cls = np.repeat(["a", "b", "c"], k)
    base = np.repeat([0.0, 5.0, 10.0], k)
    pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 0.5, n),
            "f2": RNG.normal(0, 1, n) + base / 2,
            "tipo": RNG.choice(["u", "v"], n),
            "label": cls,
        }
    ).to_csv(path, index=False)


def test_end_to_end_classification(tmp_path):
    _separable_csv(tmp_path / "d.csv")
    rep = inspect_files([tmp_path / "d.csv"])
    r = train_models(rep["recipe"], seed=0, out_dir=tmp_path / "run")
    assert r["task"] == "classification" and r["classes"] == ["a", "b", "c"]
    assert set(r["models"]) == {"decision_tree", "random_forest", "logreg", "svm", "mlp"}
    for m in r["models"].values():
        assert m["test"]["accuracy"] > 0.9
    assert all(r["baseline"][s]["f1_macro"] < 0.5 for s in ("val", "test"))
    # se elige con val, no con test
    best = max(r["models"], key=lambda n: r["models"][n]["val"]["f1_macro"])
    assert r["selection"] == {"metric": "f1_macro", "split": "val", "best": best}
    # nada que no sea feature entra al modelo
    assert not any(f.startswith(("label", "source")) for f in r["features_used"])
    saved = {p.name for p in (tmp_path / "run").iterdir()}
    assert {"results.json", "recipe_fitted.json", "encoder.json", f"model_{best}.joblib"} <= saved
    json.dumps(r)


def test_reproducible_and_exclude(tmp_path):
    _separable_csv(tmp_path / "d.csv")
    recipe = inspect_files([tmp_path / "d.csv"])["recipe"]
    a = train_models(recipe, ["random_forest", "decision_tree"], seed=3)
    b = train_models(recipe, ["random_forest", "decision_tree"], seed=3)
    for n in a["models"]:
        assert a["models"][n]["test"] == b["models"][n]["test"]
    assert a["reproducibility"]["recipe_sha256"] == b["reproducibility"]["recipe_sha256"]
    assert len(a["reproducibility"]["data_sha256"]["d.csv"]) == 64
    c = train_models(recipe, ["decision_tree"], exclude=("f1",))
    assert "f1" not in c["features_used"] and c["features_excluded_by_user"] == ["f1"]


def test_regression(tmp_path):
    n = 400
    x = RNG.normal(size=n)
    pd.DataFrame({"x1": x, "x2": RNG.normal(size=n), "SalePrice": 3 * x + 100}).to_csv(
        tmp_path / "h.csv", index=False
    )
    recipe = inspect_files([tmp_path / "h.csv"])["recipe"]
    r = train_models(recipe, ["linear", "random_forest"], seed=0)
    assert r["task"] == "regression" and r["selection"]["metric"] == "r2"
    assert r["models"]["linear"]["test"]["r2"] > 0.99


def test_unsupervised_recipe_refused(tmp_path):
    pd.DataFrame({"a": RNG.normal(size=100), "b": RNG.normal(size=100)}).to_csv(
        tmp_path / "s.csv", index=False
    )
    with pytest.raises(ValueError):
        train_models(inspect_files([tmp_path / "s.csv"])["recipe"])


def test_unknown_model_refused(tmp_path):
    _separable_csv(tmp_path / "d.csv")
    recipe = inspect_files([tmp_path / "d.csv"])["recipe"]
    with pytest.raises(ValueError):
        train_models(recipe, ["linear"])  # 'linear' es de regresión




