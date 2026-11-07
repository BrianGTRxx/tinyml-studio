"""Banco ampliado de modelos: entrenamiento, complejidad, importancias y exportación a C."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.api.catalog import get_catalog
from tinyml_studio.dataset import inspect_files
from tinyml_studio.export import export_run
from tinyml_studio.export.cgen import EMITTERS
from tinyml_studio.train import models as M
from tinyml_studio.train import train_models

NEW_CLF = ["knn", "gaussian_nb", "extra_trees", "gradient_boosting", "lda", "linear_svm"]
NEW_REG = ["knn", "extra_trees", "gradient_boosting", "svr", "lasso", "elasticnet"]
HAS_CC = shutil.which("cc") is not None


def _clf(path, n=450, seed=3):
    rng = np.random.default_rng(seed)
    k = n // 3
    base = np.repeat([0.0, 6.0, 12.0], k)
    pd.DataFrame(
        {
            "f1": base + rng.normal(0, 1.0, n),
            "f2": rng.normal(0, 1, n) + base / 2,
            "tipo": rng.choice(["u", "v"], n),
            "label": np.repeat(["a", "b", "c"], k),
        }
    ).to_csv(path, index=False)


def _reg(path, n=500, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    pd.DataFrame(
        {"x1": x, "x2": rng.normal(size=n), "y": 3 * x + 10 + rng.normal(0, 0.1, n)}
    ).to_csv(path, index=False)


def _train(tmp_path, builder, models):
    builder(tmp_path / "d.csv")
    recipe = inspect_files([tmp_path / "d.csv"])["recipe"]
    res = train_models(recipe, models, seed=0, out_dir=tmp_path / "run")
    return res, tmp_path / "run"


@pytest.fixture(scope="module")
def clf_run(tmp_path_factory):
    return _train(tmp_path_factory.mktemp("clf"), _clf, NEW_CLF)


@pytest.fixture(scope="module")
def reg_run(tmp_path_factory):
    return _train(tmp_path_factory.mktemp("reg"), _reg, NEW_REG)


def test_catalog_lists_and_meta():
    cat = get_catalog()["classical"]
    for t, names in (("classification", NEW_CLF), ("regression", NEW_REG)):
        assert set(names) <= set(cat["models"][t])
        for n in cat["models"][t]:
            assert n in cat["defaults"]
            m = cat["meta"][n]
            assert m["family"] in {
                "tree",
                "linear",
                "kernel",
                "neighbors",
                "bayes",
                "ensemble",
                "neural",
            }
            assert isinstance(m["exportable_c"], bool) and isinstance(m["needs_scaling"], bool)
            assert m["note"]
    # la ficha coincide con los emisores reales
    for n, m in M.meta().items():
        assert m["exportable_c"] == (n in EMITTERS), n
    assert {p["id"] for p in get_catalog()["keras_mlp"]["presets"]} == {"small", "medium", "deep"}


def test_existing_defaults_untouched():
    assert M.DEFAULTS["svm"] == {"C": 1.0, "kernel": "rbf", "max_train_rows": 15000}
    assert M.DEFAULTS["random_forest"]["n_estimators"] == 50
    assert M.available("classification")[:5] == (
        "decision_tree",
        "random_forest",
        "logreg",
        "svm",
        "mlp",
    )


def test_classifiers_train(clf_run):
    res, _ = clf_run
    for n in NEW_CLF:
        r = res["models"][n]
        assert r["test"]["accuracy"] > 0.85, n
        assert r["complexity"] and all(v > 0 for v in r["complexity"].values()), n
        assert "error" not in r["quality"], (n, r["quality"])
        if n not in ("knn", "gradient_boosting"):
            assert r["top_features"], n


def test_regressors_train(reg_run):
    res, _ = reg_run
    for n in NEW_REG:
        r = res["models"][n]
        assert r["test"]["r2"] > 0.8, n
        assert r["complexity"] and all(v > 0 for v in r["complexity"].values()), n


def test_svr_row_cap(tmp_path):
    _reg(tmp_path / "d.csv", n=400)
    recipe = inspect_files([tmp_path / "d.csv"])["recipe"]
    res = train_models(recipe, ["svr"], seed=0, params={"svr": {"max_train_rows": 50}})
    assert any("SVR" in n for n in res["models"]["svr"]["notes"])


@pytest.mark.skipif(not HAS_CC, reason="sin compilador C")
@pytest.mark.parametrize("model", ["extra_trees", "gaussian_nb", "lda", "linear_svm"])
def test_export_classifiers(clf_run, tmp_path, model):
    _, run = clf_run
    rep = export_run(run, tmp_path / "c", model, flash_kb=512, ram_kb=64)
    v = rep["verification"]
    assert v["compile_ok"] and v["compile_warnings"] == [], v
    assert v["compiles_as_cpp"] is True
    assert v["features_max_rel_diff"] < 1e-5
    assert v["model_match"] >= (1.0 if model == "extra_trees" else 0.995), v
    assert v["raw_to_class_match"] >= 0.99
    meta = json.loads((tmp_path / "c" / "model_meta.json").read_text())
    assert meta["model"] == model


@pytest.mark.skipif(not HAS_CC, reason="sin compilador C")
@pytest.mark.parametrize("model", ["extra_trees", "lasso", "elasticnet"])
def test_export_regressors(reg_run, tmp_path, model):
    _, run = reg_run
    rep = export_run(run, tmp_path / "c", model)
    v = rep["verification"]
    assert v["compile_ok"] and v["compile_warnings"] == [], (model, v)
    assert v["model_match"] >= 0.99 and v["vs_full_python_pipeline"] >= 0.95, (model, v)


@pytest.mark.parametrize("model", ["knn", "gradient_boosting"])
def test_non_exportable_clear_error(clf_run, tmp_path, model):
    _, run = clf_run
    assert M.meta()[model]["exportable_c"] is False
    with pytest.raises(ValueError, match="(?i)cannot be exported|no se puede exportar"):
        export_run(run, tmp_path / "c", model)


def test_svr_not_exportable(reg_run, tmp_path):
    _, run = reg_run
    with pytest.raises(ValueError, match="(?i)cannot be exported|no se puede exportar"):
        export_run(run, tmp_path / "c", "svr")
