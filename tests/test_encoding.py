import json
import shutil

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.dataset import inspect_files
from tinyml_studio.export import export_run
from tinyml_studio.train import train_models
from tinyml_studio.train.features import FeatureEncoder, encoding_advice, normalize_encoding

TRAIN = pd.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "c": ["b", "a", "c", "b", "b", "a"],
        "y": [0, 1, 0, 1, 0, 1],
    }
)
NEW = pd.DataFrame({"x": [1.0] * 4, "c": ["a", "b", "zzz", None], "y": [0] * 4})


def _fit(method, **kw):
    return FeatureEncoder.fit(TRAIN, ["x", "c"], "y", "regression", {"default": method, **kw})


def test_onehot_is_default_and_unchanged():
    enc = FeatureEncoder.fit(TRAIN, ["x", "c"], "y", "regression")
    assert enc.feature_names == ["x", "c=a", "c=b", "c=c"]
    X = enc.transform(NEW)
    assert X[:, 1:].tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]]


def test_dummy_drops_reference_and_unknown_is_reference():
    enc = _fit("dummy")
    assert enc.feature_names == ["x", "c=b", "c=c"]
    X = enc.transform(NEW)
    assert X[:, 1:].tolist() == [[0, 0], [1, 0], [0, 0], [0, 0]]  # a, zzz, NaN -> ceros


def test_ordinal_single_column_unknown_and_missing():
    enc = _fit("ordinal")
    assert enc.feature_names == ["x", "c#ordinal"]
    assert enc.mode_index["c"] == 1  # 'b' es la moda de train
    assert enc.transform(NEW)[:, 1].tolist() == [0, 1, -1, 1]


def test_overrides_and_max_categories():
    enc = _fit("onehot", overrides={"c": "ordinal"})
    assert enc.method("c") == "ordinal"
    enc = FeatureEncoder.fit(TRAIN, ["c"], "y", "regression", {"max_categories": 2})
    assert "c" in enc.skipped


@pytest.mark.parametrize(
    "bad",
    [
        {"default": "target"},
        {"overrides": {"c": "x"}},
        {"max_categories": 0},
        {"max_categories": "5"},
        {"foo": 1},
        "onehot",
    ],
)
def test_invalid_encoding_rejected(bad):
    with pytest.raises(ValueError):
        normalize_encoding(bad)
    assert normalize_encoding(None) == {
        "default": "onehot",
        "overrides": {},
        "max_categories": 20,
    }


@pytest.mark.parametrize("method", ["onehot", "dummy", "ordinal"])
def test_json_roundtrip(method):
    enc = _fit(method)
    enc2 = FeatureEncoder.from_dict(json.loads(json.dumps(enc.to_dict())))
    assert enc2.feature_names == enc.feature_names
    np.testing.assert_array_equal(enc2.transform(NEW), enc.transform(NEW))


def test_old_format_encoder_loads_as_onehot():
    d = _fit("dummy").to_dict()
    for k in ("methods", "mode_index"):
        d.pop(k)
    enc = FeatureEncoder.from_dict(d)
    assert enc.method("c") == "onehot" and enc.feature_names == ["x", "c=a", "c=b", "c=c"]


def test_no_leakage_categories_only_from_train():
    enc = _fit("ordinal")
    assert enc.categorical["c"] == ["a", "b", "c"]
    enc.transform(pd.DataFrame({"x": [1.0], "c": ["new"]}))
    assert enc.categorical["c"] == ["a", "b", "c"]  # transform no aprende


def test_advice():
    assert encoding_advice({"encoding": {"default": "ordinal"}}, "logreg", {"c": ["a"]})
    assert encoding_advice({"default": "ordinal"}, "random_forest", {"c": ["a"]}) == []
    assert encoding_advice({"default": "dummy"}, "decision_tree", {"c": ["a", "b"]}) == []
    wide = {"c": [str(i) for i in range(15)]}
    assert encoding_advice({}, "mlp", wide)  # onehot ancho
    assert encoding_advice(_fit("ordinal"), "knn")


# ------------------------------------------------------------ Python <-> C
pytestmark_c = pytest.mark.skipif(shutil.which("cc") is None, reason="sin compilador C")


def _csv(path, n=600):
    rng = np.random.default_rng(5)
    t = rng.choice(["u", "v", "w", "x"], n)
    f1 = rng.normal(size=n)
    label = np.where((t == "v") | (f1 > 0.8), "pos", "neg")
    pd.DataFrame({"f1": f1, "tipo": t, "label": label}).to_csv(path, index=False)


@pytestmark_c
@pytest.mark.parametrize("method", ["dummy", "ordinal"])
@pytest.mark.parametrize("model", ["decision_tree", "logreg"])
def test_export_matches_python(tmp_path, method, model):
    _csv(tmp_path / "d.csv")
    recipe = inspect_files([tmp_path / "d.csv"])["recipe"]
    assert recipe["encoding"]["default"] == "onehot"
    recipe["encoding"] = {"default": method}
    res = train_models(recipe, [model], seed=0, out_dir=tmp_path / "run")
    assert any("tipo" in f for f in res["features_used"])
    if method == "ordinal" and model == "logreg":
        assert any("ordinal" in w for w in res["warnings"])
    rep = export_run(tmp_path / "run", tmp_path / "c")
    v = rep["verification"]
    assert v["compile_ok"] and v["compile_warnings"] == [], v
    assert v["features_max_rel_diff"] < 1e-5
    assert v["model_match"] >= (1.0 if model == "decision_tree" else 0.995)
    meta = json.loads((tmp_path / "c" / "model_meta.json").read_text())
    assert meta["inputs"]["cat"]["tipo"]["encoding"] == method
    expected = (
        ["f1", "tipo#ordinal"] if method == "ordinal" else ["f1", "tipo=v", "tipo=w", "tipo=x"]
    )
    assert meta["features"] == expected
