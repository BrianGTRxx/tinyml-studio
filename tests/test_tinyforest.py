"""Exportación compacta de bosques: exactitud, cuantización, C compilado y casos límite."""

import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tinyml_studio.export import export_run
from tinyml_studio.export import tinyforest as tf
from tinyml_studio.export import tinyforest_host as H
from tinyml_studio.export.cgen import emit_trees

needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="sin compilador C")
RNG = np.random.default_rng(5)

LOSSLESS = [
    dict(layout="dfs", dag=False),
    dict(layout="ref", dag=False),
    dict(layout="ref", dag=True),
    dict(layout="ref", dag=True, thr_table=False),
    dict(layout="ref", dag=True, thr_table=None),
    dict(layout="ref", dag=True, quant_input=True),
    dict(layout="dfs", dag=False, quant_input=True),
]


def _cls_data(n=600, f=6, k=3):
    X = RNG.normal(size=(n, f)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int) + (X[:, 2] > 0.7).astype(int) * (k > 2)
    y = np.clip(y, 0, k - 1)
    return X, y


def _probe(X):
    """Entradas pegadas a los umbrales: x == t y sus vecinos float32 (donde un error se nota)."""
    return X


def _near_thresholds(est, X):
    ths = []
    ms = est.estimators_ if hasattr(est, "estimators_") else [est]
    for m in ms:
        T = m.tree_
        for f, t in zip(T.feature, T.threshold, strict=True):
            if f >= 0:
                ths.append((f, np.float32(t)))
    rows = []
    for f, t in ths[:300]:
        for v in (t, np.nextafter(t, np.float32(np.inf)), np.nextafter(t, np.float32(-np.inf))):
            r = X[RNG.integers(len(X))].copy()
            r[f] = v
            rows.append(r)
    return np.vstack([X, np.array(rows, dtype=np.float32)])


@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("kind", ["rf", "et", "dt"])
def test_lossless_variants_match_sklearn_and_each_other(kind, k):
    X, y = _cls_data(k=k)
    est = {
        "rf": RandomForestClassifier(15, max_depth=8, random_state=0),
        "et": ExtraTreesClassifier(15, max_depth=8, random_state=0),
        "dt": DecisionTreeClassifier(max_depth=8, random_state=0),
    }[kind].fit(X, y)
    Xp = _near_thresholds(est, X)
    ref = None
    for kw in LOSSLESS:
        m = tf.build(est, **kw)
        p = m.predict(Xp)
        # entrada cuantizada y comparación float dan lo mismo (la cuantización no pierde nada)
        if m.thr_table:
            assert np.array_equal(m.predict(Xp, use_quant=True), m.predict(Xp, use_quant=False))
        ref = p if ref is None else ref
        assert np.array_equal(p, ref), kw
        assert (p == est.predict(Xp)).mean() >= 0.995, kw  # float32 vs float64: solo empates
    if kind == "dt":
        assert np.array_equal(ref, est.predict(Xp))  # un árbol: exacto


def test_nan_goes_right_like_sklearn_float32():
    X, y = _cls_data()
    est = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X, y)
    Xn = X.copy()
    Xn[::7, 0] = np.nan
    for kw in (dict(), dict(quant_input=True)):
        m = tf.build(est, **kw)
        exp = np.array([m.predict(Xn[i : i + 1])[0] for i in range(60)])
        assert np.array_equal(m.predict(Xn[:60], use_quant=True), exp)


@pytest.mark.parametrize("kind", ["rf", "dt"])
def test_regression_exact(kind):
    X = RNG.normal(size=(500, 5)).astype(np.float32)
    y = 3 * X[:, 0] + np.sin(X[:, 1]) + RNG.normal(0, 0.1, 500)
    est = (
        RandomForestRegressor(12, max_depth=7, random_state=0)
        if kind == "rf"
        else DecisionTreeRegressor(max_depth=7, random_state=0)
    ).fit(X, y)
    base = None
    for kw in LOSSLESS:
        p = tf.build(est, **kw).predict(X)
        base = p if base is None else base
        assert np.array_equal(p, base), kw
    assert np.allclose(base, est.predict(X), rtol=1e-5, atol=1e-5)
    q = tf.build(est, leaf="u16").predict(X)  # con pérdida, acotada por el paso de cuantización
    assert np.abs(q - est.predict(X)).max() < (y.max() - y.min()) / 65535 + 1e-4


def test_lossy_variants_report_small_delta():
    X, y = _cls_data(1500, 8, 3)
    est = RandomForestClassifier(30, max_depth=9, random_state=1).fit(X, y)
    sk = est.predict(X)
    for kw, tol in (
        (dict(leaf="u8"), 0.99),
        (dict(leaf="vote"), 0.95),
        (dict(max_bins=16), 0.9),
    ):
        assert (tf.build(est, **kw).predict(X) == sk).mean() >= tol, kw


def test_integer_inputs_exact_on_integer_data():
    X = RNG.integers(0, 17, size=(800, 10)).astype(np.float32)
    y = ((X[:, 0] + X[:, 3]) // 6) % 3
    est = RandomForestClassifier(20, max_depth=9, random_state=2).fit(X, y)
    a = tf.build(est, integer_inputs=True)
    b = tf.build(est)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert a.layout_arrays()["tf_thr"][0] == "uint8_t"


def test_edge_single_leaf_constant_feature_and_errors():
    X = np.ones((50, 3), dtype=np.float32)  # características constantes
    y = np.zeros(50, dtype=int)
    for est in (
        DecisionTreeClassifier().fit(X, y),
        RandomForestClassifier(5, random_state=0).fit(X, y),
    ):
        for kw in LOSSLESS + [dict(leaf="vote"), dict(leaf="u8")]:
            if isinstance(est, DecisionTreeClassifier) and kw.get("leaf"):
                continue
            m = tf.build(est, **kw)
            assert (m.predict(X) == 0).all()
            code, db, ram, info = m.to_c()
            assert db >= 1
    r = DecisionTreeRegressor().fit(X, np.full(50, 2.5))
    assert np.allclose(tf.build(r).predict(X), 2.5)
    with pytest.raises(ValueError):
        tf.build(RandomForestClassifier(3).fit(X, y), layout="dfs", dag=True)
    with pytest.raises(ValueError):
        tf.build(DecisionTreeRegressor().fit(X, np.ones(50)), leaf="u8")
    Y = np.stack([y, y], 1)
    with pytest.raises(ValueError):
        tf.build(DecisionTreeClassifier().fit(X, Y))


@needs_cc
@pytest.mark.parametrize("task", ["cls", "reg"])
def test_c_matches_numpy_reference_and_current_export(task):
    X, y = _cls_data(700, 7, 3)
    cls = task == "cls"
    if cls:
        est = RandomForestClassifier(20, max_depth=9, random_state=0).fit(X, y)
    else:
        est = RandomForestRegressor(20, max_depth=8, random_state=0).fit(X, X[:, 0] * 2 + X[:, 2])
    Xp = _near_thresholds(est, X)
    C = len(est.classes_) if cls else 1
    spec = SimpleNamespace(n_features=X.shape[1])
    code0, db0, _, _ = emit_trees(
        est, "random_forest", spec, "classification" if cls else "regression", C
    )
    base = H.run_host(code0, Xp, cls, reps=0)
    assert base["ok"] and base["warnings"] == []
    variants = (
        LOSSLESS + [dict(leaf="u8"), dict(leaf="vote")] if cls else LOSSLESS + [dict(leaf="u16")]
    )
    for kw in variants:
        m = tf.build(est, **kw)
        code, db, ram, _ = m.to_c()
        r = H.run_host(code, Xp, cls, reps=0)
        assert r["ok"] and r["warnings"] == [], (kw, r)
        ref = m.predict_index(Xp) if cls else m.predict(Xp)
        assert np.array_equal(r["pred"], ref.astype(r["pred"].dtype)), kw  # C == numpy, bit a bit
        if kw not in ({"leaf": "u8"}, {"leaf": "vote"}, {"leaf": "u16"}):
            assert np.array_equal(r["pred"], base["pred"]), kw  # == export actual
        assert db < db0, (kw, db, db0)  # estrictamente menor que el export actual
        assert db == m.data_bytes


@needs_cc
def test_integer_inputs_c_equivalence():
    X = RNG.integers(-5, 40, size=(600, 6)).astype(np.float32)
    y = ((X[:, 1] + X[:, 4]) // 15).astype(int) % 3
    est = ExtraTreesClassifier(10, max_depth=8, random_state=0).fit(X, y)
    m = tf.build(est, integer_inputs=True)
    assert m.layout_arrays()["tf_thr"][0] == "int8_t"
    r = H.run_host(m.to_c()[0], X, True, reps=0)
    assert r["ok"] and r["warnings"] == []
    assert np.array_equal(r["pred"], m.predict_index(X))
    assert np.array_equal(m.predict(X), tf.build(est).predict(X))


def _csv(path, n=600):
    k = n // 3
    base = np.repeat([0.0, 5.0, 10.0], k)
    pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1.0, n),
            "f2": RNG.normal(0, 1, n) + base / 2,
            "label": np.repeat(["a", "b", "c"], k),
        }
    ).to_csv(path, index=False)


@needs_cc
@pytest.mark.parametrize("model", ["decision_tree", "random_forest", "extra_trees"])
def test_export_run_compact_option(tmp_path, model):
    from tinyml_studio.dataset import inspect_files
    from tinyml_studio.train import train_models

    _csv(tmp_path / "d.csv")
    recipe = inspect_files([tmp_path / "d.csv"])["recipe"]
    train_models(recipe, [model], seed=0, out_dir=tmp_path / "run")
    plain = export_run(tmp_path / "run", tmp_path / "c0")
    comp = export_run(tmp_path / "run", tmp_path / "c1", compact=True)
    v = comp["verification"]
    assert v["compile_ok"] and v["compile_warnings"] == [] and v["compiles_as_cpp"] is True
    assert v["model_match"] == 1.0
    assert comp["compact"] is True and "compact" not in plain
    assert comp["data_bytes"] < plain["data_bytes"]
    # el modo por defecto no cambia: sigue usando las tablas tm_feat/tm_thr
    assert "tm_thr" in (tmp_path / "c0" / "tm_model.c").read_text()
