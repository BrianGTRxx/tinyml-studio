import json

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.clean import Pipeline, prepare_splits
from tinyml_studio.clean.compare import compare_stats, figure, histogram_data
from tinyml_studio.clean.steps import (
    AbsurdToNaN,
    DropDuplicates,
    FilterRule,
    Impute,
    OutlierHandling,
    ReplaceValues,
    Scale,
)
from tinyml_studio.dataset import inspect_files
from tinyml_studio.split import make_split

RNG = np.random.default_rng(1)


def frame(n=300, **extra):
    return pd.DataFrame({"x": RNG.normal(10, 1, n), "src": ["a"] * n, **extra})


def test_replace_values():
    df = pd.DataFrame({"x": [1.0, -999.0, 3.0]})
    out = ReplaceValues(columns=["x"], values=[-999.0]).transform(df)
    assert out["x"].isna().tolist() == [False, True, False]


def test_absurd_bounds_fit_only_on_train_and_per_group():
    tr = pd.DataFrame(
        {
            "x": np.r_[RNG.normal(10, 1, 200), RNG.normal(1000, 5, 200)],
            "g": ["a"] * 200 + ["b"] * 200,
        }
    )
    s = AbsurdToNaN(columns=["x"], factor=20, by="g").fit(tr)
    te = pd.DataFrame({"x": [10.0, 1000.0, 1000.0, 10.0], "g": ["a", "b", "a", "b"]})
    out = s.transform(te)
    # 1000 es normal en el grupo b pero absurdo en a; 10 es absurdo en b
    assert out["x"].isna().tolist() == [False, False, True, True]
    # los límites no cambian al ver datos de test
    before = json.dumps(s.state, sort_keys=True)
    s.transform(te)
    assert json.dumps(s.state, sort_keys=True) == before


def test_impute_uses_train_statistics():
    tr = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    va = pd.DataFrame({"x": [100.0, 200.0, np.nan]})
    s = Impute(columns=["x"], strategy="median").fit(tr)
    assert s.transform(va)["x"].iloc[2] == 2.0  # mediana de train, no de val


def test_impute_int_column_gets_fractional_median():
    tr = pd.DataFrame({"x": pd.array([1, 2, None, 4], dtype="Int64")})
    out = Impute(columns=["x"], strategy="median").fit(tr).transform(tr)
    assert out["x"].isna().sum() == 0


def test_impute_mode_categorical_and_ffill():
    tr = pd.DataFrame({"c": ["a", "a", "b", None]})
    assert Impute(columns=["c"], strategy="mode").fit(tr).transform(tr)["c"].iloc[3] == "a"
    t2 = pd.DataFrame({"x": [1.0, np.nan, np.nan, 5.0, np.nan]})
    out = Impute(columns=["x"], strategy="ffill").fit(t2).transform(t2)
    assert out["x"].tolist()[:3] == [1.0, 1.0, 1.0]


def test_scale_and_outliers_and_filter():
    tr = pd.DataFrame({"x": RNG.normal(50, 5, 500)})
    z = Scale(columns=["x"], method="standard").fit(tr).transform(tr)["x"]
    assert abs(z.mean()) < 1e-9 and z.std(ddof=0) == pytest.approx(1.0)
    mm = Scale(columns=["x"], method="minmax").fit(tr).transform(tr)["x"]
    assert mm.min() == pytest.approx(0) and mm.max() == pytest.approx(1)
    o = OutlierHandling(columns=["x"], method="iqr", k=1.5, action="clip").fit(tr)
    assert o.transform(pd.DataFrame({"x": [1e6]}))["x"].iloc[0] < 100
    f = FilterRule(column="x", op=">", value=50)
    assert (f.transform(tr)["x"] > 50).all()
    with pytest.raises(ValueError):
        FilterRule(column="x", op="~", value=1).transform(tr)


def test_drop_duplicates_only_train():
    df = pd.DataFrame({"x": [1, 1, 2], "source": ["a", "b", "a"]})
    p = Pipeline([DropDuplicates(ignore=["source"])]).fit(df)
    assert len(p.transform(df, is_train=True)) == 2
    assert len(p.transform(df, is_train=False)) == 3


def test_pipeline_roundtrip_and_unfitted_guard():
    tr = frame(200)
    tr.loc[3, "x"] = np.nan
    p = Pipeline([Impute(columns=["x"]), Scale(columns=["x"])]).fit(tr)
    items = json.loads(json.dumps(p.to_list()))
    q = Pipeline.from_list(items, fitted=True)
    pd.testing.assert_frame_equal(p.transform(tr), q.transform(tr))
    with pytest.raises(RuntimeError):
        Pipeline.from_list(items).transform(tr)
    with pytest.raises(ValueError):
        Pipeline.from_list([{"kind": "nope", "params": {}}])


@pytest.mark.parametrize(
    "strategy,extra",
    [
        ("random", {}),
        ("stratified", {}),
        ("group", {"group_column": "g"}),
        ("temporal_blocks", {"time_column": "t", "group_column": "g"}),
    ],
)
def test_split_strategies_partition_exactly(strategy, extra):
    n = 300
    df = pd.DataFrame(
        {
            "y": RNG.integers(0, 3, n),
            "g": np.repeat(list("abcdefghij"), 30),
            "t": pd.date_range("2026-01-01", periods=n, freq="s"),
        }
    )
    idx = make_split(df, {"strategy": strategy, **extra}, "y", seed=0)
    allpos = np.concatenate(list(idx.values()))
    assert sorted(allpos) == list(range(n))
    assert all(len(v) > 0 for v in idx.values())
    if strategy == "group":
        gs = {k: set(df.loc[v, "g"]) for k, v in idx.items()}
        assert not (gs["train"] & gs["val"]) and not (gs["train"] & gs["test"])
    if strategy == "stratified":
        for k in idx.values():
            assert set(df.loc[k, "y"]) == {0, 1, 2}
    # reproducible
    idx2 = make_split(df, {"strategy": strategy, **extra}, "y", seed=0)
    assert all((idx[k] == idx2[k]).all() for k in idx)


def test_split_group_needs_three_groups():
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5})
    with pytest.raises(ValueError):
        make_split(df, {"strategy": "group", "group_column": "g"}, None)


def _dirty_csv(path):
    n = 600
    x = RNG.normal(10, 1, n)
    x[5] = 9_000_000.0
    s = RNG.normal(5, 1, n)
    s[[10, 20, 30]] = -999
    s[100:110] = np.nan
    pd.DataFrame(
        {"x": x, "s": s, "z": RNG.normal(size=n), "label": RNG.choice(["a", "b"], n)}
    ).to_csv(path, index=False)


def test_guided_cleaning_end_to_end(tmp_path):
    _dirty_csv(tmp_path / "d.csv")
    rep = inspect_files([tmp_path / "d.csv"])
    kinds = [s["kind"] for s in rep["recipe"]["cleaning"]]
    assert kinds[:2] == ["replace_values", "absurd_to_nan"] and "impute" in kinds
    assert kinds[-1] == "scale"
    assert all(s["why"] for s in rep["recipe"]["cleaning"])
    P = prepare_splits(rep["recipe"])
    assert set(P.clean) == {"train", "val", "test"}
    for df in P.clean.values():
        assert df[["x", "s", "z"]].isna().sum().sum() == 0
    assert pd.concat(P.raw.values())["x"].max() > 1e6  # el absurdo existía antes de limpiar
    allx = pd.concat(P.clean.values())["x"]
    assert allx.abs().max() < 20  # el absurdo ya no está (valores escalados)
    # el escalador se ajustó solo con train
    assert abs(P.clean["train"]["z"].mean()) < 1e-6
    assert abs(P.clean["val"]["z"].mean()) > 1e-9
    # estadísticos y gráficas
    stats = compare_stats(P.raw["train"], P.clean["train"], ["x", "s"])
    sx = next(r for r in stats if r["column"] == "x")
    assert sx["nulls_after"] == 0
    fig = figure(P.raw["train"], P.clean["train"], ["x", "s", "z"])
    assert len(fig.data) == 6
    json.dumps(P.pipeline.to_list())


def test_histogram_counts_out_of_range():
    b = pd.Series(np.r_[RNG.normal(0, 1, 500), [1e6]])
    a = pd.Series(RNG.normal(0, 1, 500))
    h = histogram_data(b, a)
    assert h["outside_before"] == 1 and sum(h["after"]) == 500 and len(h["edges"]) == 41




def test_cli_clean(tmp_path, dataset_dir):
    from tinyml_studio.cli import main

    _dirty_csv(tmp_path / "d.csv")
    rep = inspect_files([tmp_path / "d.csv"])
    (tmp_path / "r.json").write_text(json.dumps(rep["recipe"]))
    out = tmp_path / "out"
    assert main(["clean", str(tmp_path / "r.json"), "--out-dir", str(out), "--html"]) == 0
    assert {
        "train.csv",
        "val.csv",
        "test.csv",
        "recipe_fitted.json",
        "before_after_stats.json",
        "before_after.html",
    } <= {p.name for p in out.iterdir()}
    fitted = json.loads((out / "recipe_fitted.json").read_text())
    assert any(s["state"] for s in fitted["cleaning"])


def test_unit_view_compares_in_original_units(tmp_path):
    from tinyml_studio.clean.compare import numeric_columns, unit_view

    _dirty_csv(tmp_path / "d.csv")
    rep = inspect_files([tmp_path / "d.csv"])
    P = prepare_splits(rep["recipe"])
    before, after = unit_view(P)
    st = {r["column"]: r for r in compare_stats(before, after, ["x", "s"])}
    assert st["x"]["max_before"] > 1e6 and st["x"]["max_after"] < 100  # absurdo fuera
    assert 5 < st["x"]["mean_after"] < 15  # sigue en unidades originales (no z-score)
    assert st["s"]["nulls_after"] == 0 and st["s"]["nulls_before"] > 0
    assert numeric_columns(before, ["x", "label", "nope"]) == ["x"]
    h = histogram_data(before["x"], after["x"])
    assert h["outside_before"] >= 1
