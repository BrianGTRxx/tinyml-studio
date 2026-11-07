
import numpy as np
import pandas as pd
import pytest

from tinyml_studio.profile import profile_dataframe
from tinyml_studio.split import split_temporal_blocks


def test_profile_nulls_and_balance():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "c": ["a", "a", "a", "b"]})
    rep = profile_dataframe(df, label_column="c")
    x = next(c for c in rep["columns"] if c["name"] == "x")
    assert x["nulls"] == 1 and x["nulls_pct"] == 25.0
    assert rep["class_balance"]["counts"] == {"a": 3, "b": 1}
    assert rep["class_balance"]["imbalance_ratio"] == 3.0
    with pytest.raises(KeyError):
        profile_dataframe(df, label_column="zzz")


def test_split_blocks_no_overlap_and_contiguous():
    t = pd.date_range("2026-01-01", periods=100, freq="s")
    df = pd.DataFrame({"g": ["a"] * 60 + ["b"] * 40, "t": t})
    s = split_temporal_blocks(df, group_column="g", time_column="t")
    allpos = np.concatenate(list(s.values()))
    assert sorted(allpos) == list(range(100))  # partición exacta, sin solapes
    for g_rows in (range(0, 60), range(60, 100)):
        tr = [i for i in s["train"] if i in g_rows]
        te = [i for i in s["test"] if i in g_rows]
        assert max(tr) < min(te)  # el test es posterior a train
    assert len(s["train"]) == 48 + 32 and len(s["val"]) == 6 + 4 and len(s["test"]) == 6 + 4


def test_split_bad_ratios():
    with pytest.raises(ValueError):
        split_temporal_blocks(pd.DataFrame({"a": [1]}), ratios=(0.5, 0.5, 0.5))
