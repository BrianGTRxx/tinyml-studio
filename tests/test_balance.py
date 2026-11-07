"""Equilibrado por submuestreo: la clase grande se recorta, la pequeña no se toca."""

import numpy as np
import pandas as pd

from tinyml_studio.clean.prepare import prepare_splits
from tinyml_studio.split.balance import balance_split


def _df():
    n = 1050
    return pd.DataFrame(
        {
            "t": np.arange(n),
            "x": np.random.default_rng(0).normal(size=n),
            "y": ["b" if i % 20 == 0 else "a" for i in range(n)],  # 5 % de la clase pequeña
        }
    )


def test_big_class_cut_to_small_train_only():
    df = _df()
    idx = {"train": np.arange(0, 840), "val": np.arange(840, 945), "test": np.arange(945, 1050)}
    out, summary = balance_split(df, idx, "y", "train", "t")
    ytr = df["y"].to_numpy()[out["train"]]
    before, after = summary["train"]["b"]
    assert before == after  # la clase pequeña no se toca
    assert (ytr == "a").sum() == (ytr == "b").sum() == after
    assert len(out["val"]) == 105 and len(out["test"]) == 105  # real proportion kept
    a_rows = out["train"][ytr == "a"]
    assert a_rows.max() - a_rows.min() > 700  # spread over the whole recording, not just the start


def test_all_mode_and_off():
    df = _df()
    idx = {"train": np.arange(0, 840), "val": np.arange(840, 945), "test": np.arange(945, 1050)}
    out, _ = balance_split(df, idx, "y", "all", "t")
    for s in ("train", "val", "test"):
        y = df["y"].to_numpy()[out[s]]
        assert (y == "a").sum() == (y == "b").sum()
    same, summ = balance_split(df, idx, "y", "off")
    assert same is idx and summ == {}


def test_prepare_splits_uses_recipe_flag():
    df = _df()
    recipe = {
        "task": "classification",
        "target": "y",
        "roles": {"t": "time", "x": "feature", "y": "target"},
        "split": {"strategy": "random", "ratios": [0.8, 0.1, 0.1], "balance": "train"},
        "cleaning": [],
    }
    import tinyml_studio.clean.prepare as P

    P_apply = P.apply_recipe
    P.apply_recipe = lambda r: df
    try:
        prep = prepare_splits(recipe)
    finally:
        P.apply_recipe = P_apply
    vc = prep.raw["train"]["y"].value_counts()
    assert vc["a"] == vc["b"]
