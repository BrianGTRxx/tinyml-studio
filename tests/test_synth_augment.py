"""Aumento de datos: filas nuevas a partir de las existentes, solo en entrenamiento."""
# ruff: noqa: E501

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.augment.synth import synthesize
from tinyml_studio.clean.prepare import prepare_splits


def _cls():
    rng = np.random.default_rng(0)
    big = pd.DataFrame(
        {"a": rng.normal(10, 1, 300), "b": rng.normal(0, 1, 300), "txt": "x", "y": "big"}
    )
    small = pd.DataFrame(
        {"a": rng.normal(20, 1, 30), "b": rng.normal(5, 1, 30), "txt": "z", "y": "small"}
    )
    return pd.concat([big, small], ignore_index=True)


def test_off_changes_nothing():
    df = _cls()
    out, s = synthesize(df, ["a", "b"], "y", "classification", {"method": "off"})
    assert out is df and s["added"] == 0


def test_interpolate_balances_classes_and_stays_inside_the_class():
    df = _cls()
    out, s = synthesize(
        df, ["a", "b"], "y", "classification", {"method": "interpolate", "mode": "balance", "k": 5}
    )
    counts = out["y"].value_counts()
    assert counts["small"] == counts["big"] == 300
    new = out.iloc[len(df) :]
    assert set(new["y"]) == {"small"} and set(new["txt"]) == {"z"}  # clase y texto copiados
    lo, hi = df[df.y == "small"][["a", "b"]].min(), df[df.y == "small"][["a", "b"]].max()
    assert (new[["a", "b"]] >= lo - 1e-9).all().all() and (new[["a", "b"]] <= hi + 1e-9).all().all()
    assert s["added"] == 270 and s["per_class"] == {"small": 270}


def test_noise_ratio_adds_rows_near_the_originals():
    df = _cls()
    out, s = synthesize(
        df,
        ["a", "b"],
        "y",
        "classification",
        {"method": "noise", "ratio": 0.5, "noise_frac": 0.05},
        seed=1,
    )
    assert s["added"] == 165
    sd = df["a"].std()
    new = out.iloc[len(df) :]
    assert abs(new["a"].mean() - df["a"].mean()) < 3 * sd  # ruido pequeño: no se aleja de los datos
    assert out["y"].isin(["big", "small"]).all()


def test_regression_interpolates_the_target_between_two_real_values():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, 200)
    df = pd.DataFrame({"x": x, "y": 3 * x + 1})
    out, _ = synthesize(
        df, ["x"], "y", "regression", {"method": "interpolate", "ratio": 1.0, "k": 3}, seed=3
    )
    new = out.iloc[len(df) :]
    assert len(new) == 200
    assert np.allclose(new["y"], 3 * new["x"] + 1, atol=1e-6)  # la recta se conserva


def test_deterministic_and_cap():
    df = _cls()
    a, _ = synthesize(df, ["a"], "y", "classification", {"method": "noise", "ratio": 1.0}, seed=5)
    b, _ = synthesize(df, ["a"], "y", "classification", {"method": "noise", "ratio": 1.0}, seed=5)
    assert a.equals(b)
    c, s = synthesize(
        df,
        ["a"],
        "y",
        "classification",
        {"method": "noise", "ratio": 1.0, "max_synthetic": 50},
        seed=5,
    )
    assert s["added"] <= 50


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        synthesize(_cls(), ["a"], "y", "classification", {"method": "magia"})


def test_prepare_splits_augments_train_only(monkeypatch):
    import tinyml_studio.clean.prepare as P

    df = _cls().sample(frac=1.0, random_state=0).reset_index(drop=True)
    df["t"] = np.arange(len(df))
    recipe = {
        "task": "classification",
        "target": "y",
        "roles": {"t": "time", "a": "feature", "b": "feature", "y": "target", "txt": "meta"},
        "split": {"strategy": "random", "ratios": [0.8, 0.1, 0.1]},
        "augment": {"method": "interpolate", "mode": "balance"},
        "cleaning": [],
    }
    monkeypatch.setattr(P, "apply_recipe", lambda r: df)
    base = dict(recipe, augment={"method": "off"})
    p0, p1 = prepare_splits(base), prepare_splits(recipe)
    assert len(p1.raw["val"]) == len(p0.raw["val"]) and len(p1.raw["test"]) == len(p0.raw["test"])
    vc = p1.raw["train"]["y"].value_counts()
    assert vc["big"] == vc["small"] and len(p1.raw["train"]) > len(p0.raw["train"])
