"""Reutilizar una receta en otro dataset: emparejado de columnas, notas y sin mutar entradas."""

import copy

import numpy as np
import pandas as pd

from tinyml_studio.dataset import inspect_files
from tinyml_studio.dataset.transfer import transfer_recipe

RNG = np.random.default_rng(5)


def make(path, n=300, **rename):
    base = np.repeat([0.0, 5.0, 10.0], n // 3)
    df = pd.DataFrame(
        {
            "temp": base + RNG.normal(0, 1, n),
            "hum": RNG.normal(size=n) + base / 2,
            "volt": RNG.normal(3.3, 0.1, n),
            "label": np.repeat(["a", "b", "c"], n // 3),
        }
    )
    df = df.rename(columns=rename)
    df.to_csv(path, index=False)
    return path


def old_recipe(tmp_path):
    rep = inspect_files([make(tmp_path / "old.csv")])
    r = rep["recipe"]
    r["encoding"]["overrides"] = {"label": "ordinal"}
    r["split"] = {**r["split"], "strategy": "random", "ratios": [0.7, 0.15, 0.15]}
    r["cleaning"] = [
        {
            "kind": "impute",
            "params": {"columns": ["temp", "hum"], "strategy": "mean"},
            "why": "x",
            "state": {"stats": {"temp": 1.0}},
        },
        {
            "kind": "scale",
            "params": {"columns": ["temp", "hum", "volt"], "method": "minmax"},
            "why": "y",
            "state": {"scale": {"temp": {"center": 1, "scale": 2}}},
        },
    ]
    return r


def test_same_schema_other_file(tmp_path):
    old = old_recipe(tmp_path)
    new_rep = inspect_files([make(tmp_path / "new.csv")])
    snap_old, snap_rep = copy.deepcopy(old), copy.deepcopy(new_rep)
    out = transfer_recipe(old, new_rep)
    assert old == snap_old and new_rep == snap_rep  # no muta entradas
    r = out["recipe"]
    assert r["files"] == new_rep["recipe"]["files"] and r["target"] == "label"
    assert r["split"]["strategy"] == "random" and r["split"]["ratios"] == [0.7, 0.15, 0.15]
    assert r["encoding"]["overrides"] == {"label": "ordinal"}
    sc = next(s for s in r["cleaning"] if s["kind"] == "scale")
    assert sc["params"]["method"] == "minmax" and sc["state"] == {}
    assert all(s["state"] == {} for s in r["cleaning"])
    assert out["applied"]["matched_exact"] >= 4 and out["applied"]["columns_missing"] == 0
    assert out["applied"]["target"] == 1


def test_renamed_columns_by_case_and_underscore(tmp_path):
    old = old_recipe(tmp_path)
    new_rep = inspect_files([make(tmp_path / "n.csv", temp="Temp", hum="HUM", volt="Volt")])
    out = transfer_recipe(old, new_rep)
    # el motor ya normaliza al cargar; aquí la receta antigua trae nombres distintos
    old2 = copy.deepcopy(old)
    old2["cleaning"][1]["params"]["columns"] = ["TEMP", "HUM", "VOLT"]
    old2["roles"] = {k.upper() if k != "source" else k: v for k, v in old2["roles"].items()}
    old2["cleaning"][0]["params"]["columns"] = ["TEMP", "HUM"]
    old2["encoding"]["overrides"] = {"LABEL": "ordinal"}
    old2["target"] = "LABEL"
    out2 = transfer_recipe(old2, new_rep)
    sc = next(s for s in out2["recipe"]["cleaning"] if s["kind"] == "scale")
    assert sc["params"]["columns"] == ["hum", "temp", "volt"]
    assert out2["recipe"]["target"] == "label"
    assert out2["applied"]["matched_renamed"] == 4 and out2["applied"]["columns_missing"] == 0
    assert any("equivalente" in n or "equivalent" in n for n in out2["notes"])
    assert out["recipe"]["target"] == "label"


def test_token_match_reordered_words(tmp_path):
    old = old_recipe(tmp_path)
    old["roles"]["temp_c"] = old["roles"].pop("temp")
    old["cleaning"][1]["params"]["columns"] = ["hum", "temp_c", "volt"]
    new_rep = inspect_files([make(tmp_path / "n.csv", temp="c_temp")])
    out = transfer_recipe(old, new_rep)
    sc = next(s for s in out["recipe"]["cleaning"] if s["kind"] == "scale")
    assert "c_temp" in sc["params"]["columns"] and out["applied"]["matched_renamed"] >= 1
    assert out["recipe"]["roles"]["c_temp"] == "feature"


def test_missing_target_keeps_new_suggestion_and_warns(tmp_path):
    old = old_recipe(tmp_path)
    old["target"] = "clase"
    new_rep = inspect_files([make(tmp_path / "n.csv")])
    out = transfer_recipe(old, new_rep)
    assert out["recipe"]["target"] == new_rep["recipe"]["target"]
    assert out["applied"]["target"] == 0
    assert any("clase" in n and ("ATENCI" in n or "WARNING" in n) for n in out["notes"])


def test_missing_cleaning_column_adjusts_or_drops_step(tmp_path):
    old = old_recipe(tmp_path)
    old["cleaning"][0]["params"]["columns"] = ["pressure"]  # solo existe en el antiguo
    new_rep = inspect_files([make(tmp_path / "n.csv", volt="voltage")])
    out = transfer_recipe(old, new_rep)
    kinds = [s["kind"] for s in out["recipe"]["cleaning"]]
    assert "impute" not in kinds or all(
        "pressure" not in s["params"].get("columns", []) for s in out["recipe"]["cleaning"]
    )
    sc = next(s for s in out["recipe"]["cleaning"] if s["kind"] == "scale")
    assert "volt" not in sc["params"]["columns"] and "voltage" in sc["params"]["columns"]
    assert out["applied"]["cleaning_dropped"] >= 1 and out["applied"]["columns_missing"] >= 2
    assert any("pressure" in n for n in out["notes"])


def test_extra_column_gets_new_suggestions_and_note(tmp_path):
    old = old_recipe(tmp_path)
    p = make(tmp_path / "n.csv")
    df = pd.read_csv(p)
    df["extra"] = RNG.normal(size=len(df))
    df.to_csv(p, index=False)
    out = transfer_recipe(old, inspect_files([p]))
    assert out["applied"]["columns_new"] == 1
    assert out["recipe"]["roles"]["extra"] == "feature"
    assert any("extra" in n for n in out["notes"])
    sc = next(s for s in out["recipe"]["cleaning"] if s["kind"] == "scale")
    assert "extra" in sc["params"]["columns"] and sc["params"]["method"] == "minmax"


def test_kind_mismatch_numeric_became_text(tmp_path):
    old = old_recipe(tmp_path)
    p = make(tmp_path / "n.csv")
    df = pd.read_csv(p)
    df["volt"] = np.array(["alto", "medio", "bajo", "n/a"])[np.arange(len(df)) % 4]
    df.to_csv(p, index=False)
    out = transfer_recipe(old, inspect_files([p]))
    sc = next(s for s in out["recipe"]["cleaning"] if s["kind"] == "scale")
    assert "volt" not in sc["params"]["columns"]
    assert any("volt" in n for n in out["notes"])


def test_classification_target_now_continuous_switches_task(tmp_path):
    old = old_recipe(tmp_path)
    p = make(tmp_path / "n.csv")
    df = pd.read_csv(p)
    df["label"] = RNG.normal(size=len(df))
    df.to_csv(p, index=False)
    out = transfer_recipe(old, inspect_files([p]))
    assert out["recipe"]["target"] == "label" and out["recipe"]["task"] == "regression"


def test_label_from_file_carries_over(tmp_path):
    from conftest import three_files

    old = three_files(tmp_path)
    assert old["label_from_file"]["enabled"]
    d2 = tmp_path / "d2"
    d2.mkdir()
    three_files(d2)
    (d2 / "c.csv").rename(d2 / "z.csv")
    old["label_from_file"]["mapping"]["a.csv"] = "motor_a"
    out = transfer_recipe(old, inspect_files([d2]))
    m = out["recipe"]["label_from_file"]["mapping"]
    assert m["a.csv"] == "motor_a" and m["b.csv"] == "b" and m["z.csv"] == "z"
    assert out["recipe"]["target"] == "label" and out["recipe"]["roles"]["label"] == "target"
    assert any("z.csv" in n for n in out["notes"])


def test_group_split_without_group_column_falls_back(tmp_path):
    old = old_recipe(tmp_path)
    old["split"] = {
        "strategy": "group",
        "group_column": "device",
        "time_column": None,
        "ratios": [0.6, 0.2, 0.2],
    }
    out = transfer_recipe(old, inspect_files([make(tmp_path / "n.csv")]))
    assert out["recipe"]["split"]["strategy"] in ("stratified", "random")
    assert out["recipe"]["split"]["ratios"] == [0.6, 0.2, 0.2]
    assert any("device" in n for n in out["notes"])




def test_new_dataset_nulls_get_imputation_added(tmp_path):
    old = old_recipe(tmp_path)
    old["cleaning"] = [s for s in old["cleaning"] if s["kind"] == "scale"]
    p = make(tmp_path / "n.csv")
    df = pd.read_csv(p)
    df.loc[::7, "hum"] = np.nan
    df.to_csv(p, index=False)
    out = transfer_recipe(old, inspect_files([p]))
    kinds = [s["kind"] for s in out["recipe"]["cleaning"]]
    assert "impute" in kinds and kinds.index("impute") < kinds.index("scale")
    assert out["applied"]["cleaning_added"] >= 1
