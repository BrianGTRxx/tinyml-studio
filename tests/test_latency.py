"""Estimación de latencia: monotonía, FPU y honestidad (es un rango, no una medición)."""

import pytest
from test_tune import _recipe

from tinyml_studio.export.latency import estimate_latency
from tinyml_studio.train.run import train_models


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("lat")
    rec = _recipe(d)
    train_models(
        rec,
        ["decision_tree", "random_forest", "logreg"],
        seed=0,
        out_dir=d / "run",
        params={"random_forest": {"n_estimators": 40, "max_depth": 8}},
    )
    return d / "run"


def test_forest_slower_than_tree_and_range_is_ordered(run):
    tree = estimate_latency(run, "decision_tree", "esp32-s3")["boards"][0]
    forest = estimate_latency(run, "random_forest", "esp32-s3")["boards"][0]
    assert 0 < tree["ms_lo"] < tree["ms_hi"] and forest["ms_hi"] > tree["ms_hi"]
    assert forest["fpu_known"] is False  # la ficha del S3 no confirma la FPU: se avisa


def test_known_soft_float_board_is_much_slower_than_fpu_board(run):
    r = estimate_latency(run, "logreg", "all")["boards"]
    by = {b["board"]: b for b in r}
    slow, fast = by["stm32f103"], by["stm32f411"]  # sin FPU vs con FPU
    assert slow["ms_hi"] / slow["clock_mhz"] > fast["ms_hi"] / fast["clock_mhz"] * 5
    assert "ESTIM" in estimate_latency(run, "logreg")["disclaimer"]


def test_target_and_unknown_board(run):
    res = estimate_latency(run, "decision_tree", "all", target_ms=1000.0)
    assert all(b["meets_target"] for b in res["boards"])
    with pytest.raises(ValueError):
        estimate_latency(run, "decision_tree", "no-such-board")
