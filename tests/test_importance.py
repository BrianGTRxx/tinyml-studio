"""Importancia por permutación: lo informativo cae, el ruido no."""

from test_tune import _recipe

from tinyml_studio.train.importance import permutation_importance
from tinyml_studio.train.run import train_models


def test_noise_column_is_useless_and_informative_ones_rank_first(tmp_path):
    rec = _recipe(tmp_path)
    train_models(rec, ["random_forest"], seed=0, out_dir=tmp_path / "run")
    res = permutation_importance(tmp_path / "run", "random_forest", n_repeats=3)
    order = [r["feature"] for r in res["features"]]
    assert order[-1] == "x3" and "x3" in res["useless"]
    assert set(order[:2]) == {"x1", "x2"}
    assert res["features"][0]["drop"] > 0.05
