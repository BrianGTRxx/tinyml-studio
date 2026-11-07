"""Exportación float16 junto a float e int8 (no requiere tflite-micro)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from tinyml_studio.dataset import inspect_files  # noqa: E402
from tinyml_studio.export.tflite import export_tflite  # noqa: E402
from tinyml_studio.train.keras_mlp import train_keras_mlp  # noqa: E402


def test_float16_is_written_smaller_and_matches_float(tmp_path):
    rng = np.random.default_rng(1)
    n = 600
    base = np.repeat([0.0, 6.0, 12.0], n // 3)
    pd.DataFrame(
        {
            "f1": base + rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n) + base / 2,
            "label": np.repeat(["a", "b", "c"], n // 3),
        }
    ).to_csv(tmp_path / "d.csv", index=False)
    recipe = inspect_files([tmp_path / "d.csv"])["recipe"]
    train_keras_mlp(
        recipe, hidden=(64, 64), epochs=5, batch_size=32, seed=0, out_dir=tmp_path / "run"
    )
    rep = export_tflite(tmp_path / "run", tmp_path / "out")
    sizes = rep["sizes_bytes"]
    assert (tmp_path / "out" / "model_float16.tflite").exists()
    assert sizes["tflite_float16"] < sizes["tflite_float"]
    assert rep["comparison"]["float16_vs_float_outputs"]["max_abs"] < 0.01
    assert rep["float16_error"] is None
    from tinyml_studio.export.report import tflite_summary

    assert "float16" in tflite_summary(rep, str(tmp_path / "out"))
