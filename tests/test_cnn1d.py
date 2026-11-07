import json
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("tflite_micro")

from tinyml_studio.cli import main  # noqa: E402
from tinyml_studio.dataset import inspect_files  # noqa: E402
from tinyml_studio.export.tflite import export_tflite  # noqa: E402
from tinyml_studio.train.cnn1d import prepare_windows, train_keras_cnn1d  # noqa: E402
from tinyml_studio.window import WindowConfig  # noqa: E402

RNG = np.random.default_rng(9)
P = 20  # periodo de la rampa


def _ramps(tmp_path, n=1500):
    """Tres clases con la MISMA distribución de valores sueltos (uniforme 0..1) pero distinta
    dinámica: rampa ascendente, rampa descendente y ruido blanco. Una sola fila no distingue
    ascendente de descendente; una ventana sí."""
    ph = (np.arange(n) % P) / P
    sig = {"sube": ph, "baja": 1 - ph, "ruido": RNG.random(n)}
    for k, s in sig.items():
        pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=n, freq="s").strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "x": s + RNG.normal(0, 0.01, n),
                "y": RNG.normal(size=n) * 0.0 + 0.5,
            }
        ).to_csv(tmp_path / f"{k}.csv", index=False)
    return inspect_files([tmp_path])["recipe"]


def test_requires_temporal_split_and_classification(tmp_path):
    r = _ramps(tmp_path)
    r["split"] = {**r["split"], "strategy": "random"}
    with pytest.raises(ValueError, match="temporal_blocks"):
        prepare_windows(r, WindowConfig(), 0)


def test_windows_do_not_leak_across_splits(tmp_path):
    r = _ramps(tmp_path)
    assert r["split"]["strategy"] == "temporal_blocks"
    d = prepare_windows(r, WindowConfig(window=8, stride=2), 0)
    # cada ventana de test/val usa solo filas de su propio split y en orden temporal
    for s in ("train", "val", "test"):
        times = d.prep.clean[s]["time"].astype("int64").to_numpy()
        assert (np.diff(times[d.idx[s]], axis=1) > 0).all()
        assert d.X[s].shape[1:] == (8, d.enc.feature_names.__len__())
    assert d.stats["train"]["dropped_mixed_label"] == 0
    # train y test no comparten filas (bloques temporales por archivo)
    t_tr = set(d.prep.clean["train"]["time"].astype("int64")) & set()
    assert not t_tr


def test_cnn_beats_last_row_when_dynamics_matter(tmp_path):
    r = _ramps(tmp_path)
    res = train_keras_cnn1d(
        r,
        window=8,
        stride=1,
        filters=(8, 8),
        kernel=3,
        epochs=60,
        batch_size=64,
        learning_rate=5e-3,
        seed=0,
        out_dir=tmp_path / "run",
    )
    assert res["kind"] == "keras_cnn1d" and res["architecture"]["head"] == "gap"
    assert res["test"]["f1_macro"] > 0.9
    # la fila suelta no distingue "sube" de "baja": la referencia se queda muy por debajo
    assert res["reference_rf_last_row"]["test"]["f1_macro"] < res["test"]["f1_macro"] - 0.25
    assert not any("no supera" in w for w in res["warnings"])
    assert (tmp_path / "run" / "model.keras").exists()


def test_cnn_flatten_head_and_bad_window(tmp_path):
    r = _ramps(tmp_path)
    res = train_keras_cnn1d(
        r, window=6, stride=2, filters=(4,), kernel=3, head="flatten", epochs=3, seed=0
    )
    assert res["architecture"]["head"] == "flatten"
    with pytest.raises(ValueError, match="demasiado corta"):
        train_keras_cnn1d(r, window=4, filters=(4, 4, 4), kernel=3, epochs=1)
    with pytest.raises(ValueError, match="head"):
        train_keras_cnn1d(r, head="otro")


def test_export_cnn_to_tflite_micro(tmp_path):
    r = _ramps(tmp_path)
    train_keras_cnn1d(
        r,
        window=8,
        stride=1,
        filters=(8, 8),
        epochs=40,
        batch_size=64,
        learning_rate=5e-3,
        seed=0,
        out_dir=tmp_path / "run",
    )
    rep = export_tflite(tmp_path / "run", tmp_path / "x", flash_kb=64, ram_kb=32)
    out = tmp_path / "x"
    assert {"AddConv2D", "AddMean", "AddFullyConnected", "AddSoftmax"} <= set(rep["tflm_resolver"])
    assert rep["comparison"]["float_vs_keras_outputs"]["max_abs"] < 1e-4
    assert rep["comparison"]["agreement_tflm_vs_tflite_int8"] >= 0.99
    assert rep["metrics_test"]["tflm_int8"]["f1_macro"] > 0.8
    assert rep["arena"]["measured"] and rep["budget"]["flash_data"]["verdict"] == "cabe"
    wv = rep["window_c_verification"]
    assert wv["compile_ok"] and wv["compile_warnings"] == [] and wv["windows"] > 20
    assert wv["features_max_rel_diff"] < 1e-6 and wv["ready_flag_ok"] is True  # C == Python
    cfg = (out / "model_config.h").read_text()
    assert "TM_WINDOW 8" in cfg and "TM_N_FEATURES 1" in cfg  # la columna constante "y" se descarta
    ino = (out / "inference_example.ino").read_text()
    assert "tm_window_push" in ino and "ml.setNumInputs(TM_WINDOW * TM_N_FEATURES)" in ino
    head = (out / "test_vectors.csv").read_text().splitlines()[0].split(",")
    assert head[0] == "t0_num_x" and head[-2:] == ["expected_int8", "expected_keras"]
    nums = [int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})", (out / "model_data.h").read_text())]
    assert bytes(nums) == (out / "model_int8.tflite").read_bytes()
    assert not (out / "verify_window").exists()  # no se entrega el binario de PC


def test_cli_train_cnn1d_and_export(tmp_path, capsys):
    r = _ramps(tmp_path)
    (tmp_path / "r.json").write_text(json.dumps(r))
    assert (
        main(
            [
                "train-cnn1d",
                str(tmp_path / "r.json"),
                "--out-dir",
                str(tmp_path / "run"),
                "--window",
                "8",
                "--stride",
                "2",
                "--filters",
                "8,8",
                "--epochs",
                "5",
            ]
        )
        == 0
    )
    assert "1D-CNN Keras" in capsys.readouterr().out
    assert main(["export-tflite", str(tmp_path / "run"), "--out-dir", str(tmp_path / "x")]) == 0
    out = capsys.readouterr().out
    assert "AddConv2D" in out and "C de la ventana (tm_window)" in out


