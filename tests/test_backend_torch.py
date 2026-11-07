"""Backend Keras-sobre-PyTorch (solo CPU en estas pruebas: no hay GPU) y 'device=gpu'."""

import json
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from tinyml_studio.api import create_app  # noqa: E402
from tinyml_studio.dataset import inspect_files  # noqa: E402
from tinyml_studio.export.tflite import export_tflite  # noqa: E402
from tinyml_studio.hw import detect as D  # noqa: E402
from tinyml_studio.train import backend as B  # noqa: E402
from tinyml_studio.train.cnn1d import train_keras_cnn1d  # noqa: E402
from tinyml_studio.train.keras_mlp import train_keras_mlp  # noqa: E402

RNG = np.random.default_rng(11)


def _blobs(tmp, n=600):
    k = n // 3
    base = np.repeat([0.0, 6.0, 12.0], k)
    pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1.0, n),
            "f2": RNG.normal(0, 1, n) + base / 2,
            "label": np.repeat(["a", "b", "c"], k),
        }
    ).to_csv(tmp / "d.csv", index=False)
    return inspect_files([tmp / "d.csv"])["recipe"]


def _ramps(tmp, n=1200, p=20):
    ph = (np.arange(n) % p) / p
    for k, s in {"sube": ph, "baja": 1 - ph, "ruido": RNG.random(n)}.items():
        pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=n, freq="s").strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "x": s + RNG.normal(0, 0.01, n),
                "y": RNG.normal(size=n) * 0.0 + 0.5,
            }
        ).to_csv(tmp / f"{k}.csv", index=False)
    return inspect_files([tmp])["recipe"]


def _check_transfer(res):
    c = res["compute"]
    assert c["backend"] == "torch" and "torch" in c["frameworks"]
    t = c["torch_to_tf_transfer"]
    assert t["max_abs_diff"] < 1e-4 and t["rows_checked"] > 0 and t["split"] == "test"


def _only_transfer_artifacts_removed(run):
    assert not list(run.glob("transfer_*"))
    assert (run / "model.keras").exists() and (run / "results.json").exists()


def test_mlp_torch_trains_transfers_and_exports(tmp_path):
    r = _blobs(tmp_path)
    run = tmp_path / "run"
    res = train_keras_mlp(
        r,
        hidden=(8,),
        epochs=60,
        batch_size=32,
        learning_rate=0.01,
        seed=0,
        out_dir=run,
        backend="torch",
        optimizer="sgd",
        nesterov=True,
        lr_schedule="cosine",
    )
    _check_transfer(res)
    _only_transfer_artifacts_removed(run)
    assert res["hyperparameters"]["optimizer"] == "sgd" and res["test"]["f1_macro"] > 0.8
    assert json.loads((run / "results.json").read_text())["compute"]["backend"] == "torch"
    # el model.keras guardado es ahora un modelo TF normal: la exportación no cambia
    rep = export_tflite(run, tmp_path / "x")
    assert rep["comparison"]["float_vs_keras_outputs"]["max_abs"] < 1e-4
    assert rep["metrics_test"]["tflite_int8"]["f1_macro"] > 0.8
    if rep["metrics_test"].get("tflm_int8"):
        assert rep["comparison"]["agreement_tflm_vs_tflite_int8"] >= 0.99


def test_regression_mlp_torch_keeps_original_units(tmp_path):
    n = 500
    x = RNG.normal(size=n)
    pd.DataFrame({"x": x, "y": 100 + 20 * x + RNG.normal(0, 1, n)}).to_csv(
        tmp_path / "r.csv", index=False
    )
    rec = inspect_files([tmp_path / "r.csv"])["recipe"]
    rec["task"], rec["target"] = "regression", "y"
    res = train_keras_mlp(
        rec, hidden=(8,), epochs=40, seed=0, out_dir=tmp_path / "run", backend="torch"
    )
    _check_transfer(res)  # la capa final ya plegada (media/desvío) coincide en la copia TF
    assert res["test"]["r2"] > 0.8


def test_cnn1d_torch_trains_transfers_and_exports(tmp_path):
    r = _ramps(tmp_path)
    run = tmp_path / "run"
    res = train_keras_cnn1d(
        r,
        window=8,
        stride=1,
        filters=(8, 8),
        epochs=40,
        batch_size=64,
        learning_rate=5e-3,
        seed=0,
        out_dir=run,
        backend="torch",
        optimizer="adamw",
    )
    _check_transfer(res)
    _only_transfer_artifacts_removed(run)
    assert res["test"]["f1_macro"] > 0.8 and res["hyperparameters"]["optimizer"] == "adamw"
    rep = export_tflite(run, tmp_path / "x", flash_kb=64, ram_kb=32)
    assert rep["comparison"]["float_vs_keras_outputs"]["max_abs"] < 1e-4
    assert rep["metrics_test"]["tflite_int8"]["f1_macro"] > 0.8




def test_transfer_refuses_diverging_weights_and_unknown_layers(tmp_path):
    with pytest.raises(ValueError, match="sin transferencia"):
        B._build({"kind": "lstm"})
    with pytest.raises(ValueError, match="backend"):
        B.check_backend("jax")
    # pesos que no reproducen las predicciones guardadas: no se reemplaza model.keras
    arch = {
        "kind": "mlp", "n_features": 2, "n_out": 3, "task": "classification",
        "hidden": [4], "activation": "relu", "dropout": 0.0, "l2": 0.0,
    }  # fmt: skip
    (tmp_path / "transfer_arch.json").write_text(json.dumps(arch))
    shapes = [(2, 4), (4,), (4, 3), (3,)]
    np.savez(
        tmp_path / "transfer_weights.npz", **{f"w{i}": np.ones(s) for i, s in enumerate(shapes)}
    )
    np.savez(
        tmp_path / "transfer_check.npz",
        X=np.ones((5, 2), "float32"),
        pred=np.zeros((5, 3), "float32"),
    )
    with pytest.raises(RuntimeError, match="difiere|differs"):
        B._transfer(tmp_path)
    assert not (tmp_path / "model.keras").exists()


# ------------------------------------------------------------------ device='gpu'


def _fake_probe(monkeypatch, tf_gpus=0, torch_gpus=0):
    gpus = {"tensorflow": tf_gpus, "torch": torch_gpus}
    monkeypatch.setattr(
        D, "probe_framework", lambda n, refresh=False: {"importable": True, "gpu_count": gpus[n]}
    )


def test_plan_gpu_without_gpu_falls_back_with_warning(monkeypatch):
    _fake_probe(monkeypatch)
    for b in B.BACKENDS:
        plan = B.plan_compute(b, "gpu")
        assert plan["backend"] == b and plan["device"] == "cpu" and plan["gpu_seen_by"] == []
        assert "CPU" in plan["notes"][0] and "GPU_AMD" in plan["notes"][0]


def test_plan_gpu_switches_to_the_framework_that_sees_it(monkeypatch):
    _fake_probe(monkeypatch, torch_gpus=1)
    plan = B.plan_compute("tensorflow", "gpu")
    assert plan["backend"] == "torch" and plan["device"] == "gpu" and "torch" in plan["notes"][0]
    _fake_probe(monkeypatch, tf_gpus=1)
    assert B.plan_compute("torch", "gpu")["backend"] == "tensorflow"
    plan = B.plan_compute("tensorflow", "gpu")  # ya lo ve: sin cambios ni avisos
    assert plan["backend"] == "tensorflow" and plan["device"] == "gpu" and plan["notes"] == []
    assert B.plan_compute("torch", "cpu")["device"] == "cpu"
    with pytest.raises(ValueError):
        B.plan_compute("torch", "tpu")


def test_library_gpu_request_without_gpu_still_trains(tmp_path, monkeypatch):
    _fake_probe(monkeypatch)
    res = train_keras_mlp(_blobs(tmp_path), hidden=(4,), epochs=2, seed=0, device="gpu")
    assert res["compute"]["device_seen_by_framework"] == "cpu"
    assert any("no framework" in w or "ningún framework" in w for w in res["warnings"])


# ------------------------------------------------------------------ a través de la API


def test_api_job_torch_with_device_gpu_falls_back(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    recipe = _blobs(data)
    tok = "k" * 32
    client = TestClient(create_app(tok, [data], tmp_path / "ws"), base_url="http://127.0.0.1:8000")
    h = {"Authorization": f"Bearer {tok}"}
    r = client.post(
        "/jobs",
        headers=h,
        json={
            "kind": "train_keras",
            "params": {
                "recipe": recipe, "hidden": [8], "epochs": 30, "batch_size": 32,
                "learning_rate": 0.01, "seed": 0, "backend": "torch", "device": "gpu",
                "optimizer": "adamw", "lang": "en",
            },
        },
    )  # fmt: skip
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    t0 = time.time()
    while time.time() - t0 < 600:
        j = client.get(f"/jobs/{jid}", headers=h).json()
        if j["state"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.5)
    assert j["state"] == "succeeded", j
    res = client.get(f"/jobs/{jid}/result", headers=h).json()["results"]
    assert res["compute"]["backend"] == "torch" and res["compute"]["device_requested"] == "gpu"
    _check_transfer(res)
    assert any("no framework" in w for w in res["warnings"])  # idioma 'en' del trabajo
    bad = client.post(
        "/jobs",
        headers=h,
        json={"kind": "train_keras", "params": {"recipe": recipe, "backend": "jax"}},
    )
    assert bad.status_code in (400, 422)
