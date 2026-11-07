"""Entorno del núcleo (fuente/lite/full) y degradación elegante sin Keras/TensorFlow."""

import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tinyml_studio import environment as E
from tinyml_studio.api import create_app
from tinyml_studio.api.catalog import get_catalog
from tinyml_studio.dataset import inspect_files

TOKEN = "e" * 32
REAL_FIND_SPEC = importlib.util.find_spec


def no_ml(monkeypatch, *names):
    """Simula que estos paquetes no están instalados (sin importar TF)."""
    gone = set(names or ("tensorflow", "keras", "torch"))

    def fake(name, *a, **k):
        return None if name.split(".")[0] in gone else REAL_FIND_SPEC(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_catalog_environment_source_mode():
    env = get_catalog()["environment"]
    assert set(env) == {"keras", "torch", "frozen", "variant"}
    assert env["frozen"] is False and env["variant"] == "source"
    assert env["keras"] is (
        REAL_FIND_SPEC("keras") is not None and REAL_FIND_SPEC("tensorflow") is not None
    )


def test_environment_without_ml_packages(monkeypatch):
    no_ml(monkeypatch)
    assert E.environment() == {"keras": False, "torch": False, "frozen": False, "variant": "source"}
    assert get_catalog()["environment"]["keras"] is False


def test_variant_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    (tmp_path / "tinyml_variant.txt").write_text("full\n")
    assert E.environment()["variant"] == "full" and E.environment()["frozen"] is True
    (tmp_path / "tinyml_variant.txt").write_text("lite\n")
    assert E.variant() == "lite"
    (tmp_path / "tinyml_variant.txt").unlink()
    no_ml(monkeypatch)
    assert E.variant() == "lite"  # sin marcador: se deduce de lo que hay


def test_resources_resolve_to_meipass_when_frozen(monkeypatch, tmp_path):
    from tinyml_studio.resources import resource_path

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert resource_path("data", "boards") == tmp_path / "data" / "boards"
    monkeypatch.undo()
    assert (resource_path("data") / "boards.seed.yaml").is_file()


@pytest.fixture
def client_recipe(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    n, rng = 300, np.random.default_rng(5)
    base = np.repeat([0.0, 6.0, 12.0], n // 3)
    pd.DataFrame(
        {
            "f1": base + rng.normal(0, 1, n),
            "f2": rng.normal(size=n),
            "label": np.repeat(["a", "b", "c"], n // 3),
        }
    ).to_csv(data / "d.csv", index=False)
    client = TestClient(
        create_app(TOKEN, [data], tmp_path / "ws"), base_url="http://127.0.0.1:8000"
    )
    return client, inspect_files([data / "d.csv"])["recipe"]


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.mark.parametrize("kind", ["train_keras", "train_cnn1d", "export_tflite"])
def test_keras_jobs_fail_early_and_clearly_without_keras(monkeypatch, client_recipe, kind):
    client, recipe = client_recipe
    no_ml(monkeypatch)
    params = {"run_job": "0" * 12} if kind == "export_tflite" else {"recipe": recipe}
    r = client.post("/jobs", headers=_auth(), json={"kind": kind, "params": params})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "Keras" in detail and "compilación completa" in detail and "Traceback" not in detail
    en = client.post(
        "/jobs?lang=en", headers=_auth(), json={"kind": kind, "params": params}
    ).json()["detail"]
    assert "does not include Keras" in en and "Install the full build" in en


def test_classical_jobs_are_not_blocked_without_keras(monkeypatch, client_recipe):
    client, recipe = client_recipe
    no_ml(monkeypatch)
    r = client.post(
        "/jobs",
        headers=_auth(),
        json={"kind": "train", "params": {"recipe": recipe, "models": ["decision_tree"]}},
    )
    assert r.status_code == 202
    for _ in range(300):
        j = client.get(f"/jobs/{r.json()['id']}", headers=_auth()).json()
        if j["state"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.2)
    assert j["state"] == "succeeded", j


def test_benchmark_without_tensorflow_runs_classical_and_lists_keras_skipped(
    monkeypatch, client_recipe
):
    """find_spec simulado: el benchmark corre lo clásico y deja los presets Keras en
    models_skipped (en el mismo proceso, igual que dentro del trabajo congelado)."""
    from tinyml_studio.train import bench as B

    _, recipe = client_recipe
    no_ml(monkeypatch)
    r = B.run_benchmark(
        recipe,
        models=["decision_tree"],
        keras=[{"name": "k1", "hidden": (4,), "epochs": 1}],
        seed=1,
    )
    assert [s["name"] for s in r["models_skipped"]] == ["keras_mlp:k1"]
    assert "TensorFlow" in r["models_skipped"][0]["reason"]
    assert len(r["leaderboard"]) == 1
    json.dumps(r, default=str)
