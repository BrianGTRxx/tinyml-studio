import json
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("tflite_micro")

from fastapi.testclient import TestClient  # noqa: E402

from tinyml_studio.api import create_app  # noqa: E402
from tinyml_studio.dataset import inspect_files  # noqa: E402

TOKEN = "k" * 32


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def wait(client, jid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{jid}", headers=auth()).json()
        if j["state"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(0.3)
    raise AssertionError("timeout")


def test_keras_train_and_tflite_export_through_the_api(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    n = 900
    rng = np.random.default_rng(2)
    base = np.repeat([0.0, 6.0, 12.0], n // 3)
    pd.DataFrame(
        {
            "f1": base + rng.normal(0, 1, n),
            "f2": rng.normal(size=n) + base / 2,
            "label": np.repeat(["a", "b", "c"], n // 3),
        }
    ).to_csv(data / "d.csv", index=False)
    client = TestClient(
        create_app(TOKEN, [data], tmp_path / "ws"), base_url="http://127.0.0.1:8000"
    )
    recipe = inspect_files([data / "d.csv"])["recipe"]
    r = client.post(
        "/jobs",
        headers=auth(),
        json={
            "kind": "train_keras",
            "params": {
                "recipe": recipe,
                "hidden": [8],
                "epochs": 40,
                "batch_size": 32,
                "learning_rate": 0.01,
                "seed": 0,
                "device": "cpu",
            },
        },
    )
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    with client.stream("GET", f"/jobs/{jid}/events", headers=auth()) as s:
        body = "".join(s.iter_text())
    epochs = [
        json.loads(ln[6:])
        for ln in body.splitlines()
        if ln.startswith("data: ") and '"stage": "epoch"' in ln
    ]
    assert epochs and epochs[0]["loss"] > 0 and "val_loss" in epochs[0]  # curvas en vivo
    assert wait(client, jid)["state"] == "succeeded"
    res = client.get(f"/jobs/{jid}/result", headers=auth()).json()
    assert res["family"] == "keras_mlp" and res["results"]["test"]["f1_macro"] > 0.85
    assert len(res["results"]["history"]["loss"]) == res["results"]["hyperparameters"]["epochs_run"]

    e = client.post(
        "/jobs",
        headers=auth(),
        json={"kind": "export_tflite", "params": {"run_job": jid, "flash_kb": 64, "ram_kb": 32}},
    ).json()
    assert wait(client, e["id"])["state"] == "succeeded"
    rep = client.get(f"/jobs/{e['id']}/result", headers=auth()).json()["results"]
    assert rep["comparison"]["agreement_tflm_vs_tflite_int8"] >= 0.99
    full = client.get(f"/jobs/{e['id']}/result", headers=auth()).json()
    assert full["graph"]["format"] == "tflite" and full["graph"]["nodes"]
    assert {a["variant"] for a in full["arduino"]} == {"eloquent", "tflm"}
    assert all(a["verification"]["level"] == "not_compiled" for a in full["arduino"])
    q = res["results"]["quality"]
    assert q["kind"] == "classification" and q["curves"]["pr"]
    assert (
        rep["budget"]["flash_data"]["verdict"] == "cabe"
        and "AddFullyConnected" in rep["tflm_resolver"]
    )
    # exportar a C desde una corrida de Keras no tiene sentido: error claro, no caída
    bad = client.post(
        "/jobs", headers=auth(), json={"kind": "export_c", "params": {"run_job": jid}}
    ).json()
    assert wait(client, bad["id"])["state"] == "failed"
