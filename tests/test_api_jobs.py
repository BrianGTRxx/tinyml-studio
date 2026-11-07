"""API de trabajos: subprocesos reales, SSE, cancelación, seguridad y persistencia."""

import json
import shutil
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tinyml_studio.api import create_app
from tinyml_studio.api.jobs import JobManager
from tinyml_studio.dataset import inspect_files

TOKEN = "j" * 32
RNG = np.random.default_rng(21)


def auth(tok=TOKEN):
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    n = 450
    base = np.repeat([0.0, 5.0, 10.0], n // 3)
    pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1, n),
            "f2": RNG.normal(size=n) + base / 2,
            "label": np.repeat(["a", "b", "c"], n // 3),
        }
    ).to_csv(data / "d.csv", index=False)
    app = create_app(TOKEN, [data], tmp_path / "ws", enable_test_jobs=True)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    recipe = inspect_files([data / "d.csv"])["recipe"]
    return client, recipe, data, tmp_path


def wait(client, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{jid}", headers=auth()).json()
        if j["state"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return j
        time.sleep(0.2)
    raise AssertionError("timeout esperando el trabajo")


def submit(client, kind, params, expect=202):
    r = client.post("/jobs", headers=auth(), json={"kind": kind, "params": params})
    assert r.status_code == expect, r.text
    return r.json()


def test_catalog_endpoints(env):
    client, *_ = env
    hw = client.get("/hardware", headers=auth()).json()
    assert hw["cpu"]["logical_cores"] >= 1 and hw["suggested_device"] in ("cpu", "gpu")
    b = client.get("/boards", headers=auth()).json()
    assert any(x["id"] == "esp32-s3" for x in b["boards"])
    assert b["all_verified"] is True and b["n_verified"] == len(b["boards"])
    m = client.get("/models", headers=auth()).json()
    assert "random_forest" in m["classical"]["models"]["classification"]
    assert m["keras_cnn1d"]["defaults"]["window"] == 16


def test_requires_token_for_jobs(env):
    client, recipe, *_ = env
    assert client.get("/jobs").status_code == 401
    assert client.post("/jobs", json={"kind": "train", "params": {}}).status_code == 401


def test_train_job_lifecycle_with_sse_and_files(env):
    client, recipe, data, tmp = env
    job = submit(
        client, "train", {"recipe": recipe, "models": ["decision_tree", "random_forest"], "seed": 0}
    )
    assert job["state"] in ("queued", "running")
    with client.stream("GET", f"/jobs/{job['id']}/events", headers=auth()) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [json.loads(ln[6:]) for ln in body.splitlines() if ln.startswith("data: ")]
    stages = [e["stage"] for e in events]
    assert "train" in stages and "model_done" in stages and stages[-1] == "end"
    assert [e["seq"] for e in events if "seq" in e] == sorted(
        e["seq"] for e in events if "seq" in e
    )
    done = wait(client, job["id"])
    assert done["state"] == "succeeded" and done["progress"]["pct"] == 100
    res = client.get(f"/jobs/{job['id']}/result", headers=auth()).json()
    assert res["family"] == "classical"
    assert res["results"]["selection"]["best"] in ("decision_tree", "random_forest")
    assert "Mejor por" in res["summary_es"]
    files = {f["path"] for f in client.get(f"/jobs/{job['id']}/files", headers=auth()).json()}
    assert {"results.json", "encoder.json", "model_random_forest.joblib"} <= files
    txt = client.get(f"/jobs/{job['id']}/files/results.json", headers=auth())
    assert txt.status_code == 200 and json.loads(txt.text)["task"] == "classification"
    # binarios no se sirven por la API (se guardan desde el proceso principal)
    assert (
        client.get(
            f"/jobs/{job['id']}/files/model_random_forest.joblib", headers=auth()
        ).status_code
        == 415
    )
    # reconexión: ?after= solo devuelve lo posterior
    with client.stream("GET", f"/jobs/{job['id']}/events?after=9999", headers=auth()) as r:
        assert "event: end" in "".join(r.iter_text())


def test_clean_job_returns_histograms_and_split_sizes(env):
    client, recipe, *_ = env
    job = submit(client, "clean", {"recipe": recipe})
    assert wait(client, job["id"])["state"] == "succeeded"
    res = client.get(f"/jobs/{job['id']}/result", headers=auth()).json()
    assert res["split"]["sizes"]["train"] > res["split"]["sizes"]["test"] > 0
    assert set(res["split"]["class_distribution"]["train"]) == {"a", "b", "c"}
    c = res["columns"][0]
    assert len(res["histograms"][c]["edges"]) == len(res["histograms"][c]["before"]) + 1
    assert any(s["kind"] == "scale" for s in res["steps"])
    assert "unidades originales" in res["units_note"]
    st = next(s for s in res["stats"] if s["column"] == "f1")
    assert 0 < st["mean_after"] < 10  # media de f1 en unidades originales, no z-score (~0)


@pytest.mark.skipif(shutil.which("cc") is None, reason="sin compilador C")
def test_export_c_chained_from_train_job(env):
    client, recipe, *_ = env
    t = submit(client, "train", {"recipe": recipe, "models": ["decision_tree"], "seed": 0})
    assert wait(client, t["id"])["state"] == "succeeded"
    e = submit(client, "export_c", {"run_job": t["id"], "flash_kb": 64, "ram_kb": 16})
    assert wait(client, e["id"])["state"] == "succeeded"
    res = client.get(f"/jobs/{e['id']}/result", headers=auth()).json()
    assert res["family"] == "c" and res["results"]["verification"]["model_match"] == 1.0
    assert res["results"]["budget"]["flash_data"]["verdict"] == "cabe"
    names = {f["path"] for f in client.get(f"/jobs/{e['id']}/files", headers=auth()).json()}
    assert {"tm_model.c", "tm_model.h", "model_meta.json", "export_report.json"} <= names
    assert "tm_predict" in client.get(f"/jobs/{e['id']}/files/tm_model.h", headers=auth()).text
    # structure viewer + Arduino sketch ship with the export
    assert res["graph"]["format"] == "sklearn" and res["graph"]["model"]["type"] == "tree"
    ard = res["arduino"][0]
    assert ard["variant"] == "raw_c" and ard["verification"]["level"] == "host_compiled"
    assert any(f.endswith(".ino") for f in ard["code"]) and ard["input_table"]
    # quality block of the training run: per-split metrics, curves and mAP
    tr_res = client.get(f"/jobs/{t['id']}/result", headers=auth()).json()["results"]
    q = tr_res["models"]["decision_tree"]["quality"]
    assert {"train", "val", "test"} <= set(q["splits"]) and q["curves"]["roc"]
    assert 0 <= q["splits"]["test"]["mAP"] <= 1


def test_security_and_validation(env):
    client, recipe, data, tmp = env
    # archivo de la receta fuera de las raíces
    bad = json.loads(json.dumps(recipe))
    outside = tmp / "otro.csv"
    outside.write_text("a\n1\n")
    bad["files"] = [str(outside)]
    assert submit(client, "train", {"recipe": bad}, 403)["detail"].startswith("ruta fuera")
    bad["files"] = [str(data / ".." / ".." / "etc" / "passwd")]
    submit(client, "train", {"recipe": bad}, 403)
    # tipos y parámetros
    assert "desconocido" in submit(client, "rm_rf", {}, 422)["detail"]
    assert "inválidos" in submit(client, "train", {"recipe": recipe, "inventado": 1}, 422)["detail"]
    assert (
        "inválidos"
        in submit(client, "train_keras", {"recipe": recipe, "epochs": 10**6}, 422)["detail"]
    )
    submit(client, "train", {"recipe": {"files": []}}, 422)
    # run_job inexistente / con formato raro / sin terminar
    submit(client, "export_c", {"run_job": "../../etc"}, 404)
    submit(client, "export_c", {"run_job": "0" * 12}, 404)
    # id y rutas raros
    assert client.get("/jobs/..%2F..", headers=auth()).status_code == 404
    j = submit(client, "_sleep", {"seconds": 0.2})
    wait(client, j["id"])
    assert client.get(f"/jobs/{j['id']}/files/..%2Fjob.json", headers=auth()).status_code == 404
    assert client.get(f"/jobs/{j['id']}/files/../../job.json", headers=auth()).status_code in (
        404,
        422,
    )


def test_test_jobs_disabled_by_default(tmp_path):
    app = create_app(TOKEN, [tmp_path], tmp_path / "ws")
    c = TestClient(app, base_url="http://127.0.0.1:8000")
    r = c.post("/jobs", headers=auth(), json={"kind": "_sleep", "params": {"seconds": 1}})
    assert r.status_code == 422


def test_cancel_running_job_kills_process(env):
    client, *_ = env
    j = submit(client, "_sleep", {"seconds": 60})
    t0 = time.time()
    while client.get(f"/jobs/{j['id']}", headers=auth()).json()["state"] != "running":
        assert time.time() - t0 < 20
        time.sleep(0.1)
    r = client.post(f"/jobs/{j['id']}/cancel", headers=auth())
    assert r.status_code == 200 and r.json()["state"] == "cancelled"
    assert wait(client, j["id"], 30)["state"] == "cancelled"
    assert client.post(f"/jobs/{j['id']}/cancel", headers=auth()).status_code == 409
    assert client.get(f"/jobs/{j['id']}/result", headers=auth()).status_code == 409


def test_queue_runs_one_at_a_time_and_failed_job_reports_error(env):
    client, recipe, *_ = env
    a = submit(client, "_sleep", {"seconds": 1.5})
    b = submit(client, "_sleep", {"seconds": 0.2})
    assert client.get(f"/jobs/{b['id']}", headers=auth()).json()["state"] == "queued"
    ja, jb = wait(client, a["id"]), wait(client, b["id"])
    assert ja["state"] == jb["state"] == "succeeded" and jb["started"] >= ja["finished"] - 0.5
    # un modelo inexistente falla con el mensaje, sin tumbar el sidecar
    f = submit(client, "train", {"recipe": recipe, "models": ["no_existe"]})
    fj = wait(client, f["id"])
    assert fj["state"] == "failed" and "no_existe" in fj["error"]
    assert client.get("/health", headers=auth()).status_code == 200
    assert [x["id"] for x in client.get("/jobs", headers=auth()).json()][0] == f["id"]


def test_delete_and_restart_persistence(env):
    client, recipe, data, tmp = env
    j = submit(client, "_sleep", {"seconds": 0.2})
    wait(client, j["id"])
    long = submit(client, "_sleep", {"seconds": 60})
    time.sleep(1.0)
    # "reinicio" del sidecar: otro gestor sobre la misma carpeta de trabajo
    m2 = JobManager(tmp / "ws", lambda p: None)
    states = {x["id"]: x["state"] for x in m2.list()}
    assert states[j["id"]] == "succeeded" and states[long["id"]] == "interrupted"
    # un trabajo vivo no se borra
    assert client.delete(f"/jobs/{long['id']}", headers=auth()).status_code == 409
    client.post(f"/jobs/{long['id']}/cancel", headers=auth())
    wait(client, long["id"], 30)
    assert client.delete(f"/jobs/{j['id']}", headers=auth()).status_code == 200
    assert client.get(f"/jobs/{j['id']}", headers=auth()).status_code == 404
    assert not (tmp / "ws" / "jobs" / j["id"]).exists()


def test_validate_groups_job_reports_what_cannot_be_measured(env):
    client, recipe, data, tmp = env
    job = submit(client, "validate_groups", {"recipe": recipe, "model": "decision_tree", "seed": 0})
    done = wait(client, job["id"])
    assert done["state"] == "succeeded"
    res = client.get(f"/jobs/{job['id']}/result", headers=auth()).json()
    assert res["family"] == "group_validation"
    r = res["results"]
    assert r["verdict"] in ("partial", "not_measurable", "measurable")
    assert r["with_all_features"][
        "classes_not_measurable"
    ]  # el dataset de ejemplo: casi todo sin 2ª sesión
    assert "sesión" in res["summary"] or "session" in res["summary"]


def test_export_onnx_job_verifies_against_sklearn(env):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")
    client, recipe, *_ = env
    t = submit(client, "train", {"recipe": recipe, "models": ["decision_tree"], "seed": 0})
    assert wait(client, t["id"])["state"] == "succeeded"
    e = submit(client, "export_onnx", {"run_job": t["id"]})
    assert wait(client, e["id"])["state"] == "succeeded"
    res = client.get(f"/jobs/{e['id']}/result", headers=auth()).json()
    assert res["family"] == "onnx" and res["results"]["verification"]["ok"] is True
    names = {f["path"] for f in client.get(f"/jobs/{e['id']}/files", headers=auth()).json()}
    assert {"model.onnx", "onnx_meta.json", "infer_onnx.py"} <= names


def test_tune_job_respects_budget(env):
    client, recipe, *_ = env
    j = submit(
        client, "tune", {"recipe": recipe, "model": "decision_tree", "n_trials": 6, "seed": 0}
    )
    assert wait(client, j["id"])["state"] == "succeeded"
    res = client.get(f"/jobs/{j['id']}/result", headers=auth()).json()
    assert res["family"] == "tuning" and res["results"]["best"]["test_score"] is not None
    bad = submit(client, "tune", {"recipe": recipe, "model": "knn"})
    assert wait(client, bad["id"])["state"] == "failed"


def test_importance_job_from_train_job(env):
    client, recipe, *_ = env
    t = submit(client, "train", {"recipe": recipe, "models": ["decision_tree"], "seed": 0})
    assert wait(client, t["id"])["state"] == "succeeded"
    j = submit(client, "importance", {"run_job": t["id"], "n_repeats": 2})
    assert wait(client, j["id"])["state"] == "succeeded"
    res = client.get(f"/jobs/{j['id']}/result", headers=auth()).json()
    assert res["family"] == "importance" and res["results"]["features"]


def test_anomaly_job_reports_detection(env):
    client, recipe, *_ = env
    j = submit(client, "anomaly", {"recipe": recipe, "normal_classes": ["a"], "false_alarm": 0.05})
    assert wait(client, j["id"])["state"] == "succeeded"
    res = client.get(f"/jobs/{j['id']}/result", headers=auth()).json()
    assert res["family"] == "anomaly" and res["results"]["detection"] is not None
    bad = submit(client, "anomaly", {"recipe": recipe, "normal_classes": ["zzz"]})
    assert wait(client, bad["id"])["state"] == "failed"


def test_predict_job_describes_inputs_and_answers_with_range_warning(env):
    client, recipe, _, _ = env
    t = wait(client, submit(client, "train", {"recipe": recipe, "models": ["decision_tree"]})["id"])
    assert t["state"] == "succeeded"
    j = wait(client, submit(client, "predict", {"run_job": t["id"]})["id"])
    sch = client.get(f"/jobs/{j['id']}/result", headers=auth()).json()
    assert sch["family"] == "predict_schema"
    assert [c["name"] for c in sch["results"]["numeric"]] == ["f1", "f2"]
    j = wait(
        client,
        submit(client, "predict", {"run_job": t["id"], "values": {"f1": 0.1, "f2": 0.2}})["id"],
    )
    res = client.get(f"/jobs/{j['id']}/result", headers=auth()).json()
    assert res["family"] == "predict" and res["results"]["class"] == "a"
    j = wait(
        client,
        submit(client, "predict", {"run_job": t["id"], "values": {"f1": 900, "f2": 0}})["id"],
    )
    far = client.get(f"/jobs/{j['id']}/result", headers=auth()).json()
    assert [w["column"] for w in far["results"]["out_of_range"]] == ["f1"]
    submit(client, "predict", {"run_job": t["id"], "values": {"f1": "x" * 500}}, expect=422)
