"""Historial, params y comparación de corridas (trabajos sintéticos + extremo a extremo)."""

import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tinyml_studio.api import create_app
from tinyml_studio.api.history import (
    build_history,
    compare_jobs,
    compare_rows,
    recipe_sha,
)
from tinyml_studio.api.jobs import JobError
from tinyml_studio.dataset import inspect_files

TOKEN = "h" * 32
H = {"Authorization": f"Bearer {TOKEN}"}


def auth():
    return H


def fake_job(
    workdir, jid, kind, state="succeeded", *, created=1000.0, params=None, result=None, **extra
):
    d = workdir / "jobs" / jid
    d.mkdir(parents=True)
    job = {
        "id": jid,
        "kind": kind,
        "state": state,
        "created": created,
        "started": created + 1,
        "finished": created + 6,
        "lang": None,
        "progress": {"stage": "done", "pct": 100, "msg": "x"},
        "error": "boom" if state == "failed" else None,
        "workdir": str(workdir),
        "summary": {},
        "device": "auto",
        **extra,
    }
    (d / "job.json").write_text(json.dumps(job), encoding="utf-8")
    (d / "params.json").write_text(json.dumps(params or {}), encoding="utf-8")
    if result is not None:
        (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return job


def recipe(files=("/home/secret/user/data/d.csv",)):
    return {"version": 1, "files": list(files), "target": "label", "roles": {"f1": "feature"}}


def split(f1, acc=None):
    return {"f1_macro": f1, "accuracy": acc if acc is not None else f1, "mcc": f1 - 0.05}


TRAIN_RES = {
    "family": "classical",
    "results": {
        "task": "classification",
        "rows": {"train": 360, "val": 45, "test": 45},
        "selection": {"metric": "f1_macro", "split": "val", "best": "random_forest"},
        "models": {
            "decision_tree": {"val": split(0.8), "test": split(0.78), "fit_seconds": 0.1},
            "random_forest": {
                "val": split(0.9),
                "test": split(0.88),
                "fit_seconds": 1.5,
                "predict_us_per_row_cpu_cloud": 7.0,
            },
        },
    },
}
BENCH_RES = {
    "family": "benchmark",
    "results": {
        "task": "classification",
        "rows": {"train": 360, "val": 45, "test": 45},
        "selection": {"metric": "f1_macro", "selected": "decision_tree"},
        "leaderboard": [
            {
                "name": n,
                "metrics": {"val": split(v), "test": {**split(v - 0.02), "mAP": 0.9}},
                "size_bytes": sz,
                "fit_seconds": 0.2,
                "predict_us_per_row": 3.0,
            }
            for n, v, sz in (("decision_tree", 0.95, 63), ("knn", 0.9, None))
        ],
    },
}
EXPORT_RES = {
    "family": "c",
    "results": {
        "board": "esp32-s3",
        "budget": {
            "flash_data": {"used_bytes": 63, "verdict_code": "fits"},
            "ram_working": {"used_bytes": 8, "verdict_code": "no_budget"},
        },
    },
}


@pytest.fixture
def wd(tmp_path):
    w = tmp_path / "ws"
    (w / "jobs").mkdir(parents=True)
    return w


def test_recipe_sha_is_canonical_and_order_independent():
    a = {"b": 1, "a": [1, 2]}
    assert recipe_sha(a) == recipe_sha({"a": [1, 2], "b": 1}) and len(recipe_sha(a)) == 64
    assert recipe_sha({"a": 2}) != recipe_sha(a) and recipe_sha(None) is None


def test_history_entries_per_kind(wd):
    r = recipe()
    j1 = fake_job(
        wd,
        "aaaaaaaaaaa1",
        "train",
        created=1000,
        params={"recipe": r},
        result=TRAIN_RES,
        hw_summary={"cpu_peak": 80},
    )
    j2 = fake_job(
        wd, "aaaaaaaaaaa2", "benchmark", created=2000, params={"recipe": r}, result=BENCH_RES
    )
    j3 = fake_job(
        wd,
        "aaaaaaaaaaa3",
        "export_c",
        created=3000,
        params={"run_job": "aaaaaaaaaaa1", "model": "random_forest"},
        result=EXPORT_RES,
    )
    j4 = fake_job(wd, "aaaaaaaaaaa4", "train", state="failed", created=4000, params={"recipe": r})
    j5 = fake_job(
        wd,
        "aaaaaaaaaaa5",
        "clean",
        created=5000,
        params={"recipe": r},
        result={"split": {"sizes": {"train": 8, "val": 1, "test": 1}}},
    )
    run = fake_job(wd, "aaaaaaaaaaa6", "train", state="running", created=6000, params={"recipe": r})
    jobs = [j5, j4, j3, j2, j1, run]
    hist = build_history(wd, jobs)
    assert [h["id"] for h in hist] == [
        "aaaaaaaaaaa5",
        "aaaaaaaaaaa4",
        "aaaaaaaaaaa3",
        "aaaaaaaaaaa2",
        "aaaaaaaaaaa1",
    ]  # sin el que corre
    by = {h["id"]: h for h in hist}
    t = by["aaaaaaaaaaa1"]
    assert t["files"] == ["d.csv"] and "/home" not in json.dumps(hist)  # sin rutas del servidor
    assert t["headline"] == {
        "metric_name": "f1_macro",
        "value_val": 0.9,
        "value_test": 0.88,
        "selected_model": "random_forest",
    }
    assert t["models"] == ["decision_tree", "random_forest"] and t["family"] == "classical"
    assert (
        t["duration_s"] == 5.0 and t["hw"] == {"cpu_peak": 80} and t["recipe_sha"] == recipe_sha(r)
    )
    b = by["aaaaaaaaaaa2"]["headline"]
    assert b["selected_model"] == "decision_tree" and b["value_val"] == 0.95
    e = by["aaaaaaaaaaa3"]
    assert e["headline"]["flash_verdict"] == "fits" and e["headline"]["ram_verdict"] == "no_budget"
    assert e["files"] == ["d.csv"] and e["recipe_sha"] == recipe_sha(r)  # heredado de run_job
    f = by["aaaaaaaaaaa4"]
    assert f["state"] == "failed" and f["error"] == "boom" and f["headline"] is None
    assert by["aaaaaaaaaaa5"]["headline"]["split_sizes"]["train"] == 8
    assert (
        build_history(wd, jobs, limit=2)[0]["id"] == "aaaaaaaaaaa5"
        and len(build_history(wd, jobs, 2)) == 2
    )


def test_history_cap_and_hidden_test_jobs(wd):
    jobs = [
        fake_job(
            wd,
            f"{i:012x}",
            "train",
            created=float(i),
            params={"recipe": recipe()},
            result=TRAIN_RES,
        )
        for i in range(210)
    ]
    jobs.append(fake_job(wd, "b" * 12, "_sleep", created=9999.0))
    h = build_history(wd, jobs)
    assert len(h) == 200 and h[0]["id"] == f"{209:012x}"


def test_compare_rows_and_flags(wd):
    r = recipe()
    j1 = fake_job(wd, "c" * 11 + "1", "train", params={"recipe": r}, result=TRAIN_RES)
    j2 = fake_job(wd, "c" * 11 + "2", "benchmark", params={"recipe": r}, result=BENCH_RES)
    rows = compare_rows(j2, BENCH_RES)
    assert [x["label"].split(" · ")[1] for x in rows] == ["decision_tree", "knn"]
    assert rows[0]["metrics"]["size_bytes"] == 63 and rows[1]["metrics"]["size_bytes"] is None
    assert rows[0]["metrics"]["mAP_test"] == 0.9 and set(rows[0]["metrics"]) >= {
        "f1_macro_val",
        "r2_test",
    }
    out = compare_jobs(wd, [j1, j2], {j1["id"]: TRAIN_RES, j2["id"]: BENCH_RES})
    assert len(out["rows"]) == 3 and out["same_dataset"] and out["same_recipe"]
    assert out["rows"][0]["metrics"]["f1_macro_test"] == 0.88
    assert out["rows"][0]["metrics"]["mcc_test"] == pytest.approx(0.83)
    # otra receta sobre el mismo archivo / otro archivo
    r2 = {**recipe(), "target": "other"}
    j3 = fake_job(wd, "c" * 11 + "3", "train", params={"recipe": r2}, result=TRAIN_RES)
    o = compare_jobs(wd, [j1, j3], {j1["id"]: TRAIN_RES, j3["id"]: TRAIN_RES})
    assert o["same_dataset"] and not o["same_recipe"] and o["notes"]
    j4 = fake_job(
        wd, "c" * 11 + "4", "train", params={"recipe": recipe(["/x/y.csv"])}, result=TRAIN_RES
    )
    o = compare_jobs(wd, [j1, j4], {j1["id"]: TRAIN_RES, j4["id"]: TRAIN_RES})
    assert not o["same_dataset"] and not o["same_recipe"]


def test_compare_refuses_more_than_40_rows(wd):
    big = json.loads(json.dumps(BENCH_RES))
    big["results"]["leaderboard"] = [
        {**BENCH_RES["results"]["leaderboard"][0], "name": f"m{i}"} for i in range(30)
    ]
    a = fake_job(wd, "d" * 11 + "1", "benchmark", params={"recipe": recipe()}, result=big)
    b = fake_job(wd, "d" * 11 + "2", "benchmark", params={"recipe": recipe()}, result=big)
    with pytest.raises(JobError) as e:
        compare_jobs(wd, [a, b], {a["id"]: big, b["id"]: big})
    assert e.value.status == 422


# ------------------------------------------------------------------ API con trabajos sintéticos
@pytest.fixture
def api(wd, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    r = recipe([str(data / "d.csv")])
    fake_job(wd, "e" * 11 + "1", "train", created=1, params={"recipe": r}, result=TRAIN_RES)
    fake_job(wd, "e" * 11 + "2", "benchmark", created=2, params={"recipe": r}, result=BENCH_RES)
    fake_job(wd, "e" * 11 + "3", "train", state="failed", created=3, params={"recipe": r})
    app = create_app(TOKEN, [data], wd)
    return TestClient(app, base_url="http://127.0.0.1:8000"), r, data


def test_api_history_params_compare_validation(api):
    c, r, data = api
    assert c.get("/history").status_code == 401
    h = c.get("/history", headers=auth()).json()
    assert [x["id"] for x in h] == ["e" * 11 + "3", "e" * 11 + "2", "e" * 11 + "1"]
    assert h[0]["state"] == "failed" and h[2]["files"] == ["d.csv"]
    p = c.get(f"/jobs/{'e' * 11}1/params", headers=auth()).json()
    assert p["params"]["recipe"] == r and p["files"] == ["d.csv"] and p["kind"] == "train"
    assert c.get("/jobs/zzzzzzzzzzzz/params", headers=auth()).status_code == 404
    ids = ["e" * 11 + "1", "e" * 11 + "2"]
    ok = c.post("/history/compare", headers=auth(), json={"ids": ids})
    assert ok.status_code == 200 and len(ok.json()["rows"]) == 3
    assert set(ok.json()) == {"rows", "same_dataset", "same_recipe", "notes"}
    bad = [
        {"ids": ids[:1]},  # <2
        {"ids": [ids[0], ids[0]]},  # repetidos
        {"ids": [f"{i:012x}" for i in range(7)]},  # >6
        {"ids": [ids[0], "f" * 12]},  # no existe -> 404 (abajo)
        {"ids": [ids[0], "e" * 11 + "3"]},  # fallido -> 409
        {"ids": ids, "extra": 1},
    ]
    codes = [c.post("/history/compare", headers=auth(), json=b).status_code for b in bad]
    assert codes == [422, 422, 422, 404, 409, 422]


def test_api_recipe_transfer_roots_and_result(api):
    c, r, data = api
    old = pd.DataFrame(
        {"f1": np.arange(30.0), "f2": np.arange(30.0) % 3, "label": np.repeat(["a", "b", "c"], 10)}
    )
    old.to_csv(data / "d.csv", index=False)
    rep = inspect_files([data / "d.csv"])
    new = data / "n.csv"
    old.rename(columns={"f1": "F1"}).to_csv(new, index=False)
    out = c.post(
        "/recipe/transfer",
        headers=auth(),
        json={"old_recipe": rep["recipe"], "paths": [str(new)], "lang": "en"},
    )
    assert out.status_code == 200, out.text
    j = out.json()
    assert set(j) == {"report", "recipe", "notes", "applied"}
    assert (
        j["recipe"]["files"] == [str(new.resolve())] and j["report"]["files"][0]["file"] == "n.csv"
    )
    assert j["recipe"]["target"] == "label"
    outside = c.post(
        "/recipe/transfer",
        headers=auth(),
        json={"old_recipe": rep["recipe"], "paths": ["/etc/passwd"]},
    )
    assert outside.status_code == 403
    assert (
        c.post(
            "/recipe/transfer",
            headers=auth(),
            json={"old_recipe": {}, "paths": [str(data / "zz.csv")]},
        ).status_code
        == 404
    )


# ------------------------------------------------------------------ extremo a extremo
def wait(client, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{jid}", headers=auth()).json()
        if j["state"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return j
        time.sleep(0.2)
    raise AssertionError("timeout")


def test_end_to_end_train_history_compare(tmp_path):
    rng = np.random.default_rng(2)
    data = tmp_path / "data"
    data.mkdir()
    n = 300
    base = np.repeat([0.0, 5.0, 10.0], n // 3)
    pd.DataFrame(
        {
            "f1": base + rng.normal(0, 1, n),
            "f2": rng.normal(size=n) + base / 2,
            "label": np.repeat(["a", "b", "c"], n // 3),
        }
    ).to_csv(data / "d.csv", index=False)
    rec = inspect_files([data / "d.csv"])["recipe"]
    c = TestClient(create_app(TOKEN, [data], tmp_path / "ws"), base_url="http://127.0.0.1:8000")
    ids = []
    for models in (["decision_tree"], ["random_forest"]):
        j = c.post(
            "/jobs",
            headers=auth(),
            json={"kind": "train", "params": {"recipe": rec, "models": models}},
        ).json()
        assert wait(c, j["id"])["state"] == "succeeded"
        ids.append(j["id"])
    h = c.get("/history", headers=auth()).json()
    assert [x["id"] for x in h] == ids[::-1]
    assert all(
        x["headline"]["metric_name"] == "f1_macro" and x["headline"]["value_test"] is not None
        for x in h
    )
    assert h[0]["files"] == ["d.csv"] and h[0]["recipe_sha"] == h[1]["recipe_sha"] == recipe_sha(
        rec
    )
    assert str(data) not in json.dumps(h)
    cmp_ = c.post("/history/compare", headers=auth(), json={"ids": ids}).json()
    assert len(cmp_["rows"]) == 2 and cmp_["same_dataset"] and cmp_["same_recipe"]
    assert all(r["metrics"]["f1_macro_test"] is not None for r in cmp_["rows"])
    p = c.get(f"/jobs/{ids[0]}/params", headers=auth()).json()
    assert p["params"]["recipe"]["files"] == rec["files"] and p["recipe_sha"] == recipe_sha(rec)
    # reusar la configuración con otro archivo del mismo esquema
    new = data / "again.csv"
    pd.read_csv(data / "d.csv").sample(frac=1, random_state=1).to_csv(new, index=False)
    t = c.post(
        "/recipe/transfer",
        headers=auth(),
        json={"old_recipe": p["params"]["recipe"], "paths": [str(new)]},
    ).json()
    assert t["recipe"]["target"] == "label" and t["recipe"]["files"] == [str(new.resolve())]
    j = c.post(
        "/jobs",
        headers=auth(),
        json={"kind": "train", "params": {"recipe": t["recipe"], "models": ["decision_tree"]}},
    ).json()
    assert wait(c, j["id"])["state"] == "succeeded"
