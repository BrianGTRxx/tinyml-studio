"""i18n de la API, los trabajos y la CLI: con ``lang='en'`` no debe colarse español."""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tinyml_studio.api import create_app
from tinyml_studio.cli import main as cli_main
from tinyml_studio.dataset import inspect_files
from tinyml_studio.i18n import set_lang

TOKEN = "i" * 32
ACCENTS = re.compile(r"[áéíóúñ¿¡]")
RNG = np.random.default_rng(5)


@pytest.fixture(autouse=True)
def _reset_lang():
    yield
    set_lang(None)


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    n = 450
    base = np.repeat([0.0, 5.0, 10.0], n // 3)
    df = pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1, n),
            "f2": RNG.normal(size=n) + base / 2,
            "label": np.repeat(["a", "b", "c"], n // 3),
        }
    )
    df.loc[:2, "f1"] = -999  # centinela: genera alertas
    df.to_csv(data / "d.csv", index=False)
    app = create_app(TOKEN, [data], tmp_path / "ws", enable_test_jobs=True)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    return client, data, tmp_path


def _no_spanish(texts):
    assert texts
    for t in texts:
        assert not ACCENTS.search(str(t)), t


def _wait(client, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{jid}", headers=auth()).json()
        if j["state"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return j
        time.sleep(0.2)
    raise AssertionError("timeout")


def test_inspect_and_suggest_cleaning_lang(env):
    client, data, _ = env
    path = str(data / "d.csv")
    es = client.post("/inspect", headers=auth(), json={"paths": [path]}).json()
    en = client.post("/inspect", headers=auth(), json={"paths": [path], "lang": "en"}).json()
    assert es["summary"] != en["summary"] and "Files:" in en["summary"]
    _no_spanish([a["message"] for a in en["alerts"]] + [en["summary"]])
    # ?lang= también vale, y el idioma no se filtra a la petición siguiente
    q = client.post("/inspect?lang=en", headers=auth(), json={"paths": [path]}).json()
    assert q["summary"] == en["summary"]
    assert client.post("/inspect", headers=auth(), json={"paths": [path]}).json() == es

    body = {"columns": en["columns"], "alerts": en["alerts"], "multi_file": False}
    s_es = client.post("/suggest-cleaning", headers=auth(), json=body).json()["steps"]
    s_en = client.post("/suggest-cleaning", headers=auth(), json={**body, "lang": "en"}).json()[
        "steps"
    ]
    assert s_es and [s["why"] for s in s_es] != [s["why"] for s in s_en]
    _no_spanish([s["why"] for s in s_en])

    # errores de la API en inglés
    r = client.post("/inspect", headers=auth(), json={"paths": ["/etc/passwd"], "lang": "en"})
    assert r.status_code == 403 and r.json()["detail"] == "path outside the allowed folders"
    r = client.post("/inspect", headers=auth(), json={"paths": ["/etc/passwd"]})
    assert r.json()["detail"] == "ruta fuera de las carpetas permitidas"
    r = client.get("/jobs/nope?lang=en", headers=auth())
    assert r.status_code == 404 and r.json()["detail"] == "job not found"
    r = client.get("/jobs?lang=en")  # sin token
    assert r.status_code == 401 and r.json()["detail"] == "unauthorized"


def test_catalog_endpoints_lang(env, monkeypatch):
    client, *_ = env
    seed = Path(__file__).resolve().parents[1] / "data" / "boards.seed.yaml"
    monkeypatch.setenv("TINYML_BOARDS_FILE", str(seed))  # the unverified seed triggers the warning
    b_es = client.get("/boards", headers=auth()).json()
    b_en = client.get("/boards?lang=en", headers=auth()).json()
    assert "SIN VERIFICAR" in b_es["warning"] and "NOT VERIFIED" in b_en["warning"]
    h_en = client.get("/hardware?lang=en", headers=auth()).json()
    _no_spanish([h_en["device_note"], *h_en["notes"]])
    m_es = client.get("/models", headers=auth()).json()
    m_en = client.get("/models?lang=en", headers=auth()).json()
    assert m_es["classical"]["models"] == m_en["classical"]["models"]
    assert m_es["classical"]["label"] != m_en["classical"]["label"]
    _no_spanish([v.get("label", "") + v.get("notes", "") for v in m_en.values()])


def test_train_and_export_job_in_english(env):
    client, data, _ = env
    recipe = inspect_files([data / "d.csv"])["recipe"]
    r = client.post(
        "/jobs",
        headers=auth(),
        json={
            "kind": "train",
            "params": {"recipe": recipe, "models": ["decision_tree"], "seed": 0, "lang": "en"},
        },
    )
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    with client.stream("GET", f"/jobs/{jid}/events", headers=auth()) as ev:
        body = "".join(ev.iter_text())
    events = [json.loads(ln[6:]) for ln in body.splitlines() if ln.startswith("data: ")]
    msgs = [e["msg"] for e in events if e.get("msg")]
    assert any(m.startswith("Training decision_tree") for m in msgs)
    _no_spanish(msgs)
    assert _wait(client, jid)["state"] == "succeeded"
    res = client.get(f"/jobs/{jid}/result", headers=auth()).json()
    assert res["summary"] == res["summary_es"] and "Best by" in res["summary"]
    _no_spanish([res["summary"], *res["results"]["warnings"]])

    r = client.post(
        "/jobs?lang=en",
        headers=auth(),
        json={"kind": "export_c", "params": {"run_job": jid, "flash_kb": 512, "ram_kb": 128}},
    )
    assert r.status_code == 202, r.text
    ex = r.json()["id"]
    assert _wait(client, ex)["state"] == "succeeded"
    rep = client.get(f"/jobs/{ex}/result", headers=auth()).json()
    assert "Exported to the output folder" in rep["summary"]
    b = rep["results"]["budget"]["flash_data"]
    assert b["verdict"] == "cabe" and b["verdict_code"] == "fits" and b["verdict_label"] == "fits"
    _no_spanish([rep["summary"], rep["results"]["budget"]["note"]])

    # error de validación y de estado en inglés
    r = client.post(
        "/jobs",
        headers=auth(),
        json={"kind": "train", "params": {"recipe": recipe, "bogus": 1, "lang": "en"}},
    )
    assert r.status_code == 422 and r.json()["detail"].startswith("invalid parameters")


def test_job_default_language_is_spanish(env):
    client, data, _ = env
    recipe = inspect_files([data / "d.csv"])["recipe"]
    r = client.post(
        "/jobs",
        headers=auth(),
        json={"kind": "train", "params": {"recipe": recipe, "models": ["decision_tree"]}},
    )
    jid = r.json()["id"]
    assert _wait(client, jid)["state"] == "succeeded"
    res = client.get(f"/jobs/{jid}/result", headers=auth()).json()
    assert "Mejor por" in res["summary_es"] and res["summary"] == res["summary_es"]


def test_cli_lang_en(env, capsys):
    _, data, tmp = env
    csv = str(data / "d.csv")
    assert cli_main(["--lang", "en", "inspect", csv, "--recipe-out", str(tmp / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "Files:" in out
    _no_spanish([out])
    set_lang(None)
    assert (
        cli_main(["--lang", "en", "train", str(tmp / "r.json"), "--models", "decision_tree"]) == 0
    )
    out = capsys.readouterr().out
    assert "Best by" in out and "Baseline" in out
    _no_spanish([out])
    set_lang(None)
    with pytest.raises(SystemExit):
        cli_main(["--lang", "en", "train", "--help"])
    _no_spanish([capsys.readouterr().out])
    set_lang(None)
    assert cli_main(["train", str(tmp / "r.json"), "--models", "decision_tree"]) == 0
    assert "Mejor por" in capsys.readouterr().out


def test_job_result_carries_the_spanish_english_pairs_it_said(tmp_path):
    """Para cambiar de idioma sin repetir el trabajo se guardan los pares [es, en] dichos."""
    import os

    from tinyml_studio.i18n import collected_pairs, set_lang, tr

    os.environ["TINYML_JOB"] = "1"
    try:
        set_lang("es")
        a = tr("hay {n} filas sin etiqueta", "there are {n} rows without a label", n=3)
        set_lang("en")
        b = tr("hay {n} filas sin etiqueta", "there are {n} rows without a label", n=3)
    finally:
        os.environ.pop("TINYML_JOB", None)
        set_lang(None)
    assert a == "hay 3 filas sin etiqueta" and b == "there are 3 rows without a label"
    assert ["hay 3 filas sin etiqueta", "there are 3 rows without a label"] in collected_pairs()
