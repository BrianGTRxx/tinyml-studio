"""Monitor de hardware: sampler, parseo de nvidia-smi, recorte de historial, API y ETA."""

import json
import os
import stat
import time

import pytest
from fastapi.testclient import TestClient

from tinyml_studio.api import create_app
from tinyml_studio.hw import monitor as M

TOKEN = "m" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}

FAKE_SMI = """#!/bin/sh
echo "0, NVIDIA GeForce RTX 4050 Laptop GPU, 37, 1200, 6141, 55, 23.45"
echo "1, Fake, GPU, With Comma, [N/A], 100, 2000, N/A, [Not Supported]"
"""


@pytest.fixture
def fake_smi(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    f = d / "nvidia-smi"
    f.write_text(FAKE_SMI)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.skipif(os.name == "nt", reason="script sh falso")
def test_sampler_with_fake_nvidia_smi(fake_smi):
    s = M.HardwareSampler(os.getpid()).sample()
    json.dumps(s)  # JSON-safe
    assert s["gpus"][0] == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "util_pct": 37.0,
        "mem_used_mb": 1200.0,
        "mem_total_mb": 6141.0,
        "temp_c": 55.0,
        "power_w": 23.45,
    }
    g1 = s["gpus"][1]
    assert g1["name"] == "Fake, GPU, With Comma" or g1["name"].startswith("Fake")
    assert g1["util_pct"] is None and g1["temp_c"] is None and g1["power_w"] is None
    assert g1["mem_used_mb"] == 100.0
    assert s["proc_rss_mb"] > 0 and 0 <= s["cpu_pct"] <= 100 and s["cores"] >= 1


def test_parse_tolerant():
    assert M.parse_nvidia_smi("") == []
    assert M.parse_nvidia_smi("basura\nx, y\n") == []
    g = M.parse_nvidia_smi("0, A, [Not Supported], N/A, 8000, 40, [N/A]")[0]
    assert g["util_pct"] is None and g["mem_total_mb"] == 8000.0 and g["power_w"] is None


def test_sampler_without_temps_or_gpu(monkeypatch):
    monkeypatch.setattr(M.shutil, "which", lambda _x: None)
    monkeypatch.delattr(M.psutil, "sensors_temperatures", raising=False)
    from tinyml_studio.hw import cputemp

    monkeypatch.setattr(cputemp, "read_cpu_temp", lambda force=False: (None, None))  # sin sensores
    s = M.HardwareSampler().sample()
    assert s["cpu_temp_c"] is None and s["gpus"] == []
    assert s["proc_cpu_pct"] is None and s["proc_rss_mb"] is None  # sin pid
    notes = M.notes_for(s)
    assert any("NVIDIA" in n for n in notes) and any("scikit-learn" in n for n in notes)
    assert any("temperatura" in n.lower() or "temperature" in n.lower() for n in notes)


def test_temp_from_psutil(monkeypatch):
    class T:
        def __init__(self, c):
            self.current = c

    from tinyml_studio.hw import cputemp

    monkeypatch.setattr(cputemp.os, "name", "posix")
    monkeypatch.setattr(
        M.psutil, "sensors_temperatures", lambda: {"coretemp": [T(50.0), T(61.5)]}, raising=False
    )
    assert M.HardwareSampler._cpu_temp() == 61.5


def test_windows_note(monkeypatch):
    from tinyml_studio.hw import cputemp

    monkeypatch.setattr(cputemp.os, "name", "nt")
    notes = M.notes_for({"gpus": [], "cpu_temp_c": None})
    zone = ("zona térmica", "thermal zone")
    assert any("Windows" in n and any(z in n.lower() for z in zone) for n in notes)




def test_windows_without_any_sensor_is_none(monkeypatch):
    from tinyml_studio.hw import cputemp

    monkeypatch.setattr(cputemp.os, "name", "nt")
    monkeypatch.setattr(cputemp, "_cache", {"t": 0.0, "value": None, "source": None})
    monkeypatch.setattr(cputemp, "_powershell", lambda script: None)
    assert cputemp.read_cpu_temp(force=True) == (None, None)


def test_cap_lines_and_downsample():
    lines = [str(i) for i in range(5001)]
    out = M.cap_lines(lines, 5000, 1000)
    assert len(out) < 5000 and out[-1] == "5000"
    assert out[-1000:] == lines[-1000:]  # lo reciente intacto
    assert out[:3] == ["0", "2", "4"]  # lo antiguo, una de cada dos
    assert M.cap_lines(lines[:10], 5000) == lines[:10]
    ds = M.downsample_even(list(range(1000)), 600)
    assert len(ds) == 600 and ds[0] == 0 and ds[-1] == 999
    assert M.downsample_even([1, 2], 600) == [1, 2] and M.downsample_even([1, 2], 0) == []


def test_recorder_compacts_file(tmp_path):
    rec = M.JobHwRecorder(os.getpid(), tmp_path / "hw.jsonl", interval=0.0, max_lines=30)
    rec.start()
    time.sleep(1.0)
    summ = rec.stop()
    lines = (tmp_path / "hw.jsonl").read_text().splitlines()
    assert 0 < len(lines) <= 31 and all(json.loads(x)["t"] for x in lines)
    assert summ["samples"] >= len(lines)  # el resumen es exacto aunque se recorte


def test_summary():
    ss = [
        {"t": 0, "cpu_pct": 10, "proc_cpu_pct": 5, "proc_rss_mb": 100, "ram_used_mb": 1000,
         "cpu_temp_c": None, "gpus": [{"index": 0, "name": "G", "util_pct": 20, "mem_used_mb": 5,
                                       "temp_c": 40, "power_w": None}]},
        {"t": 4, "cpu_pct": 30, "proc_cpu_pct": 15, "proc_rss_mb": 300, "ram_used_mb": 900,
         "cpu_temp_c": 70, "gpus": [{"index": 0, "name": "G", "util_pct": 60, "mem_used_mb": 9,
                                     "temp_c": 50, "power_w": None}]},
    ]  # fmt: skip
    r = M.summarize(ss)
    assert r["duration_s"] == 4 and r["cpu_mean_pct"] == 20 and r["cpu_peak_pct"] == 30
    assert r["rss_peak_mb"] == 300 and r["ram_peak_mb"] == 1000 and r["cpu_temp_peak_c"] == 70
    g = r["gpus"][0]
    assert g["util_mean_pct"] == 40 and g["mem_peak_mb"] == 9 and g["temp_peak_c"] == 50
    assert g["power_peak_w"] is None


@pytest.fixture
def client(tmp_path):
    app = create_app(TOKEN, [tmp_path], tmp_path / "ws", enable_test_jobs=True)
    return TestClient(app, base_url="http://127.0.0.1:8000")


def test_hardware_live(client):
    assert client.get("/hardware/live").status_code == 401
    r = client.get("/hardware/live", headers=AUTH)
    assert r.status_code == 200
    j = r.json()
    assert j["cpu_pct"] >= 0 and j["ram_total_mb"] > 0 and isinstance(j["notes"], list)


def test_job_hw_end_to_end(client, tmp_path):
    r = client.post("/jobs", headers=AUTH, json={"kind": "_sleep", "params": {"seconds": 3.5}})
    jid = r.json()["id"]
    assert client.get(f"/jobs/{jid}/hw").status_code == 401
    assert client.get("/jobs/zzzzzzzzzzzz/hw", headers=AUTH).status_code == 404
    seen_eta = None
    deadline = time.time() + 120
    while time.time() < deadline:
        j = client.get(f"/jobs/{jid}", headers=AUTH).json()
        assert "elapsed_s" in j and "eta_s" in j and "hw_summary" in j
        if j["state"] == "running" and j["eta_s"] is not None:
            seen_eta = j["eta_s"]
        if j["state"] in ("succeeded", "failed"):
            break
        time.sleep(0.3)
    assert j["state"] == "succeeded"
    assert seen_eta is not None and seen_eta >= 0
    assert j["eta_s"] is None and j["elapsed_s"] > 2
    hw = client.get(f"/jobs/{jid}/hw", headers=AUTH).json()
    assert hw["running"] is False and len(hw["samples"]) >= 2
    assert hw["samples"][0]["proc_rss_mb"] is not None
    assert hw["summary"]["samples"] >= 2 and hw["summary"]["rss_peak_mb"] > 0
    assert j["hw_summary"]["cpu_peak_pct"] is not None and hw["notes"]
    assert (tmp_path / "ws" / "jobs" / jid / "hw.jsonl").is_file()
    stored = json.loads((tmp_path / "ws" / "jobs" / jid / "job.json").read_text())
    assert stored["hw_summary"]["samples"] == hw["summary"]["samples"]


def test_hw_live_while_running_and_queued_eta(client):
    jid = client.post(
        "/jobs", headers=AUTH, json={"kind": "_sleep", "params": {"seconds": 30}}
    ).json()["id"]
    time.sleep(2.5)
    hw = client.get(f"/jobs/{jid}/hw", headers=AUTH).json()
    assert hw["running"] is True and hw["summary"]["samples"] >= 1
    client.post(f"/jobs/{jid}/cancel", headers=AUTH)
    for _ in range(100):
        if client.get(f"/jobs/{jid}", headers=AUTH).json()["state"] == "cancelled":
            break
        time.sleep(0.1)
    time.sleep(0.5)
    j = client.get(f"/jobs/{jid}", headers=AUTH).json()
    assert j["eta_s"] is None


def test_eta_rules(tmp_path):
    from tinyml_studio.api.jobs import JobManager

    m = JobManager(tmp_path, lambda _p: None, enable_test_jobs=True)
    job = {"id": "a" * 12, "state": "running", "started": time.time() - 10,
           "progress": {"pct": 4}}  # fmt: skip
    m._jobs[job["id"]] = job
    m._update_eta(job["id"])
    assert m._timing(job)[1] is None  # pct < 5
    job["progress"]["pct"] = 50
    m._update_eta(job["id"])
    elapsed, eta = m._timing(job)
    assert 9 < elapsed < 12 and 8 < eta < 12  # 50 % a los 10 s -> ~10 s restantes
    job["state"] = "failed"
    assert m._timing(job)[1] is None






def test_windows_thermal_zone_works_without_admin_or_extra_programs(monkeypatch):
    from tinyml_studio.hw import cputemp

    monkeypatch.setattr(cputemp.os, "name", "nt")
    monkeypatch.setattr(cputemp, "_cache", {"t": 0.0, "value": None, "source": None})
    calls = []

    def fake(script):
        calls.append(script)
        return 3502 if "ThermalZone" in script else None  # 350.2 K

    monkeypatch.setattr(cputemp, "_powershell", fake)
    assert cputemp.read_cpu_temp(force=True) == (77.1, "Thermal zone")


def test_zone_conversion_and_note():
    from tinyml_studio.hw import monitor as M
    from tinyml_studio.hw.cputemp import zone_to_celsius

    assert zone_to_celsius([3502, 3000]) == 77.1
    assert zone_to_celsius(0) is None and zone_to_celsius(None) is None
    notes = M.notes_for({"gpus": [], "cpu_temp_c": 77.1, "cpu_temp_source": "Thermal zone"})
    assert any("approximate" in n.lower() or "aproximada" in n.lower() for n in notes)
