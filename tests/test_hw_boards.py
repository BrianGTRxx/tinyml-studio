from pathlib import Path

import pytest

from tinyml_studio.boards import load_boards
from tinyml_studio.hw import detect as D


def test_detect_without_gpu(monkeypatch):
    monkeypatch.setattr(D.shutil, "which", lambda _x: None)
    monkeypatch.setattr(D, "_amd", lambda: [])  # aislar: la GPU integrada del PC no cuenta
    hw = D.detect_hardware()
    assert hw["gpus"] == [] and hw["suggested_device"] == "cpu"
    assert hw["cpu"]["logical_cores"] >= 1 and hw["ram_gb"] is None or hw["ram_gb"] > 0
    assert any("TinyML" in n or "CPU" in n for n in hw["notes"])


def test_detect_nvidia_with_fake_smi(monkeypatch):
    monkeypatch.setattr(
        D.shutil, "which", lambda x: "/usr/bin/nvidia-smi" if x == "nvidia-smi" else None
    )
    monkeypatch.setattr(
        D, "_run", lambda cmd, timeout=8: "NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 555.85\n"
    )
    hw = D.detect_hardware()
    assert hw["gpus"] == [
        {
            "vendor": "NVIDIA",
            "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "memory_mb": 6141,
            "driver": "555.85",
        }
    ]
    assert hw["suggested_device"] == "gpu" and "PyTorch" in hw["device_note"]


def test_windows_note(monkeypatch):
    monkeypatch.setattr(D.platform, "system", lambda: "Windows")
    monkeypatch.setattr(D.shutil, "which", lambda _x: None)
    monkeypatch.setattr(D, "_run", lambda cmd, timeout=8: "")
    assert any("WSL2" in n for n in D.detect_hardware()["notes"])


def test_unverified_seed_is_flagged(monkeypatch):
    """The raw seed file stays unverified and the loader warns about it (ES and EN)."""
    seed = Path(__file__).resolve().parents[1] / "data" / "boards.seed.yaml"
    monkeypatch.setenv("TINYML_BOARDS_FILE", str(seed))
    b = load_boards()
    esp = next(x for x in b["boards"] if x["id"] == "esp32-s3")
    assert esp["priority"] == 1 and esp["verified"] is False
    assert b["all_verified"] is False and "SIN VERIFICAR" in b["warning"]


def test_default_boards_come_from_verified_fragments():
    b = load_boards()
    assert len(b["boards"]) >= 30 and b["all_verified"] is True and b["warning"] == ""
    esp = next(x for x in b["boards"] if x["id"] == "esp32-s3")
    assert esp["priority"] == 1 and esp["verification"] in ("official_search", "datasheet_read")
    assert esp["sram_kb"] == 512 and esp["evidence_count"] >= 3 and esp["variants"]
    uno = next(x for x in b["boards"] if x["id"] == "arduino-uno-r3")
    assert uno["sram_kb"] == 2 and uno["ai_tier"] in ("none", "classical_only")
    assert {x["family"] for x in b["boards"]} >= {"ESP32", "Raspberry Pi", "STM32", "Arduino"}


def test_boards_file_override(tmp_path, monkeypatch):
    f = tmp_path / "b.yaml"
    f.write_text("boards:\n  - id: x\n    verified: true\n    source: https://ejemplo\n")
    monkeypatch.setenv("TINYML_BOARDS_FILE", str(f))
    b = load_boards()
    assert b["all_verified"] is True and b["warning"] == "" and b["boards"][0]["source"]
    monkeypatch.setenv("TINYML_BOARDS_FILE", str(tmp_path / "no.yaml"))
    with pytest.raises(FileNotFoundError):
        load_boards()


def test_cpu_name_is_human_readable_on_this_machine():
    name = D._cpu_name()
    assert name and "Family" not in name.split(",")[0][:6]  # no "AMD64 Family 25 Model…" en Windows


def test_probe_children_do_not_inherit_stdin(monkeypatch):
    """En Windows, un hijo que hereda la tubería de stdin del sidecar se cuelga: usar DEVNULL."""
    seen = []

    def fake_run(cmd, **kw):
        seen.append(kw.get("stdin"))

        class R:
            returncode = 0
            stdout = '{"importable": true}'
            stderr = ""

        return R()

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    D._probe_script("print(1)")
    D._run(["x"])
    assert seen == [D.subprocess.DEVNULL, D.subprocess.DEVNULL]


def test_device_plan_by_hardware():
    amd = {"vendor": "AMD", "name": "AMD Radeon(TM) Graphics"}
    nv = {"vendor": "NVIDIA", "name": "RTX 4050"}
    assert D._device_plan([], "Windows", False)[0] == "cpu"
    dev, note = D._device_plan([amd], "Windows", False)  # integrada AMD en Windows: CPU, y lo dice
    assert dev == "cpu" and "AMD Radeon" in note and "CPU" in note
    assert D._device_plan([amd], "Linux", True)[0] == "gpu"  # AMD con ROCm en Linux
    dev, note = D._device_plan([nv, amd], "Windows", False)  # NVIDIA gana a la integrada
    assert dev == "gpu" and "RTX 4050" in note and "Windows nativo" in note
