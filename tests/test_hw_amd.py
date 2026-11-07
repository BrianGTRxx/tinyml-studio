import json

import pytest

from tinyml_studio.hw import detect as D

# Salidas SIMULADAS con el formato de las herramientas reales (no se probó con hardware AMD).
ROCM_SMI = json.dumps(
    {
        "card0": {
            "Card Series": "Radeon RX 7900 XTX",
            "Card Model": "0x744c",
            "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
            "VRAM Total Memory (B)": "25753026560",
            "VRAM Total Used Memory (B)": "11534336",
        },
        "card1": {"Card series": "Radeon 780M", "VRAM Total Memory (B)": "536870912"},
        "system": {"Driver version": "6.8.5"},
    }
)
ROCMINFO = """ROCk module is loaded
Agent 1
*******
  Name:                    AMD Ryzen 7 7735HS
  Marketing Name:          AMD Ryzen 7 7735HS with Radeon Graphics
  Device Type:             CPU
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
"""
LSPCI = (
    "00:02.0 Host bridge [0600]: Intel Corporation Device [8086:1234]\n"
    "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
    "Navi 31 [Radeon RX 7900 XT/7900 XTX] [1002:744c] (rev c8)\n"
    "04:00.0 Audio device [0403]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 HDMI [1002:ab30]\n"
)
PS_OUT = (
    "NVIDIA GeForce RTX 4050|6442450944\n"
    "AMD Radeon RX 6800 XT|4293918720\n"
    "AMD Radeon(TM) Graphics|536870912\n"
)
WMIC = "Node,AdapterRAM,Name\r\nPC,536870912,AMD Radeon(TM) Graphics\r\nPC,1073741824,Intel UHD\r\n"


def test_parse_rocm_smi_json():
    g = D.parse_rocm_smi_json(ROCM_SMI)
    assert [(x["name"], x["vram_mb"]) for x in g] == [
        ("Radeon RX 7900 XTX", 24560),
        ("Radeon 780M", 512),
    ]
    assert g[0]["vendor"] == "AMD" and g[0]["rocm_present"] is True and g[0]["memory_mb"] == 24560
    assert D.parse_rocm_smi_json("no es json") == [] and D.parse_rocm_smi_json("[]") == []


def test_parse_rocminfo_only_gpu_agents():
    g = D.parse_rocminfo(ROCMINFO)
    assert [x["name"] for x in g] == ["AMD Radeon RX 7900 XTX"] and g[0]["vram_mb"] is None


def test_parse_lspci_only_amd_graphics():
    g = D.parse_lspci(LSPCI)
    assert len(g) == 1 and "Radeon RX 7900" in g[0]["name"] and g[0]["rocm_present"] is False


def test_parse_sysfs(tmp_path):
    for card, vendor, vram in (("card0", "0x1002", "8589934592"), ("card1", "0x10de", None)):
        d = tmp_path / card / "device"
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor + "\n")
        if vram:
            (d / "mem_info_vram_total").write_text(vram + "\n")
    (tmp_path / "card0-HDMI-A-1").mkdir()
    g = D.parse_sysfs_amd(str(tmp_path))
    assert len(g) == 1 and g[0]["vram_mb"] == 8192
    assert D.parse_sysfs_amd(str(tmp_path / "nada")) == []


def test_parse_windows_powershell_and_wmic():
    g = D.parse_windows_video(PS_OUT)
    assert [x["name"] for x in g] == ["AMD Radeon RX 6800 XT", "AMD Radeon(TM) Graphics"]
    assert g[0]["vram_mb"] is None  # AdapterRAM de 32 bits topado en ~4 GB: no es fiable
    assert g[1]["vram_mb"] == 512
    w = D.parse_windows_video(WMIC)
    assert [(x["name"], x["vram_mb"]) for x in w] == [("AMD Radeon(TM) Graphics", 512)]


def _fake(monkeypatch, tools, outputs, system="Linux"):
    monkeypatch.setattr(D.platform, "system", lambda: system)
    monkeypatch.setattr(D.shutil, "which", lambda x: f"/usr/bin/{x}" if x in tools else None)
    monkeypatch.setattr(D, "_run", lambda cmd, timeout=8: outputs.get(cmd[0]))
    monkeypatch.setattr(D.os.path, "isdir", lambda p: False)


def test_amd_via_rocm_smi(monkeypatch):
    _fake(monkeypatch, {"rocm-smi"}, {"rocm-smi": ROCM_SMI})
    hw = D.detect_hardware()
    assert [g["name"] for g in hw["amd_gpus"]][0] == "Radeon RX 7900 XTX"
    assert hw["rocm_present"] is True and hw["frameworks"] is None
    assert hw["gpus"][0]["vendor"] == "AMD" and any(n for n in hw["notes"])
    assert {g["id"] for g in hw["guidance"]} >= {
        "nvidia_cuda",
        "amd_rocm_linux",
        "wsl2",
        "windows_tf",
    }


def test_amd_falls_back_to_rocminfo_then_lspci(monkeypatch):
    _fake(monkeypatch, {"rocm-smi", "rocminfo"}, {"rocm-smi": "", "rocminfo": ROCMINFO})
    assert D._amd()[0]["source"] == "rocminfo"
    _fake(monkeypatch, {"lspci"}, {"lspci": LSPCI})
    monkeypatch.setattr(D, "parse_sysfs_amd", lambda root="/sys/class/drm", rocm=False: [])
    g = D._amd()
    assert g[0]["source"] == "lspci" and g[0]["rocm_present"] is False


def test_amd_windows_powershell_then_wmic(monkeypatch):
    _fake(monkeypatch, set(), {"powershell": PS_OUT}, system="Windows")
    assert len(D._amd()) == 2
    _fake(monkeypatch, set(), {"powershell": None, "wmic": WMIC}, system="Windows")
    assert [g["name"] for g in D._amd()] == ["AMD Radeon(TM) Graphics"]


def test_probe_never_raises_and_parses(monkeypatch):
    def boom(*a, **k):
        raise D.subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(D.subprocess, "run", boom)
    r = D._probe_script("print(1)")
    assert r["importable"] is None and r["gpu_count"] == 0 and "timeout" in r["error"]


def test_probe_real_subprocess_cpu_only():
    # sin GPU en este entorno: el sondeo real de torch responde y no ve ninguna
    pytest.importorskip("torch")
    D._probe_cache.clear()
    info = D.probe_framework("torch")
    assert info["importable"] is True and info["gpu_count"] == 0 and "version" in info


def test_detect_hardware_with_frameworks(monkeypatch):
    monkeypatch.setattr(
        D, "probe_framework", lambda n, refresh=False: {"importable": True, "gpu_count": 1}
    )
    monkeypatch.setattr(D.shutil, "which", lambda _x: None)
    hw = D.detect_hardware(probe_frameworks_too=True)
    assert hw["frameworks"]["torch"]["gpu_count"] == 1
