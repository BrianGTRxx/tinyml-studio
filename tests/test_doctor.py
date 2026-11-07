"""Doctor: sondeo y evaluación de entornos Python (el actual y uno externo vacío)."""

import json
import subprocess
import sys

import pytest

from tinyml_studio import doctor as D
from tinyml_studio.cli import main as cli_main
from tinyml_studio.i18n import set_lang


@pytest.fixture(autouse=True)
def _reset_lang():
    yield
    set_lang(None)  # la CLI fija el idioma global: no debe filtrarse a otras pruebas


def test_parse_and_compare_versions():
    assert D.parse_version("2.1.0rc1") == (2, 1, 0)
    assert D.parse_version("2.4.1+cu121") == (2, 4, 1)
    assert D.parse_version("0.dev0") == (0,)
    assert D._cmp((2, 0), (2, 0, 0)) == 0 and D._cmp((1, 9), (2,)) < 0


def _fake(**pkgs):
    return {
        "executable": "/x/bin/python",
        "version": [3, 12, 1],
        "platform": "linux",
        "machine": "x86_64",
        "is_venv": True,
        "is_conda": False,
        "packages": pkgs,
        "tinyml_studio": "0.0.1",
    }


def test_evaluate_statuses_and_fix_commands():
    base = {n: "99.0" for n, _, _ in D.COMPONENTS["base"]["packages"]}
    base.update({"numpy": "1.20.0", "scipy": None, "pandas": "9.0.0"})  # viejo, ausente, muy nuevo
    rep = D.evaluate(_fake(**base))
    st = {r["package"]: r["status"] for r in rep["components"]["base"]["packages"]}
    assert st["numpy"] == "too_old" and st["scipy"] == "missing" and st["pandas"] == "too_new"
    assert not rep["ok"] and rep["python"]["kind"] == "venv"
    assert rep["commands"][0][:4] == ["/x/bin/python", "-m", "pip", "install"]
    assert any(f.startswith("numpy>=1.24") for f in rep["commands"][0][4:])


def test_evaluate_ok_old_python_and_unknown_component():
    full = {n: "99.0" for c in D.COMPONENTS.values() for n, _, _ in c["packages"]}
    full.update({"numpy": "2.0", "pandas": "2.2", "scipy": "1.13", "scikit-learn": "1.5"})
    full.update({"pydantic": "2.5"})
    assert D.evaluate(_fake(**full))["ok"]
    old = {**_fake(**full), "version": [3, 8, 0]}
    rep = D.evaluate(old)
    assert not rep["ok"] and any("3.8.0" in w for w in rep["warnings"])
    with pytest.raises(ValueError):
        D.evaluate(_fake(), ["nope"])


def test_probe_current_and_empty_external_venv(tmp_path):
    cur = D.doctor(None, ["base"])
    assert cur["python"]["version"].startswith(f"{sys.version_info.major}.")
    venv = tmp_path / "v"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    rep = D.doctor(str(py), ["base", "keras"])
    assert rep["python"]["kind"] == "venv" and not rep["ok"]
    assert {f["package"] for f in rep["fixes"]} >= {"numpy", "tensorflow"}
    assert rep["tinyml_studio"] is None and any("tinyml-studio" in w for w in rep["warnings"])
    with pytest.raises(FileNotFoundError):
        D.doctor(str(tmp_path / "no-python"))


def test_cli_doctor_json_and_exit_codes(tmp_path, capsys):
    assert cli_main(["--lang", "en", "doctor", "--json"]) in (0, 1)
    out = json.loads(capsys.readouterr().out)
    assert "components" in out and "python" in out
    assert cli_main(["doctor", "--python", str(tmp_path / "nope")]) == 2
    assert cli_main(["doctor", "--components", "zzz"]) == 2


def test_main_output_is_utf8_even_when_the_console_is_not(monkeypatch):
    """`python -I` ignora PYTHONUTF8: en Windows la salida saldría en cp1252 y Electron mostraría
    un carácter roto en lugar de la "á"."""
    import io

    from tinyml_studio import doctor as d

    raw = io.BytesIO()
    fake = io.TextIOWrapper(raw, encoding="cp1252", newline="")
    monkeypatch.setattr(d.sys, "stdout", fake)
    monkeypatch.setattr(
        d,
        "doctor",
        lambda *a, **k: {"ok": True, "warnings": ["no está instalado — ñ"], "commands": []},
    )
    assert d.main(["--json"]) == 0
    fake.flush()
    assert "no está instalado — ñ" in raw.getvalue().decode("utf-8")


def test_missing_tinyml_studio_hint_does_not_suggest_pypi():
    from tinyml_studio.doctor import evaluate

    probe = {
        "executable": "/x/python",
        "version": [3, 11, 6],
        "implementation": "cpython",
        "platform": "win32",
        "machine": "AMD64",
        "prefix": "/x",
        "is_venv": True,
        "is_conda": False,
        "packages": {},
        "tinyml_studio": None,
    }
    text = " ".join(evaluate(probe, ["base"])["warnings"])
    assert "--no-deps" in text and "pip install tinyml-studio)" not in text


def test_standalone_script_honors_lang_when_package_is_installed():
    """doctor_standalone.py con tinyml_studio instalado (caso de la app) debe respetar --lang."""
    out = subprocess.run(
        [sys.executable, "-I", "packaging/env/doctor_standalone.py", "--lang", "en", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    assert "Core: data" in out and "Núcleo" not in out


def test_tflm_is_skipped_outside_linux_and_cpu_torch_is_flagged():
    probe = {
        "executable": "C:/py/python.exe",
        "version": [3, 11, 6],
        "platform": "win32",
        "is_venv": True,
        "is_conda": False,
        "packages": {"torch": "2.12.0", "tflite-micro": None},
        "torch_build": "2.12.0+cpu",
        "nvidia_smi": True,
        "tinyml_studio": "0.0.1",
    }
    rep = D.evaluate(probe, ["torch", "tflm"])
    assert rep["components"]["tflm"]["ok"] and rep["components"]["tflm"]["available"] is False
    assert not rep["commands"]  # no se pide instalar tflite-micro en Windows
    assert any("2.12.0+cpu" in w and "CPU" in w for w in rep["warnings"])
    probe["torch_build"], probe["platform"] = "2.14.1+cu130", "linux"
    rep = D.evaluate(probe, ["torch", "tflm"])
    assert not any("CPU" in w for w in rep["warnings"])  # build con CUDA: sin aviso
    assert rep["components"]["tflm"]["ok"] is False  # en Linux sí falta tflite-micro
