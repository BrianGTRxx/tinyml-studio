"""Despacho de subcomandos del ejecutable congelado, sin PyInstaller."""

import subprocess
import sys

import pytest

from tinyml_studio import core_main


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "tinyml_studio.core_main", *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_run_module_behaves_like_python_dash_m():
    r = _run("run-module", "json.tool", "--help")
    assert r.returncode == 0 and "usage" in r.stdout.lower()


def test_run_module_passes_exit_code_and_argv(tmp_path):
    (tmp_path / "m.py").write_text(
        "import sys\nprint(__name__, sys.argv[1:])\nraise SystemExit(7)\n", encoding="utf-8"
    )
    r = subprocess.run(
        [sys.executable, "-m", "tinyml_studio.core_main", "run-module", "m", "a", "-b"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "PATH": ""},
        timeout=60,
    )
    assert r.returncode == 7
    assert r.stdout.strip() == "__main__ ['a', '-b']"


def test_run_code_sets_argv_like_dash_c():
    r = _run("run-code", "import sys; print(sys.argv)", "x", "y")
    assert r.returncode == 0 and r.stdout.strip() == "['-c', 'x', 'y']"


def test_run_code_error_exit_codes():
    assert _run("run-code", "raise SystemExit(3)").returncode == 3
    assert _run("run-code", "raise SystemExit('boom')").returncode == 1
    r = _run("run-code", "import modulo_que_no_existe")
    assert r.returncode != 0 and "modulo_que_no_existe" in r.stderr


def test_usage_error_and_default_is_serve(monkeypatch):
    assert core_main.main(["nada"]) == 2
    assert core_main.main(["run-module"]) == 2
    called = []
    monkeypatch.setattr(core_main, "_serve", lambda: called.append(1) or 0)
    assert core_main.main([]) == 0 and core_main.main(["serve"]) == 0
    assert len(called) == 2


def test_serve_without_token_fails_cleanly(monkeypatch):
    monkeypatch.delenv("TINYML_SESSION_TOKEN", raising=False)
    assert core_main.main(["serve"]) == 2


@pytest.mark.parametrize("code,expected", [(None, 0), (4, 4), ("x", 1)])
def test_exit_code(code, expected, capsys):
    assert core_main._exit_code(code) == expected
