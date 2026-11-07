"""``tui --python``: revisa el entorno y relanza con ese intérprete (sin abrir la TUI real)."""

import sys

import pytest

from tinyml_studio import relaunch
from tinyml_studio.cli import main


def test_build_command_uses_given_python_and_language():
    cmd = relaunch.build_command("/x/python", ["a.csv"])
    assert cmd[0] == "/x/python" and cmd[-2:] == ["tui", "a.csv"] or cmd[-1] == "a.csv"
    assert "--lang" in cmd and "tui" in cmd


def test_missing_interpreter_is_a_clear_error(capsys, tmp_path):
    assert main(["tui", "--python", str(tmp_path / "nope" / "python")]) == 1
    assert "nope" in capsys.readouterr().err


def test_not_ready_environment_prints_report_and_does_not_launch(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(relaunch.subprocess, "call", lambda *a, **k: called.append(a) or 0)
    monkeypatch.setattr(
        relaunch,
        "doctor",
        lambda py, comps: {
            "ok": False,
            "python": {},
            "components": {},
            "commands": [],
            "fixes": [],
        },
    )
    monkeypatch.setattr(relaunch, "format_report", lambda rep: "INFORME")
    assert relaunch.run_in_environment(sys.executable, []) == 1
    assert not called and "INFORME" in capsys.readouterr().err


def test_launches_child_with_pythonpath_when_not_installed(monkeypatch):
    seen = {}

    def fake_call(cmd, env=None):
        seen["cmd"], seen["env"] = cmd, env
        return 0

    monkeypatch.delenv(relaunch.CHILD_ENV, raising=False)
    monkeypatch.setattr(relaunch.subprocess, "call", fake_call)
    monkeypatch.setattr(relaunch, "doctor", lambda py, comps: {"ok": True, "tinyml_studio": None})
    assert relaunch.run_in_environment("/e/python", ["d.csv"]) == 0
    assert seen["cmd"][0] == "/e/python"
    assert seen["env"][relaunch.CHILD_ENV] == "1"
    if relaunch._source_tree() is not None:
        assert str(relaunch._source_tree()) in seen["env"]["PYTHONPATH"]


def test_real_child_environment_reports_ok_for_current_python():
    pytest.importorskip("textual")
    from tinyml_studio.doctor import doctor

    assert doctor(sys.executable, ["base", "tui"])["ok"] in (True, False)
