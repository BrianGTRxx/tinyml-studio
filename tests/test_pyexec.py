"""python_argv: subprocesos del núcleo, con y sin PyInstaller (sys.frozen simulado)."""

import sys

import pytest

from tinyml_studio.pyexec import python_argv


def test_source_mode_is_the_classic_command_line():
    assert python_argv(module="a.b", args=["x", "y"]) == [sys.executable, "-m", "a.b", "x", "y"]
    assert python_argv(code="print(1)") == [sys.executable, "-c", "print(1)"]
    assert python_argv(code="c", args=["1"], isolated=True) == [
        sys.executable,
        "-I",
        "-c",
        "c",
        "1",
    ]
    assert python_argv(code="c", isolated=False) == [sys.executable, "-c", "c"]


def test_frozen_mode_relaunches_itself_with_subcommands(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert python_argv(module="a.b", args=["x"]) == [sys.executable, "run-module", "a.b", "x"]
    assert python_argv(code="c", args=("1", "2")) == [sys.executable, "run-code", "c", "1", "2"]
    # -I no existe congelado: no hay intérprete que aislar
    assert python_argv(code="c", isolated=True) == [sys.executable, "run-code", "c"]


def test_exactly_one_of_module_or_code():
    with pytest.raises(ValueError):
        python_argv()
    with pytest.raises(ValueError):
        python_argv(module="m", code="c")
