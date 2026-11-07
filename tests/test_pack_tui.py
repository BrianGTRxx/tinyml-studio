"""Empaquetado de la TUI autónoma (sin congelar nada de verdad: un núcleo falso)."""

import importlib.util
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pack_tui", Path(__file__).resolve().parents[1] / "packaging" / "pack_tui.py"
)
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


def fake_core(tmp_path, exe):
    core = tmp_path / "tinyml-core"
    (core / "_internal").mkdir(parents=True)
    (core / exe).write_bytes(b"x")
    (core / "_internal" / "lib.bin").write_bytes(b"y")
    return core


def test_windows_zip_has_launcher_core_and_readme(tmp_path):
    out = pack.build("windows", fake_core(tmp_path, "tinyml-core.exe"), tmp_path / "out")
    assert out.name == "TinyML-Studio-TUI-windows.zip"
    staged = tmp_path / "out" / "stage" / "TinyML-Studio-TUI"
    assert (staged / "TinyML-Studio-TUI.cmd").is_file()  # la carpeta lista, sin zip dentro
    assert not list(staged.rglob("*.zip"))
    names = set(zipfile.ZipFile(out).namelist())
    assert "TinyML-Studio-TUI/TinyML-Studio-TUI.cmd" in names
    assert "TinyML-Studio-TUI/tinyml-core/tinyml-core.exe" in names
    assert "TinyML-Studio-TUI/LEEME.txt" in names
    cmd = zipfile.ZipFile(out).read("TinyML-Studio-TUI/TinyML-Studio-TUI.cmd").decode()
    assert "tinyml-core.exe" in cmd and "tui %*" in cmd and "\r\n" in cmd
    assert "pause" in cmd  # la ventana no se cierra sola: así se ve cualquier error


@pytest.mark.skipif(sys.platform == "win32", reason="NTFS no guarda el bit de ejecución")
def test_linux_tar_keeps_executable_launcher(tmp_path):
    out = pack.build("linux", fake_core(tmp_path, "tinyml-core"), tmp_path / "out")
    assert out.name == "TinyML-Studio-TUI-linux.tar.gz"
    with tarfile.open(out) as t:
        m = t.getmember("TinyML-Studio-TUI/tinyml-studio-tui.sh")
        assert m.mode & 0o111
        assert "TinyML-Studio-TUI/tinyml-core/tinyml-core" in t.getnames()


def test_missing_core_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        pack.build("windows", tmp_path / "nada", tmp_path / "out")
