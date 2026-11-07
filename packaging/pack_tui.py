"""Empaqueta la versión de terminal (TUI) como carpeta autónoma: núcleo congelado + lanzador.

Sin instalador ni Python: se descomprime y se ejecuta el lanzador (``TinyML-Studio-TUI.cmd`` en
Windows, ``tinyml-studio-tui.sh`` en Linux), que abre la TUI con ``tinyml-core tui``.

    python packaging/pack_tui.py --platform windows --core packaging/dist/tinyml-core --out tui-dist

Deja en ``--out``:
  ``stage/TinyML-Studio-TUI/``  la carpeta lista; se sube como artefacto (GitHub ya la entrega
                                dentro de un único .zip, así que no lleva otro zip dentro)
  ``archive/TinyML-Studio-TUI-<so>.zip|.tar.gz``  un solo archivo, para los Releases
"""

from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path

NAME = "TinyML-Studio-TUI"

README = """TinyML Studio - version de terminal (TUI)
==========================================
No necesita Python ni instalacion.

{how}

Teclas: 1-8 cambian de pestana | q sale | Ctrl+S guarda la receta.
En Variables: t=objetivo, f=entrada, x=no usar.
Para abrir un dataset directamente: {example}

Aviso: sin firma de codigo ({warn}). Recomendado: Windows Terminal.
Sin verificar en equipos reales (solo lo comprueba el CI).
"""

CMD = """@echo off
chcp 65001 >nul
"%~dp0tinyml-core\\tinyml-core.exe" tui %*
echo.
echo (La TUI se cerro. Si hubo un error, copia el mensaje de arriba.)
pause
"""

SH = """#!/bin/sh
exec "$(dirname "$0")/tinyml-core/tinyml-core" tui "$@"
"""


def build(platform: str, core: Path, out: Path) -> Path:
    exe = "tinyml-core.exe" if platform == "windows" else "tinyml-core"
    if not (core / exe).is_file():
        raise SystemExit(f"no se encuentra {core / exe}")
    stage = out / "stage"
    root = stage / NAME
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(core, root / "tinyml-core")
    if platform == "windows":
        (root / "TinyML-Studio-TUI.cmd").write_text(CMD.replace("\n", "\r\n"), encoding="ascii")
        how = "Doble clic en TinyML-Studio-TUI.cmd (o ejecutalo desde PowerShell)."
        example = 'TinyML-Studio-TUI.cmd "C:\\ruta\\datos.csv"'
        warn = "Windows puede mostrar SmartScreen: Mas informacion > Ejecutar de todas formas"
    else:
        launcher = root / "tinyml-studio-tui.sh"
        launcher.write_text(SH, encoding="ascii")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        how = "Ejecuta ./tinyml-studio-tui.sh en una terminal."
        example = "./tinyml-studio-tui.sh /ruta/datos.csv"
        warn = "el sistema puede pedir confirmacion"
    (root / "LEEME.txt").write_text(
        README.format(how=how, example=example, warn=warn), encoding="utf-8"
    )
    archive = out / "archive"
    archive.mkdir(parents=True)
    base = archive / f"{NAME}-{platform}"
    fmt, ext = ("zip", ".zip") if platform == "windows" else ("gztar", ".tar.gz")
    shutil.make_archive(str(base), fmt, root_dir=stage, base_dir=NAME)
    return Path(str(base) + ext)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--platform", choices=["windows", "linux"], required=True)
    ap.add_argument("--core", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(build(a.platform, a.core, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
