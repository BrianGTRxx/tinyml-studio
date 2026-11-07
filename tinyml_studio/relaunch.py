"""Lanzar la TUI con el Python de otro entorno (``tinyml-studio tui --python RUTA``).

Un proceso Python no puede cambiar de intérprete mientras corre: se revisa el entorno elegido con
el doctor y se vuelve a lanzar la TUI con *ese* ``python``. Si tinyml-studio no está instalado
allí pero este proceso corre desde el árbol de código (clon del repo), se usa ese árbol vía
``PYTHONPATH``; con una instalación normal hay que instalarlo en el entorno elegido.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tinyml_studio.doctor import doctor, format_report
from tinyml_studio.i18n import get_lang, tr

CHILD_ENV = "TINYML_TUI_CHILD"


def _source_tree() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    return root if (root / "pyproject.toml").is_file() else None


def build_command(python: str, paths: list[str]) -> list[str]:
    return [python, "-m", "tinyml_studio.cli", "--lang", get_lang(), "tui", *paths]


def run_in_environment(python: str, paths: list[str]) -> int:
    """Revisa el entorno y lanza la TUI allí. Devuelve el código de salida."""
    python = python.strip().strip('"')
    if os.environ.get(CHILD_ENV):
        raise RuntimeError(tr("--python no se puede anidar", "--python cannot be nested"))
    rep = doctor(python, ["base", "tui"])
    if not rep["ok"]:
        print(format_report(rep), file=sys.stderr)
        print(
            tr(
                "\nEl entorno no está listo para la TUI. Instala lo que falta con los comandos "
                "de arriba y repite.",
                "\nThe environment is not ready for the TUI. Install what is missing with the "
                "commands above and try again.",
            ),
            file=sys.stderr,
        )
        return 1
    env = dict(os.environ)
    env[CHILD_ENV] = "1"
    env["PYTHONUTF8"] = "1"
    if rep["tinyml_studio"] is None:
        src = _source_tree()
        if src is None:
            print(
                tr(
                    "tinyml-studio no está instalado en {p}. Instálalo allí con la rueda .whl "
                    'de la app o desde el clon: "{p}" -m pip install -e <clon> --no-deps',
                    "tinyml-studio is not installed in {p}. Install it there with the app's .whl "
                    'or from the clone: "{p}" -m pip install -e <clone> --no-deps',
                    p=python,
                ),
                file=sys.stderr,
            )
            return 1
        env["PYTHONPATH"] = os.pathsep.join(
            [str(src), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )
        print(
            tr(
                "Aviso: tinyml-studio no está instalado en ese entorno; se usa el código de {s}.",
                "Note: tinyml-studio is not installed in that environment; using the code in {s}.",
                s=src,
            ),
            file=sys.stderr,
        )
    return subprocess.call(build_command(python, paths), env=env)
