"""Rutas a los recursos del proyecto (``data/``, ``docs/``…), también congelado con PyInstaller."""

from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    """Raíz con ``data/``.

    ``sys._MEIPASS`` si está congelado; ``tinyml_studio/_data`` si está instalado desde una rueda
    (la crea ``packaging/build_wheel.py``); la raíz del repo desde el código fuente.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", None) or Path(sys.executable).parent)
    packaged = Path(__file__).resolve().parent / "_data"
    if (packaged / "data").is_dir():
        return packaged
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)
