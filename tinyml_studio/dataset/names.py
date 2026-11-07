"""Utilidades de nombres de columnas/archivos (sin conocimiento de ningún dataset concreto)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


def normalize_name(name: str) -> str:
    """minúsculas, sin acentos, no alfanumérico -> ``_``. ``Ángulo Fase`` -> ``angulo_fase``."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def name_tokens(name: str) -> set[str]:
    return {t for t in normalize_name(name).split("_") if t}


def label_from_filename(path: str | Path) -> str:
    """Etiqueta sugerida del archivo: ``Mi_Aparato1.csv`` -> ``mi_aparato``."""
    stem = normalize_name(Path(path).stem)
    return re.sub(r"(_?\d+)+$", "", stem) or stem
