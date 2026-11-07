"""Qué trae esta compilación del núcleo (fuente, lite o full), sin importar TF ni torch."""

from __future__ import annotations

import importlib.util
import sys

from tinyml_studio.i18n import tr
from tinyml_studio.resources import resource_path


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_keras() -> bool:
    """Keras entrenable aquí: hacen falta keras y TensorFlow (backend por defecto y TFLite)."""
    return _has("keras") and _has("tensorflow")


def has_torch() -> bool:
    return _has("torch")


def variant() -> str:
    """ "source" (python -m), o la variante empaquetada: "lite" | "full"."""
    if not getattr(sys, "frozen", False):
        return "source"
    try:
        v = resource_path("tinyml_variant.txt").read_text(encoding="utf-8").strip()
        if v in ("lite", "full"):
            return v
    except OSError:
        pass
    return "full" if has_keras() else "lite"


def environment() -> dict:
    return {
        "keras": has_keras(),
        "torch": has_torch(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "variant": variant(),
    }


def keras_missing_message() -> str:
    return tr(
        "Esta compilación no incluye Keras/TensorFlow: Instala la compilación completa "
        "(full) o ejecuta desde el código fuente.",
        "This build does not include Keras/TensorFlow: Install the full build or run from source.",
    )
