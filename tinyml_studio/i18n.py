"""Mensajes en español e inglés.

Los mensajes visibles para el usuario (alertas, motivos, advertencias, resúmenes) se escriben una
vez en cada idioma: ``tr("texto en español {x}", "english text {x}", x=...)``. El idioma sale de
``set_lang`` (la API/trabajos lo fijan por petición) o de la variable ``TINYML_LANG``; por defecto
es español. No afecta a los nombres de columnas ni a los datos del usuario.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

SUPPORTED = ("es", "en")
_LANG: ContextVar[str] = ContextVar("tinyml_lang", default="")


def _default() -> str:
    env = os.environ.get("TINYML_LANG", "es").lower()
    return env if env in SUPPORTED else "es"


def set_lang(lang: str | None) -> None:
    _LANG.set(lang if lang in SUPPORTED else "")


def get_lang() -> str:
    return _LANG.get() or _default()


# Pares (español, inglés) ya formateados que se dijeron en este proceso. Solo se guardan
# dentro de un trabajo (TINYML_JOB=1): al terminar se adjuntan al resultado (``_i18n``) para
# que la interfaz cambie de idioma sin volver a ejecutar el trabajo.
_PAIRS: dict[str, str] = {}
_PAIRS_MAX = 4000
_PAIRS_MIN_LEN = 10


def _format(s: str, kw: dict) -> str:
    try:
        return s.format(**kw) if kw else s
    except (KeyError, IndexError, ValueError):
        return s


def tr(es: str, en: str, **kw) -> str:
    """Devuelve el mensaje en el idioma activo; ``kw`` se aplica con ``str.format``."""
    out = _format(en if get_lang() == "en" else es, kw)
    if os.environ.get("TINYML_JOB") == "1" and len(_PAIRS) < _PAIRS_MAX:
        pes, pen = (_format(es, kw), out) if get_lang() == "en" else (out, _format(en, kw))
        if pes != pen and min(len(pes), len(pen)) >= _PAIRS_MIN_LEN:
            _PAIRS[pes] = pen
    return out


def collected_pairs() -> list[list[str]]:
    """Pares [español, inglés] dichos hasta ahora en este proceso (ver ``tr``)."""
    return [[es, en] for es, en in _PAIRS.items()]


@contextmanager
def use_lang(lang: str | None) -> Iterator[None]:
    """Fija el idioma dentro de un bloque (si ``lang`` es válido) y lo restaura al salir."""
    token = _LANG.set(lang if lang in SUPPORTED else _LANG.get())
    try:
        yield
    finally:
        _LANG.reset(token)
