"""Cómo relanzar "un Python" desde el núcleo, también cuando está congelado con PyInstaller."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def python_argv(
    *,
    module: str | None = None,
    code: str | None = None,
    args: Sequence[str] = (),
    isolated: bool = False,
) -> list[str]:
    """argv de un subproceso que ejecuta ``module`` (como ``-m``) o ``code`` (como ``-c``).

    Sin congelar: ``[sys.executable, "-m", module, *args]`` o
    ``[sys.executable, ("-I",) "-c", code, *args]``. Congelado, ``sys.executable`` es el propio
    ejecutable (no un intérprete): ``[sys.executable, "run-module"|"run-code", ...]``
    (ver ``tinyml_studio.core_main``; ``isolated`` no aplica).
    """
    if (module is None) == (code is None):
        raise ValueError("exactly one of module / code is required")
    if is_frozen():
        if module is not None:
            return [sys.executable, "run-module", module, *args]
        return [sys.executable, "run-code", str(code), *args]
    if module is not None:
        return [sys.executable, "-m", module, *args]
    return [sys.executable, *(["-I"] if isolated else []), "-c", str(code), *args]
