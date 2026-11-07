"""Instalar componentes (Keras, PyTorch, TUI…) en el entorno Python actual con versiones fijadas.

Es el equivalente en Python del instalador de la app gráfica (``ui/src/installer.js``): mismo
manifiesto (``packaging/env/manifest.json``) y mismos locks con hashes. Usa ``uv`` si está
disponible y, si no, ``pip`` (que también verifica hashes). Solo instala desde PyPI con
``--require-hashes --only-binary :all:`` (no ejecuta ``setup.py`` de terceros) y valida los ids
contra el manifiesto: nunca nombres de paquete libres. La interfaz lo ejecuta SIEMPRE con
confirmación del usuario.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from tinyml_studio.i18n import tr
from tinyml_studio.resources import resource_root


def env_dir() -> Path:
    """Carpeta con manifest.json y locks/: ``_data/env`` (rueda) o ``packaging/env`` (repo)."""
    root = resource_root()
    packaged = root / "env"
    return packaged if (packaged / "manifest.json").is_file() else root / "packaging" / "env"


def platform_key(platform: str | None = None) -> str:
    p = platform or sys.platform
    return "windows" if p.startswith("win") else "macos" if p == "darwin" else "linux"


def load_manifest() -> dict[str, Any]:
    return json.loads((env_dir() / "manifest.json").read_text(encoding="utf-8"))


def available_components(platform: str | None = None) -> list[dict[str, Any]]:
    plat = platform_key(platform)
    return [c for c in load_manifest()["components"] if plat in c["platforms"]]


def resolve(ids: list[str], platform: str | None = None) -> list[str]:
    """ids válidos del manifiesto y de esta plataforma, con ``base`` primero."""
    plat = platform_key(platform)
    by_id = {c["id"]: c for c in load_manifest()["components"]}
    out = ["base"]
    for i in ids:
        if i not in by_id:
            raise ValueError(tr("componente desconocido: {c}", "unknown component: {c}", c=i[:40]))
        if plat not in by_id[i]["platforms"]:
            raise ValueError(
                tr(
                    "componente no disponible en {p}: {c}",
                    "component not available on {p}: {c}",
                    p=plat,
                    c=i,
                )
            )
        if i not in out:
            out.append(i)
    return out


def plan(
    ids: list[str],
    python: str | None = None,
    platform: str | None = None,
    uv: str | None = None,
) -> list[list[str]]:
    """Comandos (argv) que instalan los componentes en ``python``; no ejecuta nada."""
    py = python or sys.executable
    uv_bin = uv if uv is not None else shutil.which("uv")
    cmds = []
    for cid in resolve(ids, platform):
        lock = env_dir() / "locks" / f"{cid}-{platform_key(platform)}.txt"
        if not lock.is_file():
            raise FileNotFoundError(tr("falta el lock: {f}", "missing lock: {f}", f=lock.name))
        if uv_bin:
            cmds.append(
                [uv_bin, "pip", "install", "--python", py, "--require-hashes"]
                + ["--only-binary", ":all:", "-r", str(lock)]
            )
        else:
            cmds.append(
                [py, "-m", "pip", "install", "--require-hashes", "--only-binary", ":all:"]
                + ["-r", str(lock)]
            )
    return cmds


def iter_lines(stream) -> Iterator[str]:
    for raw in stream:
        line = raw.rstrip("\r\n")
        if line.strip():
            yield line


def run(
    ids: list[str],
    on_line: Callable[[str], None],
    python: str | None = None,
    platform: str | None = None,
    uv: str | None = None,
) -> None:
    """Ejecuta el plan mostrando cada línea; lanza RuntimeError si un paso falla."""
    if getattr(sys, "frozen", False):
        raise RuntimeError(
            tr(
                "El núcleo incluido en la app no se modifica: Usa tu entorno o el instalador.",
                "The core bundled in the app is not modified: Use your environment or installer.",
            )
        )
    env = {**os.environ, "PYTHONUTF8": "1", "NO_COLOR": "1", "UV_NO_PROGRESS": "1"}
    for cmd in plan(ids, python, platform, uv):
        on_line("$ " + " ".join(cmd[:4]) + " … " + Path(cmd[-1]).name)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
        )
        assert proc.stdout is not None
        tail: list[str] = []
        for line in iter_lines(proc.stdout):
            tail = (tail + [line])[-6:]
            on_line(line)
        if proc.wait() != 0:
            raise RuntimeError(
                tr(
                    "la instalación falló (código {c}): ",
                    "installation failed (code {c}): ",
                    c=proc.returncode,
                )
                + " | ".join(tail[-3:])
            )
    on_line(tr("Listo.", "Done."))
