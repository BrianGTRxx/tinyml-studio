# ruff: noqa: E501
"""Qué versión de PyTorch conviene instalar según la GPU (consejo; la app nunca instala nada sola).

Las ruedas de PyTorch cambian con cada versión: la lista actual está siempre en
https://pytorch.org/get-started/locally/ y aquí solo se da la regla estable:

* Blackwell (RTX 50, RTX PRO Blackwell, B100/B200; capacidad de cómputo 10.x/12.x): CUDA 12.8 o superior
  (``cu128``/``cu130``) y controlador 570 o superior. Una rueda ``cu126`` o menor NO funciona en ellas.
* Hopper, Ada, Ampere, Turing y Volta (capacidad 7.0 a 9.0: H100, RTX 40/30/20, GTX 16, V100): ``cu126``
  (también sirven ``cu128``/``cu130``).
* Pascal (capacidad 6.x: GTX 10, Titan X, P100): solo las ruedas ``cu126``; las más nuevas la dejaron fuera.
* Maxwell y Kepler (capacidad menor que 6): las ruedas actuales ya no las incluyen; la app avisa de que
  no hay rueda oficial reciente.
* Se usa la capacidad de cómputo que informa nvidia-smi; si no la da, se deduce del nombre.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

from tinyml_studio.i18n import tr

PYTORCH_URL = "https://pytorch.org/get-started/locally/"
INDEX = "https://download.pytorch.org/whl/{tag}"


def family_from_cc(cc: float | None) -> str | None:
    """Generación según la capacidad de cómputo de NVIDIA (la fuente más fiable)."""
    if cc is None:
        return None
    if cc >= 10.0:
        return "blackwell"
    if cc >= 9.0:
        return "hopper"
    if cc >= 8.9:
        return "ada"
    if cc >= 8.0:
        return "ampere"
    if cc >= 7.5:
        return "turing"
    if cc >= 7.0:
        return "volta"
    if cc >= 6.0:
        return "pascal"
    return "legacy"


def gpu_family(name: str, cc: float | None = None) -> str:
    """Generación de la GPU: por capacidad de cómputo si se conoce; si no, por el nombre comercial."""
    by_cc = family_from_cc(cc)
    if by_cc:
        return by_cc
    n = name.lower()
    if re.search(r"rtx\s*(pro\s*)?\d*\s*blackwell|rtx\s*50\d{2}|\b[bg]b?[12]\d0\b|b200|b100|gb10", n):
        return "blackwell"
    if re.search(r"h100|h200|\bgh200\b", n):
        return "hopper"
    if re.search(r"rtx\s*40\d{2}|\bl40\b|\bl4\b|rtx\s*[2-6]000\s*ada", n):
        return "ada"
    if re.search(r"rtx\s*30\d{2}|\ba[1-9]\d{0,2}\b|rtx\s*a\d{3,4}", n):
        return "ampere"
    if re.search(r"rtx\s*20\d{2}|gtx\s*16\d{2}|\bt4\b|titan rtx", n):
        return "turing"
    if re.search(r"v100|titan v", n):
        return "volta"
    if re.search(r"gtx\s*10\d{2}|titan x|\bp\d{2,3}\b|quadro p", n):
        return "pascal"
    if re.search(r"gtx\s*(9\d{2}|7\d{2}|6\d{2})|\bk\d{2,3}\b|\bm\d{2,3}\b", n):
        return "legacy"
    return "other"


def wheel_tag(family: str) -> str:
    return "cu128" if family == "blackwell" else "cu126"


def driver_min(family: str, os_name: str) -> float | None:
    """Controlador mínimo para la rueda recomendada (CUDA 12.8: 570.x; CUDA 12.6: 560.x en Windows, 525 en Linux)."""
    if family == "blackwell":
        return 570.65 if os_name == "nt" else 570.26
    return 560.76 if os_name == "nt" else 525.60


def nvidia_gpu_info() -> list[dict[str, Any]]:
    """[{name, driver}] con nvidia-smi (lista vacía si no hay)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        r = subprocess.run(
            [exe, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:  # nvidia-smi antiguo sin compute_cap
            r = subprocess.run(
                [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if parts and parts[0]:
            cc = None
            if len(parts) > 2:
                try:
                    cc = float(parts[2])
                except ValueError:
                    cc = None
            out.append({"name": parts[0], "driver": parts[1] if len(parts) > 1 else "", "cc": cc})
    return out


def _driver_num(v: str) -> float | None:
    m = re.match(r"(\d+)\.(\d+)", v or "")
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def torch_gpu_advice(gpus: list[dict[str, Any]], torch_build: str, os_name: str) -> list[str]:
    """Mensajes (ya traducidos) para el doctor. Vacío si no hay nada que avisar."""
    out: list[str] = []
    cpu_only = torch_build.endswith("+cpu") or not torch_build
    for g in gpus:
        fam = gpu_family(g["name"], g.get("cc"))
        tag = wheel_tag(fam)
        if fam == "legacy":
            out.append(
                tr(
                    "Tu GPU ({n}) es anterior a Pascal: Las ruedas actuales de PyTorch ya no la incluyen. "
                    "El entrenamiento seguirá en CPU (o usa una versión antigua de torch bajo tu responsabilidad).",
                    "Your GPU ({n}) is older than Pascal: Current PyTorch wheels no longer include it. "
                    "Training will stay on the CPU (or use an old torch version at your own risk).",
                    n=g["name"],
                )
            )
            continue
        cmd = f"pip install --upgrade torch --index-url {INDEX.format(tag=tag)}"
        cur = re.search(r"\+cu(\d+)", torch_build or "")
        cur_num = int(cur.group(1)) if cur else 0
        if fam == "blackwell" and cur_num and cur_num < 128:
            out.append(
                tr(
                    "Tu GPU ({n}) es de la generación Blackwell y tu PyTorch ({b}) usa CUDA anterior a 12.8: No funcionará. "
                    "Instala una rueda cu128 o superior (cambia tu versión de torch, unos 2 GB; la app no lo hace sola): {c}",
                    "Your GPU ({n}) is Blackwell generation and your PyTorch ({b}) uses CUDA older than 12.8: It will not work. "
                    "Install a cu128 or newer wheel (changes your torch version, about 2 GB; the app never does this on its own): {c}",
                    n=g["name"],
                    b=torch_build,
                    c=cmd,
                )
            )
        elif cpu_only:
            out.append(
                tr(
                    "Tu PyTorch ({b}) es solo de CPU y hay una GPU NVIDIA ({n}): No se usará. Para usarla instala la build CUDA "
                    "{t} (cambia tu versión de torch, unos 2 GB; la app no lo hace sola): {c}. La lista vigente está en {u}",
                    "Your PyTorch ({b}) is CPU-only and an NVIDIA GPU ({n}) is present: It will not be used. To use it install the "
                    "{t} CUDA build (changes your torch version, about 2 GB; the app never does this on its own): {c}. The current list is at {u}",
                    b=torch_build or "?",
                    n=g["name"],
                    t=tag,
                    c=cmd,
                    u=PYTORCH_URL,
                )
            )
        need, have = driver_min(fam, os_name), _driver_num(g.get("driver", ""))
        if need and have and have < need and (cpu_only or fam == "blackwell"):
            out.append(
                tr(
                    "El controlador NVIDIA de {n} es {h} y la rueda {t} pide {m} o superior: Actualízalo desde nvidia.com.",
                    "The NVIDIA driver of {n} is {h} and the {t} wheel needs {m} or newer: Update it from nvidia.com.",
                    n=g["name"],
                    h=g["driver"],
                    t=tag,
                    m=f"{need:g}",
                )
            )
    return out
