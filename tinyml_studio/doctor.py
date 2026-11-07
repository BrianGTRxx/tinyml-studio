"""Doctor: ¿qué le falta o qué no es compatible a un entorno Python para TinyML Studio?

Sirve para el modo "usa tu propio entorno" (venv, conda, miniconda): se inspecciona cualquier
intérprete sin importar nada pesado (una sonda de la biblioteca estándar lee versiones con
``importlib.metadata``) y se evalúa contra una tabla de compatibilidad. La tabla es NUESTRA
(rangos que declaramos y probamos en CI con Python 3.10-3.13), no una garantía del ecosistema.

Seguridad: la ruta del intérprete la elige el usuario (diálogo nativo o argumento de la CLI); la
API HTTP del sidecar NO acepta rutas de intérprete (sería ejecución de código arbitrario).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any

try:
    from tinyml_studio.i18n import tr
except (
    ImportError
):  # ejecución autónoma (packaging/env/doctor_standalone.py): sin tinyml_studio instalado
    _LANG = {"v": "es"}

    def tr(es: str, en: str, **kw: Any) -> str:
        return (
            (en if _LANG["v"] == "en" else es).format(**kw)
            if kw
            else (en if _LANG["v"] == "en" else es)
        )


MIN_PYTHON = (3, 10)

# componente -> paquetes (nombre de distribución, mínimo, máximo EXCLUSIVO o None)
COMPONENTS: dict[str, dict[str, Any]] = {
    "base": {
        "label": (
            "Núcleo: datos, limpieza y modelos clásicos",
            "Core: data, cleaning, classical models",
        ),
        "packages": [
            ("numpy", "1.24", "3"),
            ("pandas", "2.0", "4"),
            ("scipy", "1.10", "2"),
            ("scikit-learn", "1.3", "2"),
            ("pydantic", "2.0", "3"),
            ("fastapi", "0.100", None),
            ("uvicorn", "0.20", None),
            ("psutil", "5.9", None),
            ("PyYAML", "6.0", None),
        ],
    },
    "tui": {
        "label": (
            "Interfaz de terminal, gráficas e informe PDF",
            "Terminal interface, charts and PDF report",
        ),
        "packages": [("textual", "0.80", None), ("pillow", "10.1", None), ("fpdf2", "2.7", None)],
    },
    "keras": {
        "label": (
            "Redes neuronales (Keras/TensorFlow, TFLite)",
            "Neural networks (Keras/TensorFlow, TFLite)",
        ),
        "packages": [("tensorflow", "2.15", None), ("keras", "3.0", None)],
        "note": (
            "TensorFlow con GPU solo en Linux/WSL2; en Windows nativo corre en CPU.",
            "TensorFlow with GPU only on Linux/WSL2; on native Windows it runs on CPU.",
        ),
    },
    "torch": {
        "label": ("PyTorch (GPU NVIDIA CUDA o AMD ROCm)", "PyTorch (NVIDIA CUDA or AMD ROCm GPU)"),
        "packages": [("torch", "2.2", None)],
    },
    "tflm": {
        "label": (
            "TFLite Micro (verificación real del modelo)",
            "TFLite Micro (real model verification)",
        ),
        "packages": [("tflite-micro", "0.dev0", None)],
        "platforms": ("linux",),  # no hay rueda de tflite-micro para Windows ni macOS
        "unavailable": (
            "No existe para este sistema (solo Linux x86-64): se omite.",
            "Not available on this system (Linux x86-64 only): skipped.",
        ),
    },
}
DEFAULT_COMPONENTS = ("base",)

# Sonda: SOLO biblioteca estándar, se ejecuta con ``python -I -c``; imprime una línea JSON.
PROBE = r"""
import json, os, platform, sys
from importlib import metadata
names = json.loads(sys.argv[1])
def ver(n):
    try:
        return metadata.version(n)
    except Exception:
        return None
conda = os.path.isdir(os.path.join(sys.prefix, "conda-meta"))
def torch_build():
    # metadata.version("torch") pierde el sufijo (+cpu, +cu130): se lee version.py sin importar
    try:
        from importlib import util
        spec = util.find_spec("torch")
        folder = os.path.dirname(spec.origin)
        text = open(os.path.join(folder, "version.py"), encoding="utf-8").read()
        for line in text.splitlines():
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("'\"")
    except Exception:
        pass
    return None
import shutil
print(json.dumps({
    "torch_build": torch_build(),
    "nvidia_smi": bool(shutil.which("nvidia-smi")),
    "executable": sys.executable,
    "version": list(sys.version_info[:3]),
    "implementation": platform.python_implementation(),
    "platform": sys.platform,
    "machine": platform.machine(),
    "prefix": sys.prefix,
    "is_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
    "is_conda": conda,
    "packages": {n: ver(n) for n in names},
    "tinyml_studio": ver("tinyml-studio"),
}))
"""


def _watch_list() -> list[str]:
    return sorted({p[0] for c in COMPONENTS.values() for p in c["packages"]})


def parse_version(v: str) -> tuple[int, ...]:
    """'2.1.0rc1' -> (2, 1, 0); tolera sufijos (+cu121, .dev0, rc)."""
    out = []
    for part in re.split(r"[.+-]", v):
        m = re.match(r"\d+", part)
        if not m:
            break
        out.append(int(m.group()))
    return tuple(out) or (0,)


def _cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    a, b = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def probe_interpreter(python: str | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """Inspecciona un intérprete. ``None``: el que ejecuta este código (sin subproceso)."""
    names = _watch_list()
    if python is None:
        if getattr(sys, "frozen", False):  # el ejecutable congelado no es un intérprete
            return {
                "executable": sys.executable,
                "version": list(sys.version_info[:3]),
                "implementation": "frozen",
                "platform": sys.platform,
                "machine": "",
                "prefix": sys.prefix,
                "is_venv": False,
                "is_conda": False,
                "frozen": True,
                "packages": _frozen_packages(names),
                "tinyml_studio": _frozen_packages(["tinyml-studio"]).get("tinyml-studio"),
            }
        r = subprocess.run(
            [sys.executable, "-I", "-c", PROBE, json.dumps(names)],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    else:
        if not os.path.isfile(python):
            raise FileNotFoundError(
                tr("no existe el intérprete: {p}", "interpreter not found: {p}", p=python)
            )
        try:
            r = subprocess.run(
                [python, "-I", "-c", PROBE, json.dumps(names)],
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                tr("el intérprete no respondió", "the interpreter did not respond")
            ) from e
        except OSError as e:
            raise RuntimeError(f"{python}: {e}") from e
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(
            tr(
                "no es un Python utilizable: {e}",
                "not a usable Python: {e}",
                e=(r.stderr or "")[-200:],
            )
        )
    return json.loads(lines[-1])


def _frozen_packages(names: list[str]) -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:  # noqa: BLE001 - paquete ausente o metadatos no empaquetados
            out[n] = None
    return out


def evaluate(probe: dict[str, Any], components: list[str] | None = None) -> dict[str, Any]:
    """Compara la sonda con la tabla de compatibilidad y propone los comandos de arreglo."""
    wanted = list(components or DEFAULT_COMPONENTS)
    unknown = [c for c in wanted if c not in COMPONENTS]
    if unknown:
        raise ValueError(tr("componente desconocido: {c}", "unknown component: {c}", c=unknown[0]))
    pyv = tuple(probe["version"])
    py_ok = pyv >= MIN_PYTHON
    pkgs = probe.get("packages", {})
    comp_out: dict[str, Any] = {}
    fixes: list[dict[str, Any]] = []
    for cid in wanted:
        spec = COMPONENTS[cid]
        if "platforms" in spec and not str(probe.get("platform", "")).startswith(spec["platforms"]):
            comp_out[cid] = {
                "label": tr(*spec["label"]),
                "ok": True,
                "available": False,
                "packages": [],
                "note": tr(*spec["unavailable"]),
            }
            continue
        rows = []
        for name, lo, hi in spec["packages"]:
            have = pkgs.get(name)
            if have is None:
                status = "missing"
            else:
                hv = parse_version(have)
                if _cmp(hv, parse_version(lo)) < 0:
                    status = "too_old"
                elif hi is not None and _cmp(hv, parse_version(hi)) >= 0:
                    status = "too_new"
                else:
                    status = "ok"
            req = f">={lo}" + (f",<{hi}" if hi else "")
            rows.append({"package": name, "have": have, "required": req, "status": status})
            if status != "ok":
                fixes.append(
                    {
                        "component": cid,
                        "package": name,
                        "requirement": f"{name}{req}",
                        "status": status,
                    }
                )
        comp_out[cid] = {
            "label": tr(*spec["label"]),
            "ok": all(r["status"] == "ok" for r in rows),
            "packages": rows,
            "note": tr(*spec["note"]) if "note" in spec else "",
        }
    kind = (
        "frozen"
        if probe.get("frozen")
        else "conda"
        if probe.get("is_conda")
        else "venv"
        if probe.get("is_venv")
        else "system"
    )
    py_cmd = probe["executable"]
    commands = []
    if fixes and not probe.get("frozen"):
        reqs = [f["requirement"] for f in fixes]
        commands.append([py_cmd, "-m", "pip", "install", *reqs])
    warnings = []
    if not py_ok:
        warnings.append(
            tr(
                "Python {v} es demasiado antiguo: Hace falta {m}+.",
                "Python {v} is too old: {m}+ is required.",
                v=".".join(map(str, pyv)),
                m=".".join(map(str, MIN_PYTHON)),
            )
        )
    build = str(probe.get("torch_build") or "")
    if "torch" in comp_out and probe.get("nvidia_smi"):
        from tinyml_studio.hw.advice import nvidia_gpu_info, torch_gpu_advice

        # nvidia-smi existe pero no se puede consultar (p. ej. en las pruebas): GPU NVIDIA genérica
        gpus = nvidia_gpu_info() or [{"name": "NVIDIA GPU", "driver": "", "cc": None}]
        warnings.extend(torch_gpu_advice(gpus, build, os.name))
    if kind == "conda" and fixes:
        warnings.append(
            tr(
                "Entorno conda: Mezclar pip con conda puede romperlo; "
                "si puedes, instala con conda.",
                "Conda environment: Mixing pip with conda can break it; prefer conda when you can.",
            )
        )
    if kind == "system" and fixes:
        warnings.append(
            tr(
                "Es el Python del sistema: Mejor crea un entorno virtual antes de instalar nada.",
                "This is the system Python: Create a virtual environment "
                "before installing anything.",
            )
        )
    if kind == "frozen" and fixes:
        warnings.append(
            tr(
                "Núcleo incluido en la app: No se modifica; "
                "usa tu propio entorno para añadir paquetes.",
                "Built-in core: It is not modified; use your own environment to add packages.",
            )
        )
    if probe.get("tinyml_studio") is None and not probe.get("frozen"):
        warnings.append(
            tr(
                "tinyml-studio no está instalado en este entorno (aún no está en PyPI): "
                "instala la rueda .whl que trae la app, o `pip install -e <clon del repo>`, "
                "siempre con --no-deps.",
                "tinyml-studio is not installed in this environment (it is not on PyPI yet): "
                "install the .whl that ships with the app, or `pip install -e <repo clone>`, "
                "always with --no-deps.",
            )
        )
    ok = py_ok and all(c["ok"] for c in comp_out.values())
    return {
        "ok": ok,
        "python": {
            "executable": py_cmd,
            "version": ".".join(map(str, pyv)),
            "ok": py_ok,
            "kind": kind,
            "platform": probe.get("platform"),
            "machine": probe.get("machine"),
        },
        "tinyml_studio": probe.get("tinyml_studio"),
        "components": comp_out,
        "fixes": fixes,
        "commands": commands,
        "warnings": warnings,
    }


def doctor(python: str | None = None, components: list[str] | None = None) -> dict[str, Any]:
    return evaluate(probe_interpreter(python), components)


def format_report(rep: dict[str, Any]) -> str:
    sym = {
        "ok": "OK",
        "missing": tr("FALTA", "MISSING"),
        "too_old": tr("ANTIGUO", "TOO OLD"),
        "too_new": tr("MUY NUEVO", "TOO NEW"),
    }
    py = rep["python"]
    L = [
        tr(
            "Python {v} ({k}) en {e}",
            "Python {v} ({k}) at {e}",
            v=py["version"],
            k=py["kind"],
            e=py["executable"],
        )
    ]
    for cid, c in rep["components"].items():
        state = "OK" if c["ok"] else tr("requiere arreglo", "needs fixing")
        L.append(f"\n[{cid}] {c['label']} — {state}")
        for r in c["packages"]:
            L.append(
                f"  {sym[r['status']]:9} {r['package']:14} {r['have'] or '—':12} ({r['required']})"
            )
        if c["note"]:
            L.append(f"  · {c['note']}")
    for w in rep["warnings"]:
        L.append(f"! {w}")
    if rep["commands"]:
        L.append(tr("\nPara arreglarlo:", "\nTo fix it:"))
        for cmd in rep["commands"]:
            L.append(
                "  " + " ".join(f'"{a}"' if " " in a or "<" in a or ">" in a else a for a in cmd)
            )
    elif rep["ok"]:
        L.append(tr("\nTodo en orden.", "\nAll good."))
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    """Uso autónomo: ``python doctor.py --json --components base,keras [--lang en]``.

    Con la rueda instalada se usa ``tinyml-studio doctor``; esta entrada sirve para revisar un
    Python (el propio o ``--python``) cuando ``tinyml_studio`` aún no está instalado en él.
    """
    import argparse

    # `python -I` ignora PYTHONUTF8: en Windows la salida saldría en cp1252 y quien la lee
    # (Electron) la decodifica como UTF-8 y se rompe la "á". Esta herramienta siempre emite UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(prog="doctor")
    ap.add_argument("--python")
    ap.add_argument("--components", default="base")
    ap.add_argument("--lang", choices=["es", "en"], default="es")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if "_LANG" in globals():
        globals()["_LANG"]["v"] = a.lang
    else:  # con tinyml_studio instalado, tr() usa el idioma de su módulo i18n
        from tinyml_studio.i18n import set_lang

        set_lang(a.lang)
    try:
        rep = doctor(a.python, [c for c in a.components.split(",") if c])
    except (OSError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(rep, ensure_ascii=False, indent=2) if a.json else format_report(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
