"""Detección de hardware de entrenamiento (CPU, RAM, GPU NVIDIA/AMD).

Solo informa y sugiere: el entrenamiento real usa lo que TensorFlow/scikit-learn encuentren.
Solo se probó la ruta NVIDIA con un ``nvidia-smi`` simulado y la ruta sin GPU (nube); las
rutas AMD y Windows son best-effort y no se han probado en hardware real.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tinyml_studio.i18n import tr
from tinyml_studio.pyexec import python_argv


def _run(cmd: list[str], timeout: int = 8) -> str | None:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _ram_gb() -> float | None:
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (OSError, ValueError):
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [
                    ("l", ctypes.c_ulong),
                    ("p", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("a", ctypes.c_ulonglong),
                    ("b", ctypes.c_ulonglong),
                    ("c", ctypes.c_ulonglong),
                    ("d", ctypes.c_ulonglong),
                    ("e", ctypes.c_ulonglong),
                    ("f", ctypes.c_ulonglong),
                ]

            m = MEM()
            m.l = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))  # type: ignore[attr-defined]
            return round(m.total / 2**30, 1)
        except Exception:  # noqa: BLE001
            return None
    return None


def _cpu_name() -> str:
    if platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Windows":  # platform.processor() solo da "AMD64 Family 25 Model 68..."
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as k:
                name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
            if name:
                return " ".join(name.split())
        except (OSError, ImportError):
            pass
    return platform.processor() or platform.machine()


def _nvidia() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                mem = int(float(parts[1]))
            except ValueError:
                mem = None
            gpus.append(
                {"vendor": "NVIDIA", "name": parts[0], "memory_mb": mem, "driver": parts[2]}
            )
    return gpus


# ----------------------------------------------------------------------------- AMD (parsers)

AMD_VENDOR_ID = "0x1002"


def _mb(value: Any) -> int | None:
    """Bytes (o texto con bytes) -> MiB; None si no es un número válido."""
    try:
        b = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return b // 2**20 if b > 0 else None


def _amd_entry(name: str, vram_mb: int | None, rocm: bool, source: str) -> dict[str, Any]:
    return {
        "vendor": "AMD",
        "name": name,
        "memory_mb": vram_mb,  # alias antiguo de vram_mb
        "vram_mb": vram_mb,
        "driver": "ROCm" if rocm else None,
        "rocm_present": rocm,
        "source": source,
    }


def parse_rocm_smi_json(text: str, rocm: bool = True) -> list[dict[str, Any]]:
    """Salida de ``rocm-smi --showproductname --showmeminfo vram --json`` (claves 'card0'…)."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    out = []
    for key, card in data.items() if isinstance(data, dict) else []:
        if not (key.lower().startswith("card") and isinstance(card, dict)):
            continue
        low = {str(k).lower(): v for k, v in card.items()}
        name = next(
            (
                str(low[k]).strip()
                for k in ("card series", "card model", "device name")
                if low.get(k)
            ),
            "",
        )
        vram = next((_mb(v) for k, v in low.items() if k.startswith("vram total memory")), None)
        out.append(_amd_entry(name or "AMD GPU", vram, rocm, "rocm-smi"))
    return out


def parse_rocminfo(text: str, rocm: bool = True) -> list[dict[str, Any]]:
    """Agentes de tipo GPU de ``rocminfo`` (nombre comercial; la VRAM no se deduce)."""
    out = []
    for block in re.split(r"\n(?=Agent \d+)", text or ""):
        if not re.search(r"Device Type:\s*GPU", block):
            continue
        m = re.search(r"Marketing Name:\s*(.+)", block) or re.search(r"Name:\s*(gfx\w+)", block)
        out.append(_amd_entry(m.group(1).strip() if m else "AMD GPU", None, rocm, "rocminfo"))
    return out


def parse_lspci(text: str, rocm: bool = False) -> list[dict[str, Any]]:
    """Líneas de ``lspci -nn`` de tarjetas gráficas con vendor 1002 (AMD/ATI)."""
    out = []
    for ln in (text or "").splitlines():
        if "[1002:" not in ln or not re.search(r"VGA|Display controller|3D controller", ln):
            continue
        m = re.search(r"\]:\s*(.+?)\s*\[1002:", ln)
        name = re.sub(r"^Advanced Micro Devices, Inc\.\s*", "", m.group(1)) if m else "AMD GPU"
        out.append(_amd_entry(name.strip(), None, rocm, "lspci"))
    return out


def parse_sysfs_amd(root: str = "/sys/class/drm", rocm: bool = False) -> list[dict[str, Any]]:
    """Tarjetas DRM con ``device/vendor == 0x1002`` (nombre genérico; VRAM si amdgpu la expone)."""
    out = []
    try:
        cards = sorted(p for p in os.listdir(root) if re.fullmatch(r"card\d+", p))
    except OSError:
        return []
    for c in cards:
        dev = os.path.join(root, c, "device")
        try:
            with open(os.path.join(dev, "vendor"), encoding="utf-8") as fh:
                if fh.read().strip().lower() != AMD_VENDOR_ID:
                    continue
        except OSError:
            continue
        vram = None
        try:
            with open(os.path.join(dev, "mem_info_vram_total"), encoding="utf-8") as fh:
                vram = _mb(fh.read())
        except OSError:
            pass
        out.append(_amd_entry("AMD GPU (sysfs)", vram, rocm, "sysfs"))
    return out


_ADAPTER_RAM_CAP = 4293918720  # Win32_VideoController.AdapterRAM es de 32 bits: topa en ~4 GB


def parse_windows_video(text: str, rocm: bool = False) -> list[dict[str, Any]]:
    """``Name|AdapterRAM`` (PowerShell) o CSV de ``wmic ... /format:csv`` (Node,AdapterRAM,Name)."""
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln or ln.lower().startswith(("node,", "name|", "name,")):
            continue
        if "|" in ln:
            name, _, ram = ln.partition("|")
        else:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 3:
                name, ram = ln, ""
            else:
                ram, name = parts[1], ",".join(parts[2:])
        name = name.strip()
        if not re.search(r"\b(AMD|Radeon|ATI)\b", name, re.I):
            continue
        try:
            raw = int(float(ram))
        except ValueError:
            raw = 0
        vram = None if raw >= _ADAPTER_RAM_CAP else _mb(raw)  # tope de 32 bits: no es fiable
        out.append(_amd_entry(name, vram, rocm, "windows"))
    return out


def _rocm_present() -> bool:
    return bool(shutil.which("rocm-smi") or shutil.which("rocminfo") or os.path.isdir("/opt/rocm"))


def _amd() -> list[dict[str, Any]]:
    rocm = _rocm_present()
    system = platform.system()
    if shutil.which("rocm-smi"):
        gpus = parse_rocm_smi_json(
            _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"]) or "", rocm
        )
        if gpus:
            return gpus
    if shutil.which("rocminfo"):
        gpus = parse_rocminfo(_run(["rocminfo"]) or "", rocm)
        if gpus:
            return gpus
    if system == "Windows":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "ForEach-Object { $_.Name + '|' + $_.AdapterRAM }",
            ]
        )
        if not out:
            out = _run(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM", "/format:csv"]
            )
        return parse_windows_video(out or "", rocm)
    if system == "Linux":
        gpus = parse_lspci(_run(["lspci", "-nn"]) or "", rocm) if shutil.which("lspci") else []
        sysfs = parse_sysfs_amd(rocm=rocm)
        if not gpus:
            return sysfs
        for g, sf in zip(gpus, sysfs, strict=False):  # lspci da el nombre; sysfs, la VRAM
            if len(gpus) == len(sysfs):
                g["vram_mb"] = g["memory_mb"] = sf["vram_mb"]
        return gpus
    return []


# ----------------------------------------------------------------------------- frameworks

_TORCH_PROBE = """
import json
try:
    import torch
except Exception as e:
    print(json.dumps({"importable": False, "error": str(e)[:200]}))
else:
    ok = bool(torch.cuda.is_available())
    print(json.dumps({
        "importable": True, "version": torch.__version__,
        "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if ok else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if ok else [],
    }))
"""
_TF_PROBE = """
import json
try:
    import tensorflow as tf
except Exception as e:
    print(json.dumps({"importable": False, "error": str(e)[:200]}))
else:
    try:
        b = tf.sysconfig.get_build_info()
    except Exception:
        b = {}
    g = tf.config.list_physical_devices("GPU")
    print(json.dumps({
        "importable": True, "version": tf.__version__, "gpu_count": len(g),
        "devices": [d.name for d in g], "built_with_cuda": bool(b.get("is_cuda_build")),
        "built_with_rocm": bool(b.get("is_rocm_build")),
    }))
"""
PROBE_TIMEOUT = 90
_probe_cache: dict[str, dict[str, Any]] = {}


def _probe_script(script: str) -> dict[str, Any]:
    """Ejecuta el sondeo en un subproceso aislado: nunca cuelga ni carga TF/torch aquí."""
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    try:
        r = subprocess.run(
            python_argv(code=script, isolated=True),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,  # sin heredar el stdin del sidecar (en Windows se cuelga)
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"importable": None, "gpu_count": 0, "error": f"timeout {PROBE_TIMEOUT}s"}
    except (OSError, subprocess.SubprocessError) as e:
        return {"importable": None, "gpu_count": 0, "error": str(e)[:200]}
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
    try:
        info = json.loads(lines[-1])
    except (IndexError, ValueError):
        return {"importable": None, "gpu_count": 0, "error": (r.stderr or "")[-200:]}
    info.setdefault("gpu_count", 0)
    return info


def probe_framework(name: str, refresh: bool = False) -> dict[str, Any]:
    """¿Ve una GPU el framework 'torch' o 'tensorflow' de ESTE entorno Python? (con caché)."""
    if name not in ("torch", "tensorflow"):
        raise ValueError(name)
    if refresh or name not in _probe_cache:
        if importlib.util.find_spec(name) is None:  # p. ej. compilación lite: sin subproceso
            _probe_cache[name] = {"importable": False, "gpu_count": 0, "error": "not installed"}
            return _probe_cache[name]
        _probe_cache[name] = _probe_script(_TORCH_PROBE if name == "torch" else _TF_PROBE)
    return _probe_cache[name]


def probe_frameworks(refresh: bool = False) -> dict[str, dict[str, Any]]:
    names = ("tensorflow", "torch")
    with ThreadPoolExecutor(
        max_workers=len(names)
    ) as pool:  # cada sondeo es un subproceso: en paralelo
        return dict(zip(names, pool.map(lambda n: probe_framework(n, refresh), names), strict=True))


def gpu_guidance() -> list[dict[str, str]]:
    """Qué instalar para usar la GPU. Sin versiones: cambian; ver docs/GPU_AMD.md y las URLs."""
    return [
        {
            "id": "nvidia_cuda",
            "text": tr(
                "NVIDIA: Instala el driver y una build de PyTorch o TensorFlow con CUDA. "
                "TensorFlow con GPU solo en Linux o WSL2 (extra '[and-cuda]').",
                "NVIDIA: Install the driver and a CUDA build of PyTorch or TensorFlow. "
                "TensorFlow with GPU only on Linux or WSL2 ('[and-cuda]' extra).",
            ),
            "url": "https://www.tensorflow.org/install/pip",
        },
        {
            "id": "amd_rocm_linux",
            "text": tr(
                "AMD en Linux: Instala ROCm y PyTorch para ROCm (backend 'torch' de la app); "
                "TensorFlow-ROCm existe solo en Linux. Comprueba tu GPU en la matriz de AMD.",
                "AMD on Linux: Install ROCm and PyTorch for ROCm (the app's 'torch' backend); "
                "TensorFlow-ROCm exists on Linux only. Check your GPU in AMD's support matrix.",
            ),
            "url": "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html",
        },
        {
            "id": "amd_windows",
            "text": tr(
                "AMD en Windows: No hay TensorFlow con GPU. PyTorch con ROCm en Windows está "
                "limitado a algunas Radeon/Ryzen según AMD; si no es tu caso usa Linux o WSL2.",
                "AMD on Windows: There is no TensorFlow GPU build. PyTorch with ROCm on Windows "
                "is limited to some Radeon/Ryzen parts per AMD; otherwise use Linux or WSL2.",
            ),
            "url": "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html",
        },
        {
            "id": "wsl2",
            "text": tr(
                "WSL2: Es la vía para GPU NVIDIA con TensorFlow desde Windows.",
                "WSL2: The route to an NVIDIA GPU with TensorFlow from Windows.",
            ),
            "url": "https://www.tensorflow.org/install/pip",
        },
        {
            "id": "windows_tf",
            "text": tr(
                "TensorFlow en Windows nativo no usa GPU desde la 2.11 (la 2.10 fue la última).",
                "TensorFlow on native Windows has no GPU support after 2.10 (2.10 was the last).",
            ),
            "url": "https://www.tensorflow.org/install/pip",
        },
    ]


def _device_plan(gpus: list[dict[str, Any]], system: str, rocm: bool) -> tuple[str, str]:
    """(dispositivo sugerido, texto honesto de qué se usará y qué está verificado)."""
    nvidia = next((g for g in gpus if g["vendor"] == "NVIDIA"), None)
    amd = next((g for g in gpus if g["vendor"] == "AMD"), None)
    if nvidia:
        win = (
            tr(
                "; TensorFlow no la usa en Windows nativo",
                "; TensorFlow does not use it on native Windows",
            )
            if system == "Windows"
            else ""
        )
        return "gpu", tr(
            "Hay una GPU NVIDIA ({n}). Con el backend PyTorch (CUDA) y «usar la GPU» se "
            "entrena en ella (verificado en una RTX 4050 con Windows 11){w}. En redes pequeñas "
            "la CPU suele ser igual de rápida o más: La GPU compensa con redes y lotes grandes.",
            "An NVIDIA GPU ({n}) is present. With the PyTorch (CUDA) backend and 'use the GPU' "
            "training runs on it (verified on an RTX 4050 with Windows 11){w}. On small "
            "networks the CPU is often as fast or faster: The GPU pays off with large "
            "networks and batches.",
            n=nvidia["name"],
            w=win,
        )
    if amd and system == "Linux" and rocm:
        return "gpu", tr(
            "Hay una GPU AMD ({n}) y ROCm: Con el backend PyTorch (ROCm) y «usar la GPU» se "
            "intentará entrenar en ella. Sin probar en hardware real.",
            "An AMD GPU ({n}) and ROCm are present: With the PyTorch (ROCm) backend and 'use "
            "the GPU' training will try to run on it. Not tested on real hardware.",
            n=amd["name"],
        )
    if amd:
        return "cpu", tr(
            "Hay una gráfica AMD ({n}), pero en este sistema no hay una vía para entrenar en "
            "ella (TensorFlow no usa GPU AMD; PyTorch con ROCm funciona en Linux y en pocas "
            "Radeon con Windows). Se entrena en la CPU, lo normal con una gráfica integrada.",
            "An AMD graphics chip ({n}) is present, but this system has no way to train on it "
            "(TensorFlow does not use AMD GPUs; PyTorch with ROCm works on Linux and on a few "
            "Radeon cards on Windows). Training runs on the CPU, as usual with integrated "
            "graphics.",
            n=amd["name"],
        )
    return "cpu", tr(
        "No se detectó ninguna GPU: Se entrena en la CPU.",
        "No GPU was detected: Training runs on the CPU.",
    )


def detect_hardware(probe_frameworks_too: bool = False) -> dict[str, Any]:
    """``probe_frameworks_too=True`` añade ``frameworks`` (sondeo en subprocesos, tarda unos
    segundos); por defecto es None para que la consulta siga siendo instantánea."""
    gpus = _nvidia() + _amd()
    system = platform.system()
    device_plan = _device_plan(gpus, system, _rocm_present())
    notes: list[str] = []
    if system == "Windows":
        notes.append(
            tr(
                "TensorFlow >= 2.11 en Windows nativo solo usa CPU; para GPU NVIDIA usa WSL2.",
                "TensorFlow >= 2.11 on native Windows only uses the CPU; for an NVIDIA GPU "
                "use WSL2.",
            )
        )
    if any(g["vendor"] == "AMD" for g in gpus):
        notes.append(
            tr(
                "AMD: TensorFlow/Keras no usa GPU AMD de forma estándar; se entrenará en CPU.",
                "AMD: TensorFlow/Keras does not use AMD GPUs out of the box; training will run "
                "on the CPU.",
            )
        )
    notes.append(
        tr(
            "La mayoría de los modelos TinyML (miles de parámetros) entrenan en CPU en segundos "
            "o pocos minutos; la GPU importa para CNN de audio/imagen y búsquedas grandes.",
            "Most TinyML models (thousands of parameters) train on the CPU in seconds or a few "
            "minutes; the GPU matters for audio/image CNNs and large searches.",
        )
    )
    return {
        "os": f"{system} {platform.release()}",
        "python": platform.python_version(),
        "cpu": {"name": _cpu_name(), "logical_cores": os.cpu_count()},
        "ram_gb": _ram_gb(),
        "gpus": gpus,
        "suggested_device": device_plan[0],
        "device_note": device_plan[1],
        "notes": notes,
        "amd_gpus": [g for g in gpus if g["vendor"] == "AMD"],
        "rocm_present": _rocm_present(),
        "frameworks": probe_frameworks() if probe_frameworks_too else None,
        "guidance": gpu_guidance(),
    }
