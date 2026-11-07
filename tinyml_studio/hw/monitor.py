"""Monitor de hardware en vivo (CPU, RAM, GPU NVIDIA, temperatura) y resumen por trabajo.

Honestidad: lo que el sistema operativo no expone se devuelve como ``None`` y se explica en
``notes``; nunca se inventan lecturas. La GPU solo se lista si ``nvidia-smi`` responde.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from tinyml_studio.i18n import tr

MAX_LINES = 5000  # tope de muestras guardadas por trabajo (hw.jsonl)
KEEP_RECENT = 1000  # al recortar, las últimas N muestras se conservan íntegras
SMI_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
_MB = 1024 * 1024


# ------------------------------------------------------------------ utilidades puras
def _num(s: str) -> float | None:
    """'N/A', '[Not Supported]', '' -> None; si no, float."""
    s = s.strip()
    if not s or s.upper() in ("N/A", "[N/A]") or s.startswith("["):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Parsea la salida CSV (noheader,nounits) de nvidia-smi; tolera campos no disponibles."""
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        # el nombre puede contener comas: 1 índice + nombre (resto) + 5 numéricos
        idx, name, nums = parts[0], ", ".join(parts[1:-5]), parts[-5:]
        i = _num(idx)
        if i is None:
            continue
        util, mem_u, mem_t, temp, power = (_num(x) for x in nums)
        gpus.append(
            {
                "index": int(i),
                "name": name,
                "util_pct": util,
                "mem_used_mb": mem_u,
                "mem_total_mb": mem_t,
                "temp_c": temp,
                "power_w": power,
            }
        )
    return gpus


def cap_lines(lines: list[str], max_lines: int = MAX_LINES, keep_recent: int = KEEP_RECENT):
    """Si hay más de ``max_lines``, descarta una de cada dos líneas ANTIGUAS (las recientes
    se conservan). Devuelve la lista (misma si no excede)."""
    if len(lines) <= max_lines:
        return lines
    keep_recent = min(keep_recent, max_lines // 2)
    old, recent = lines[:-keep_recent], lines[-keep_recent:]
    return old[::2] + recent


def downsample_even(items: list, n: int) -> list:
    """Como máximo ``n`` elementos repartidos uniformemente (incluye primero y último)."""
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[-1]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


class SummaryAccumulator:
    """Media y picos exactos aunque el archivo de muestras se recorte."""

    def __init__(self) -> None:
        self.n = 0
        self.t0: float | None = None
        self.t1: float | None = None
        self._sum: dict[str, float] = {}
        self._cnt: dict[str, int] = {}
        self._max: dict[str, float] = {}
        self._names: dict[int, str | None] = {}

    def _add(self, key: str, v: float | None, mean: bool = True) -> None:
        if v is None:
            return
        if mean:
            self._sum[key] = self._sum.get(key, 0.0) + v
            self._cnt[key] = self._cnt.get(key, 0) + 1
        self._max[key] = max(self._max.get(key, v), v)

    def add(self, s: dict) -> None:
        self.n += 1
        self.t0 = s["t"] if self.t0 is None else self.t0
        self.t1 = s["t"]
        self._add("cpu", s.get("cpu_pct"))
        self._add("proc", s.get("proc_cpu_pct"))
        self._add("rss", s.get("proc_rss_mb"), mean=False)
        self._add("ram", s.get("ram_used_mb"), mean=False)
        self._add("cpu_temp", s.get("cpu_temp_c"), mean=False)
        for g in s.get("gpus") or []:
            k = f"g{g['index']}"
            self._add(k + "u", g.get("util_pct"))
            self._add(k + "m", g.get("mem_used_mb"), mean=False)
            self._add(k + "t", g.get("temp_c"), mean=False)
            self._add(k + "p", g.get("power_w"), mean=False)
            self._names[g["index"]] = g.get("name")

    def _mean(self, k: str) -> float | None:
        return round(self._sum[k] / self._cnt[k], 1) if self._cnt.get(k) else None

    def _peak(self, k: str) -> float | None:
        return round(self._max[k], 1) if k in self._max else None

    def summary(self) -> dict[str, Any]:
        gpus = []
        for i, name in sorted(self._names.items()):
            k = f"g{i}"
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "util_mean_pct": self._mean(k + "u"),
                    "util_peak_pct": self._peak(k + "u"),
                    "mem_peak_mb": self._peak(k + "m"),
                    "temp_peak_c": self._peak(k + "t"),
                    "power_peak_w": self._peak(k + "p"),
                }
            )
        dur = (self.t1 - self.t0) if self.t0 is not None and self.t1 is not None else 0.0
        return {
            "samples": self.n,
            "duration_s": round(dur, 1),
            "cpu_mean_pct": self._mean("cpu"),
            "cpu_peak_pct": self._peak("cpu"),
            "proc_cpu_mean_pct": self._mean("proc"),
            "proc_cpu_peak_pct": self._peak("proc"),
            "rss_peak_mb": self._peak("rss"),
            "ram_peak_mb": self._peak("ram"),
            "cpu_temp_peak_c": self._peak("cpu_temp"),
            "gpus": gpus,
        }


def summarize(samples: list[dict]) -> dict[str, Any]:
    acc = SummaryAccumulator()
    for s in samples:
        acc.add(s)
    return acc.summary()


# ------------------------------------------------------------------ muestreador
class HardwareSampler:
    """Toma una lectura del sistema; con ``pid`` también mide ese proceso y sus hijos."""

    def __init__(self, pid: int | None = None) -> None:
        self.pid = pid
        self.cores = psutil.cpu_count(logical=True) or 1
        self._procs: dict[int, psutil.Process] = {}
        self._smi = shutil.which("nvidia-smi")
        psutil.cpu_percent(interval=None)  # la primera llamada siempre devuelve 0.0

    # --- GPU
    def _gpus(self) -> list[dict[str, Any]]:
        if not self._smi:
            return []
        try:
            out = subprocess.run(
                [self._smi, f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=4,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        return parse_nvidia_smi(out.stdout) if out.returncode == 0 else []

    # --- temperatura (Linux: psutil; Windows: zona térmica del sistema)
    @staticmethod
    def _cpu_temp() -> float | None:
        from .cputemp import read_cpu_temp

        return read_cpu_temp()[0]

    @staticmethod
    def _cpu_temp_source() -> str | None:
        from .cputemp import read_cpu_temp

        return read_cpu_temp()[1]

    @staticmethod
    def _cpu_freq() -> float | None:
        try:
            f = psutil.cpu_freq()
        except Exception:  # noqa: BLE001
            return None
        return round(f.current, 0) if f and f.current else None

    # --- árbol de procesos del trabajo
    def _tree(self) -> list[psutil.Process]:
        if self.pid is None:
            return []
        try:
            root = self._procs.get(self.pid) or psutil.Process(self.pid)
            self._procs[self.pid] = root
            tree = [root, *root.children(recursive=True)]
        except psutil.Error:
            return []
        alive = {p.pid for p in tree}
        self._procs = {k: v for k, v in self._procs.items() if k in alive}
        return [self._procs.setdefault(p.pid, p) for p in tree]

    def _proc_stats(self) -> tuple[float | None, float | None]:
        if self.pid is None:
            return None, None
        cpu = rss = 0.0
        found = False
        for p in self._tree():
            try:
                cpu += p.cpu_percent(interval=None)
                rss += p.memory_info().rss
                found = True
            except psutil.Error:
                continue
        if not found:
            return None, None
        return round(min(100.0, cpu / self.cores), 1), round(rss / _MB, 1)

    def sample(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        proc_cpu, rss = self._proc_stats()
        return {
            "t": round(time.time(), 3),
            "cpu_pct": round(psutil.cpu_percent(interval=None), 1),
            "cores": self.cores,
            "proc_cpu_pct": proc_cpu,
            "ram_used_mb": round((vm.total - vm.available) / _MB, 1),
            "ram_total_mb": round(vm.total / _MB, 1),
            "proc_rss_mb": rss,
            "cpu_temp_c": self._cpu_temp(),
            "cpu_temp_source": self._cpu_temp_source(),
            "cpu_freq_mhz": self._cpu_freq(),
            "gpus": self._gpus(),
        }


def notes_for(sample: dict | None) -> list[str]:
    """Explica con honestidad qué sensores faltan y qué hardware usa cada entrenamiento."""
    out: list[str] = []
    s = sample or {}
    if s.get("cpu_temp_c") is None:
        from .cputemp import hint

        out.append(hint())
    elif s.get("cpu_temp_source") == "Thermal zone":
        from .cputemp import approx_note

        out.append(approx_note())
    if not s.get("gpus"):
        out.append(
            tr(
                "No se detectó GPU NVIDIA (nvidia-smi); las GPU AMD/Intel no se monitorizan.",
                "No NVIDIA GPU detected (nvidia-smi); AMD/Intel GPUs are not monitored.",
            )
        )
    out.append(
        tr(
            "El entrenamiento con scikit-learn usa solo la CPU. Las redes Keras usan la GPU con el "
            "backend PyTorch (CUDA en NVIDIA, ROCm en AMD/Linux); TensorFlow nativo en Windows "
            "(>= 2.11) ya no tiene versión con GPU: Elige el backend PyTorch en el paso 6.",
            "Training with scikit-learn uses only the CPU. Keras networks use the GPU through the "
            "PyTorch backend (CUDA on NVIDIA, ROCm on AMD/Linux); native Windows TensorFlow "
            "(>= 2.11) no longer has a GPU build: Choose the PyTorch backend in step 6.",
        )
    )
    return out


# ------------------------------------------------------------------ grabador por trabajo
class JobHwRecorder:
    """Hilo que muestrea el árbol del subproceso del trabajo y escribe ``hw.jsonl``."""

    def __init__(self, pid: int, path: Path, interval: float = 1.0, max_lines: int = MAX_LINES):
        self.path, self.interval, self.max_lines = Path(path), interval, max_lines
        self.sampler = HardwareSampler(pid)
        self.acc = SummaryAccumulator()
        self.last: dict | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._lines = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                while True:
                    s = self.sampler.sample()
                    with self._lock:
                        self.acc.add(s)
                        self.last = s
                    f.write(json.dumps(s, separators=(",", ":")) + "\n")
                    f.flush()
                    self._lines += 1
                    if self._lines > self.max_lines:
                        f.close()
                        self._compact()
                        f = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
                    if self._stop.wait(self.interval):
                        break
        except OSError:
            pass  # sin disco no hay historial; el trabajo no debe fallar por esto

    def _compact(self) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        lines = cap_lines(lines, self.max_lines)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._lines = len(lines)

    def live_summary(self) -> dict[str, Any]:
        with self._lock:
            return self.acc.summary()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.live_summary()


def read_samples(path: Path) -> list[dict]:
    """Lee hw.jsonl tolerando líneas truncadas."""
    out: list[dict] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out
