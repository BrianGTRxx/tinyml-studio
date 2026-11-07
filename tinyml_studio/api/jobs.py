"""Gestor de trabajos asíncronos: cada trabajo es un subproceso aislado.

- Un subproceso por trabajo: TensorFlow solo se importa ahí, un fallo no tumba el sidecar y
  cancelar = matar el grupo de procesos.
- Progreso por eventos ``@@EVENT`` en stdout -> lista de eventos -> SSE.
- Estado en disco (``job.json``): al reiniciar el sidecar los trabajos 'running' pasan a
  'interrupted'.
- Máximo ``max_parallel`` trabajos a la vez (por defecto 1: no pelear por RAM/CPU).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tinyml_studio.environment import has_keras, keras_missing_message
from tinyml_studio.hw.monitor import (
    JobHwRecorder,
    downsample_even,
    notes_for,
    read_samples,
    summarize,
)
from tinyml_studio.i18n import tr, use_lang
from tinyml_studio.progress import PREFIX
from tinyml_studio.pyexec import python_argv

from .handlers import KINDS, NEEDS_RECIPE, NEEDS_RUN

JOB_ID = re.compile(r"^[0-9a-f]{12}$")
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
MAX_EVENTS = 5000
TEXT_SUFFIXES = {".json", ".md", ".txt", ".c", ".h", ".ino", ".csv", ".log"}
MAX_TEXT_BYTES = 2_000_000
HW_POINTS = 600  # muestras que devuelve /jobs/{id}/hw
ETA_ALPHA = 0.3  # suavizado exponencial de la duración total estimada


KERAS_KINDS = {"train_keras", "train_cnn1d", "export_tflite"}


def _needs_keras(kind: str, params: dict[str, Any]) -> bool:
    """Trabajos que no pueden correr sin Keras/TF. El benchmark NO está aquí: sigue con sus modelos
    clásicos y deja los presets Keras en ``models_skipped`` (ver train/bench.py)."""
    return kind in KERAS_KINDS


class JobError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status, self.msg = status, msg


class JobManager:
    def __init__(
        self,
        workdir: str | Path,
        check_path: Callable[[str], None],
        *,
        max_parallel: int = 1,
        enable_test_jobs: bool = False,
    ):
        self.workdir = Path(workdir).resolve()
        (self.workdir / "jobs").mkdir(parents=True, exist_ok=True)
        self.check_path = check_path
        self.max_parallel = max_parallel
        self.enable_test_jobs = enable_test_jobs
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict]] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._hw: dict[str, JobHwRecorder] = {}
        self._eta_total: dict[str, float] = {}
        self._queue: list[str] = []
        self._load()

    # ------------------------------------------------------------ persistencia
    @staticmethod
    def _tr(lang: str | None, es: str, en: str, **kw) -> str:
        """Texto en el idioma del trabajo (o el de la petición si el trabajo no fija uno)."""
        with use_lang(lang):
            return tr(es, en, **kw)

    def _dir(self, jid: str) -> Path:
        return self.workdir / "jobs" / jid

    def _save(self, job: dict) -> None:
        (self._dir(job["id"]) / "job.json").write_text(
            json.dumps(job, ensure_ascii=False), encoding="utf-8"
        )

    def _load(self) -> None:
        for d in (self.workdir / "jobs").iterdir():
            f = d / "job.json"
            if not (JOB_ID.match(d.name) and f.is_file()):
                continue
            try:
                job = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("state") in ("running", "queued"):
                with use_lang(job.get("lang")):
                    job["state"], job["error"] = (
                        "interrupted",
                        tr(
                            "el sidecar se reinició durante el trabajo",
                            "the sidecar restarted during the job",
                        ),
                    )
                f.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
            self._jobs[job["id"]] = job
            self._events[job["id"]] = []

    # ------------------------------------------------------------ envío
    def submit(self, kind: str, params: dict[str, Any]) -> dict:
        if kind not in KINDS or (kind.startswith("_") and not self.enable_test_jobs):
            raise JobError(
                422,
                tr("tipo de trabajo desconocido: {k}", "unknown job type: {k}", k=kind),
            )
        model_cls, _ = KINDS[kind]
        try:
            model_cls(**params)  # validación estricta (campos extra prohibidos)
        except ValidationError as e:
            raise JobError(
                422,
                tr("parámetros inválidos: ", "invalid parameters: ")
                + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()),
            ) from e
        if _needs_keras(kind, params) and not has_keras():
            raise JobError(422, keras_missing_message())
        if kind in NEEDS_RECIPE:
            files = params["recipe"].get("files")
            if not isinstance(files, list) or not files:
                raise JobError(422, tr("la receta no tiene archivos", "the recipe has no files"))
            for f in files:
                self.check_path(str(f))  # 403/404 si está fuera de las raíces
        if kind in NEEDS_RUN:
            rid = params["run_job"]
            if not JOB_ID.match(rid) or rid not in self._jobs:
                raise JobError(404, tr("run_job no existe", "run_job does not exist"))
            if self._jobs[rid]["state"] != "succeeded":
                raise JobError(
                    409,
                    tr("run_job no terminó correctamente", "run_job did not finish successfully"),
                )
        jid = uuid.uuid4().hex[:12]
        d = self._dir(jid)
        d.mkdir(parents=True)
        job = {
            "id": jid,
            "kind": kind,
            "state": "queued",
            "created": time.time(),
            "started": None,
            "finished": None,
            "lang": params.get("lang"),
            "progress": {
                "stage": "queued",
                "pct": 0,
                "msg": self._tr(params.get("lang"), "En cola", "Queued"),
            },
            "error": None,
            "workdir": str(self.workdir),
            "summary": {
                k: v for k, v in params.items() if k not in ("recipe", "params")
            },
            "device": params.get("device", "auto"),
        }
        (d / "params.json").write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8")
        with self._lock:
            self._jobs[jid], self._events[jid] = job, []
            self._queue.append(jid)
            self._save(job)
        self._pump()
        return self.public(jid)

    def _pump(self) -> None:
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j["state"] == "running")
            while self._queue and running < self.max_parallel:
                jid = self._queue.pop(0)
                if self._jobs[jid]["state"] != "queued":
                    continue
                self._start(jid)
                running += 1

    def _start(self, jid: str) -> None:
        job, d = self._jobs[jid], self._dir(jid)
        env = {k: v for k, v in os.environ.items() if k != "TINYML_SESSION_TOKEN"}
        env.update(
            PYTHONUNBUFFERED="1",
            PYTHONUTF8="1",  # Windows: sin esto stdout/archivos usan cp1252 y fallan con ↔, ñ…
            PYTHONIOENCODING="utf-8",
            TF_CPP_MIN_LOG_LEVEL="2",
            TINYML_JOB="1",
        )
        if job["device"] == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        log = open(d / "log.txt", "ab")  # noqa: SIM115
        proc = subprocess.Popen(
            python_argv(module="tinyml_studio.api.jobrunner", args=[str(d)]),
            stdout=subprocess.PIPE,
            stderr=log,
            stdin=subprocess.DEVNULL,
            env=env,
            **kwargs,
        )
        log.close()
        job.update(state="running", started=time.time())
        job["progress"] = {
            "stage": "start",
            "pct": 1,
            "msg": self._tr(job.get("lang"), "Iniciando…", "Starting…"),
        }
        self._procs[jid] = proc
        self._save(job)
        rec = JobHwRecorder(proc.pid, d / "hw.jsonl")
        self._hw[jid] = rec
        rec.start()
        threading.Thread(target=self._watch, args=(jid, proc), daemon=True).start()

    def _push(self, jid: str, ev: dict) -> None:
        with self._lock:
            evs = self._events[jid]
            ev["seq"] = (evs[-1]["seq"] + 1) if evs else 1
            ev["t"] = time.time()
            evs.append(ev)
            if len(evs) > MAX_EVENTS:
                del evs[: len(evs) - MAX_EVENTS]
            if ev.get("stage") not in ("done", "error", "end"):
                self._jobs[jid]["progress"] = {k: ev.get(k) for k in ("stage", "pct", "msg")}
                self._update_eta(jid)

    def _watch(self, jid: str, proc: subprocess.Popen) -> None:
        d = self._dir(jid)
        with open(d / "log.txt", "ab") as log:
            for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith(PREFIX):
                    try:
                        self._push(jid, json.loads(line[len(PREFIX) :]))
                        continue
                    except ValueError:
                        pass
                log.write(raw)
        code = proc.wait()
        rec = self._hw.pop(jid, None)
        hw_summary = rec.stop() if rec else None
        with self._lock:
            job = self._jobs[jid]
            if job["state"] == "cancelled":
                pass
            elif code == 0 and (d / "result.json").is_file():
                job["state"] = "succeeded"
                job["progress"] = {
                    "stage": "done",
                    "pct": 100,
                    "msg": self._tr(job.get("lang"), "Terminado", "Finished"),
                }
            else:
                job["state"] = "failed"
                err = d / "error.json"
                try:
                    job["error"] = json.loads(err.read_text(encoding="utf-8"))["message"]
                except (OSError, ValueError, KeyError):
                    job["error"] = self._tr(
                        job.get("lang"),
                        "el trabajo terminó con código {c}",
                        "the job ended with code {c}",
                        c=code,
                    )
            job["finished"] = time.time()
            if hw_summary and hw_summary["samples"]:
                job["hw_summary"] = hw_summary
            self._procs.pop(jid, None)
            self._eta_total.pop(jid, None)
            self._save(job)
        self._push(jid, {"stage": "end", "state": job["state"], "msg": job.get("error")})
        self._pump()

    # ------------------------------------------------------------ cancelar / borrar
    def cancel(self, jid: str) -> dict:
        job = self._get(jid)
        with self._lock:
            if job["state"] == "queued":
                job["state"], job["finished"] = "cancelled", time.time()
                self._save(job)
                self._push(
                    jid,
                    {
                        "stage": "end",
                        "state": "cancelled",
                        "msg": self._tr(job.get("lang"), "cancelado", "cancelled"),
                    },
                )
                return self.public(jid)
            if job["state"] != "running":
                raise JobError(
                    409,
                    tr("el trabajo ya está '{s}'", "the job is already '{s}'", s=job["state"]),
                )
            job["state"] = "cancelled"
            proc = self._procs.get(jid)
        if proc:
            self._kill(proc)
        return self.public(jid)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
            else:
                subprocess.run(["taskkill", "/pid", str(proc.pid), "/T", "/F"], capture_output=True)
        except (ProcessLookupError, PermissionError):
            pass

    def delete(self, jid: str) -> None:
        job = self._get(jid)
        if job["state"] not in TERMINAL:
            raise JobError(
                409, tr("cancela el trabajo antes de borrarlo", "cancel the job before deleting it")
            )
        with self._lock:
            self._jobs.pop(jid), self._events.pop(jid, None)
        shutil.rmtree(self._dir(jid), ignore_errors=True)

    def shutdown(self) -> None:
        for jid in list(self._procs):
            try:
                self.cancel(jid)
            except JobError:
                pass

    # ------------------------------------------------------------ consulta
    def _get(self, jid: str) -> dict:
        if not JOB_ID.match(jid) or jid not in self._jobs:
            raise JobError(404, tr("trabajo no encontrado", "job not found"))
        return self._jobs[jid]

    def _update_eta(self, jid: str) -> None:
        """Duración total estimada = transcurrido / fracción, suavizada (EMA). Las etapas no
        son lineales (p. ej. exportar es rápido y entrenar no), así que es solo una estimación."""
        j = self._jobs[jid]
        pct, started = (j.get("progress") or {}).get("pct"), j.get("started")
        if not started or pct is None or pct < 5 or pct >= 100:
            return
        raw = (time.time() - started) / (pct / 100.0)
        prev = self._eta_total.get(jid)
        self._eta_total[jid] = raw if prev is None else ETA_ALPHA * raw + (1 - ETA_ALPHA) * prev

    def _timing(self, j: dict) -> tuple[float | None, float | None]:
        started = j.get("started")
        if not started:
            return None, None
        end = j.get("finished") or time.time()
        elapsed = round(max(0.0, end - started), 1)
        total = self._eta_total.get(j["id"])
        if j["state"] != "running" or total is None:
            return elapsed, None
        return elapsed, round(max(0.0, total - elapsed), 1)

    def public(self, jid: str) -> dict:
        j = self._get(jid)
        elapsed, eta = self._timing(j)
        return {
            "elapsed_s": elapsed,
            "eta_s": eta,
            "hw_summary": j.get("hw_summary"),
            **{
                k: j[k]
                for k in (
                    "id",
                    "kind",
                    "state",
                    "created",
                    "started",
                    "finished",
                    "progress",
                    "error",
                    "summary",
                    "device",
                )
            },
        }

    def hw(self, jid: str) -> dict:
        """Historial de hardware del trabajo (hasta HW_POINTS muestras repartidas) + resumen."""
        j = self._get(jid)
        samples = read_samples(self._dir(jid) / "hw.jsonl")
        rec = self._hw.get(jid)
        summary = j.get("hw_summary") or (rec.live_summary() if rec else None)
        if summary is None and samples:
            summary = summarize(samples)
        return {
            "samples": downsample_even(samples, HW_POINTS),
            "summary": summary,
            "notes": notes_for(samples[-1] if samples else None),
            "running": j["state"] == "running",
        }

    def list(self) -> list[dict]:
        with self._lock:
            ids = sorted(self._jobs, key=lambda i: self._jobs[i]["created"], reverse=True)
        return [self.public(i) for i in ids]

    def events_since(self, jid: str, after: int) -> tuple[list[dict], str]:
        job = self._get(jid)
        with self._lock:
            return [e for e in self._events.get(jid, []) if e["seq"] > after], job["state"]

    def result(self, jid: str) -> dict:
        job = self._get(jid)
        if job["state"] != "succeeded":
            raise JobError(409, tr("el trabajo está '{s}'", "the job is '{s}'", s=job["state"]))
        return json.loads((self._dir(jid) / "result.json").read_text(encoding="utf-8"))

    def files(self, jid: str) -> list[dict]:
        self._get(jid)
        out = self._dir(jid) / "out"
        return (
            [
                {"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size}
                for p in sorted(out.rglob("*"))
                if p.is_file()
            ]
            if out.is_dir()
            else []
        )

    def read_file(self, jid: str, rel: str) -> tuple[Path, bool]:
        self._get(jid)
        out = (self._dir(jid) / "out").resolve()
        p = (out / rel).resolve()
        if out not in p.parents or not p.is_file():
            raise JobError(404, tr("archivo no encontrado", "file not found"))
        if p.suffix.lower() not in TEXT_SUFFIXES:
            raise JobError(
                415,
                tr(
                    "solo se sirven archivos de texto; usa 'Guardar archivos…'",
                    "only text files are served; use 'Save files…'",
                ),
            )
        if p.stat().st_size > MAX_TEXT_BYTES:
            raise JobError(
                413, tr("archivo demasiado grande para mostrarlo", "file too large to display")
            )
        return p, True

    def log_tail(self, jid: str, n: int = 4000) -> str:
        self._get(jid)
        f = self._dir(jid) / "log.txt"
        return f.read_bytes()[-n:].decode("utf-8", "replace") if f.is_file() else ""
