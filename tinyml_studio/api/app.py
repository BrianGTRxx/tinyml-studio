"""API local (FastAPI) del sidecar.

Seguridad (ver docs/ARCHITECTURE.md, "Cuidados obligatorios con Electron"):
- Solo se sirve en 127.0.0.1 (lo garantiza ``__main__``).
- Todas las rutas exigen el token de sesión (``Authorization: Bearer <token>``),
  comparado en tiempo constante. Sin token configurado la app no arranca.
- Se valida la cabecera ``Host`` (defensa contra DNS rebinding).
- Sin CORS: solo el proceso principal de Electron habla con el sidecar.
- Los archivos solo se leen dentro de las raíces permitidas.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from tinyml_studio import __version__
from tinyml_studio.i18n import SUPPORTED, set_lang, tr
from tinyml_studio.io import read_influx_csv
from tinyml_studio.profile import profile_dataframe

from .jobs import TERMINAL, JobError, JobManager

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class PathBody(BaseModel):
    path: str


class InspectBody(BaseModel):
    paths: list[str]
    lang: str | None = None  # "es" | "en"; None = el de ?lang= / por defecto


class ProfileBody(PathBody):
    label_column: str | None = None


class SuggestCleaningBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    multi_file: bool = False
    lang: str | None = None  # "es" | "en"


class JobBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    params: dict[str, Any]


def default_workdir() -> Path:
    return Path(os.environ.get("TINYML_WORKDIR") or Path.home() / ".tinyml_studio" / "workspace")


def create_app(
    token: str,
    allowed_roots: Iterable[str | Path] = (),
    workdir: str | Path | None = None,
    *,
    enable_test_jobs: bool = False,
    max_parallel: int = 1,
) -> FastAPI:
    if not token or len(token) < 16:
        raise ValueError(
            tr(
                "token de sesión requerido (>= 16 caracteres)",
                "session token required (>= 16 characters)",
            )
        )
    roots: list[Path] = [Path(r).resolve() for r in allowed_roots]

    def check_path(path: str) -> None:
        p = Path(path).resolve()
        if not any(p == r or r in p.parents for r in roots):
            raise JobError(
                403, tr("ruta fuera de las carpetas permitidas", "path outside the allowed folders")
            )
        if not p.is_file():
            raise JobError(404, tr("archivo no encontrado: {f}", "file not found: {f}", f=p.name))

    jobs = JobManager(
        workdir or default_workdir(),
        check_path,
        max_parallel=max_parallel,
        enable_test_jobs=enable_test_jobs,
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        jobs.shutdown()

    app = FastAPI(
        lifespan=lifespan,
        title="TinyML Studio sidecar",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.exception_handler(JobError)
    async def job_error(_request, exc: JobError):
        return JSONResponse({"detail": exc.msg}, status_code=exc.status)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        q_lang = request.query_params.get("lang")
        request.state.lang = q_lang if q_lang in SUPPORTED else None
        if request.state.lang:
            set_lang(
                request.state.lang
            )  # por petición: la tarea de la petición hereda este contexto
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host not in _LOOPBACK_HOSTS:
            return JSONResponse(
                {"detail": tr("host no permitido", "host not allowed")}, status_code=400
            )
        auth = request.headers.get("authorization", "")
        scheme, _, supplied = auth.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.encode(), token.encode()):
            return JSONResponse({"detail": tr("no autorizado", "unauthorized")}, status_code=401)
        return await call_next(request)

    def resolve_allowed(path: str) -> Path:
        p = Path(path).resolve()
        if not any(p == r or r in p.parents for r in roots):
            raise HTTPException(
                403, tr("ruta fuera de las carpetas permitidas", "path outside the allowed folders")
            )
        if not p.is_file():
            raise HTTPException(404, tr("archivo no encontrado", "file not found"))
        return p

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/roots")
    def add_root(body: PathBody) -> dict:
        """El proceso principal registra carpetas elegidas por el usuario (diálogo nativo)."""
        r = Path(body.path).resolve()
        if not r.is_dir():
            raise HTTPException(404, tr("carpeta no encontrada", "folder not found"))
        if r not in roots:
            roots.append(r)
        return {"roots": [str(x) for x in roots]}

    @app.post("/profile")
    def profile(body: ProfileBody) -> dict:
        p = resolve_allowed(body.path)
        try:
            res = read_influx_csv(p)
            report = profile_dataframe(res.df, label_column=body.label_column)
        except (ValueError, KeyError) as e:
            raise HTTPException(422, str(e)) from e
        report["source"] = {
            "file": p.name,
            "annotation_rows": res.annotation_rows,
            "separator": res.separator,
        }
        return report

    def collect_files(paths: list[str]) -> list[str]:
        """Rutas pedidas por la interfaz -> CSV, solo dentro de las raíces permitidas."""
        files: list[str] = []
        for raw in paths:
            p = Path(raw).resolve()
            if not any(p == r or r in p.parents for r in roots):
                raise HTTPException(
                    403,
                    tr("ruta fuera de las carpetas permitidas", "path outside the allowed folders"),
                )
            if p.is_dir():
                files += sorted(str(f) for f in p.glob("*.csv"))
            elif p.is_file():
                files.append(str(p))
            else:
                raise HTTPException(404, tr("no encontrado: {f}", "not found: {f}", f=p.name))
        return files

    @app.post("/inspect")
    def inspect(body: InspectBody, request: Request) -> dict:
        """Entiende uno o varios CSV (o carpetas) dentro de las raíces permitidas."""
        from tinyml_studio.dataset import inspect_files

        set_lang(body.lang or request.state.lang)
        files = collect_files(body.paths)
        try:
            return inspect_files(files)
        except (ValueError, KeyError) as e:
            raise HTTPException(422, str(e)) from e

    # ------------------------------------------------------------------ catálogos
    @app.get("/hardware")
    def hardware() -> dict:
        from tinyml_studio.hw import detect_hardware

        return detect_hardware()

    @app.get("/hardware/frameworks")
    def hardware_frameworks() -> dict:
        """Slow probe (a few seconds): which framework in THIS python env can see a GPU."""
        from tinyml_studio.hw import detect_hardware

        return detect_hardware(probe_frameworks_too=True)

    live_sampler: list = []  # muestreador perezoso compartido (cpu_percent necesita estado)
    live_lock = threading.Lock()

    @app.get("/hardware/live")
    def hardware_live() -> dict:
        from tinyml_studio.hw.monitor import HardwareSampler, notes_for

        with live_lock:
            if not live_sampler:
                live_sampler.append(HardwareSampler())
                time.sleep(0.25)  # intervalo mínimo para que cpu_percent no devuelva 0 siempre
            s = live_sampler[0].sample()
        return {**s, "notes": notes_for(s)}

    @app.get("/boards")
    def boards() -> dict:
        from tinyml_studio.boards import load_boards

        try:
            return load_boards()
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/suggest-cleaning")
    def suggest_cleaning_route(body: SuggestCleaningBody, request: Request) -> dict:
        """Recalcula la limpieza sugerida tras cambiar roles de columnas en la interfaz."""
        from tinyml_studio.clean.suggest import suggest_cleaning

        set_lang(body.lang or request.state.lang)
        try:
            return {"steps": suggest_cleaning(body.columns, body.alerts, body.multi_file)}
        except (KeyError, TypeError) as e:
            raise HTTPException(
                422,
                tr("columnas/alertas inválidas: {e}", "invalid columns/alerts: {e}", e=e),
            ) from e

    @app.get("/models")
    def models() -> dict:
        from .catalog import get_catalog

        return get_catalog()

    # ------------------------------------------------------------------ trabajos
    @app.post("/jobs", status_code=202)
    def submit_job(body: JobBody, request: Request) -> dict:
        params = dict(body.params)
        if request.state.lang and "lang" not in params:
            params["lang"] = request.state.lang  # ?lang= de la petición -> idioma del trabajo
        set_lang(params.get("lang") or request.state.lang)  # errores de validación en su idioma
        return jobs.submit(body.kind, params)

    @app.get("/jobs")
    def list_jobs() -> list[dict]:
        return jobs.list()

    @app.get("/jobs/{jid}")
    def get_job(jid: str) -> dict:
        out = jobs.public(jid)
        out["log_tail"] = jobs.log_tail(jid, 1500) if out["state"] == "failed" else ""
        return out

    @app.get("/jobs/{jid}/result")
    def job_result(jid: str) -> dict:
        return jobs.result(jid)

    @app.get("/jobs/{jid}/hw")
    def job_hw(jid: str) -> dict:
        return jobs.hw(jid)

    @app.post("/jobs/{jid}/cancel")
    def cancel_job(jid: str) -> dict:
        return jobs.cancel(jid)

    @app.delete("/jobs/{jid}")
    def delete_job(jid: str) -> dict:
        jobs.delete(jid)
        return {"deleted": jid}

    @app.get("/jobs/{jid}/files")
    def job_files(jid: str) -> list[dict]:
        return jobs.files(jid)

    @app.get("/jobs/{jid}/files/{rel:path}")
    def job_file(jid: str, rel: str):
        p, _ = jobs.read_file(jid, rel)
        return FileResponse(p, media_type="text/plain; charset=utf-8")

    @app.get("/jobs/{jid}/events")
    def job_events(jid: str, after: int = 0):
        """Progreso en vivo por Server-Sent Events (``id:`` = nº de secuencia)."""
        jobs.public(jid)  # 404 si no existe

        def stream():
            seq, idle = after, 0.0
            while True:
                evs, state = jobs.events_since(jid, seq)
                for e in evs:
                    seq = e["seq"]
                    yield f"id: {seq}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
                    if e.get("stage") == "end":
                        return
                if state in TERMINAL and not evs:
                    yield f"event: end\ndata: {json.dumps({'state': state})}\n\n"
                    return
                idle = 0.0 if evs else idle + 0.2
                if idle >= 15:
                    yield ": keep-alive\n\n"
                    idle = 0.0
                time.sleep(0.2)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    from .history import register_history_routes

    # /history, /history/compare, /jobs/{id}/params, /recipe/transfer
    register_history_routes(app, jobs, collect_files)
    return app
