"""Historial de ejecuciones, comparación de corridas y reutilización de configuraciones.

Solo LEE lo que ya persiste ``JobManager`` (``<workdir>/jobs/<id>/{job,params,result}.json``);
no guarda nada propio. Nunca devuelve rutas del servidor: los archivos de datos salen como
nombres base. Las funciones puras (``history_entry``, ``compare_rows``...) trabajan con una
carpeta de trabajo, así se prueban con trabajos sintéticos sin lanzar subprocesos.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path, PurePath
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from tinyml_studio.i18n import set_lang, tr

from .jobs import JOB_ID, TERMINAL, JobError

HISTORY_LIMIT = 200
COMPARE_MIN, COMPARE_MAX, COMPARE_MAX_ROWS = 2, 6, 40
MAX_MODELS_SHOWN = 12
METRIC_KEYS = (
    "f1_macro_val",
    "f1_macro_test",
    "accuracy_test",
    "mcc_test",
    "mAP_test",
    "r2_test",
    "mae_test",
    "size_bytes",
    "fit_seconds",
    "predict_us_per_row",
)
_CACHE: dict[tuple[str, int, int], dict] = {}
_CACHE_MAX = 600


class CompareBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[str]
    lang: str | None = None


class TransferBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    old_recipe: dict[str, Any]
    paths: list[str]
    lang: str | None = None


# ------------------------------------------------------------------ lectura de disco
def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jdir(workdir: Path, jid: str) -> Path:
    return Path(workdir) / "jobs" / jid


def recipe_sha(recipe: dict | None) -> str | None:
    """sha256 de la receta canónica (mismo cálculo que ``reproducibility.recipe_sha256``)."""
    if not isinstance(recipe, dict):
        return None
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()


def _basename(f: Any) -> str:
    s = str(f).replace("\\", "/")
    return PurePath(s).name or s


def job_recipe(workdir: Path, jid: str, depth: int = 0) -> dict | None:
    """Receta del trabajo; los de exportación la heredan del ``run_job`` que exportan."""
    params = _read(_jdir(workdir, jid) / "params.json") or {}
    if isinstance(params.get("recipe"), dict):
        return params["recipe"]
    src = params.get("run_job")
    if depth < 3 and isinstance(src, str) and JOB_ID.match(src):
        return job_recipe(workdir, src, depth + 1)
    return None


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _g(d: Any, *path: str) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _model_names(kind: str, res: dict) -> list[str]:
    r = res.get("results") or {}
    if kind == "benchmark":
        return [x.get("name") for x in r.get("leaderboard", []) if x.get("name")]
    if kind == "train":
        return list(r.get("models") or {})
    if kind == "train_keras":
        return ["keras_mlp"] if res.get("family") == "keras_mlp" else [str(res.get("family"))]
    if kind == "train_cnn1d":
        return ["keras_cnn1d"]
    return []


def headline(kind: str, res: dict, params: dict) -> dict:
    """Cifra principal de una corrida, según su tipo."""
    r = res.get("results") or {}
    out: dict[str, Any] = {
        "metric_name": None,
        "value_val": None,
        "value_test": None,
        "selected_model": None,
    }
    if kind == "train":
        best = _g(r, "selection", "best")
        m = (r.get("models") or {}).get(best) or {}
        key = _g(r, "selection", "metric") or (
            "f1_macro" if r.get("task") == "classification" else "r2"
        )
        out.update(
            metric_name=key,
            value_val=_num(_g(m, "val", key)),
            value_test=_num(_g(m, "test", key)),
            selected_model=best,
        )
    elif kind in ("train_keras", "train_cnn1d"):
        key = "f1_macro" if r.get("task", "classification") == "classification" else "r2"
        out.update(
            metric_name=key,
            value_val=_num(_g(r, "val", key)),
            value_test=_num(_g(r, "test", key)),
            selected_model="keras_cnn1d" if kind == "train_cnn1d" else "keras_mlp",
        )
    elif kind == "benchmark":
        sel = _g(r, "selection", "selected")
        key = _g(r, "selection", "metric") or "f1_macro"
        row = next((x for x in r.get("leaderboard", []) if x.get("name") == sel), {})
        out.update(
            metric_name=key,
            value_val=_num(_g(row, "metrics", "val", key)),
            value_test=_num(_g(row, "metrics", "test", key)),
            selected_model=sel,
        )
    elif kind in ("export_c", "export_tflite"):
        b = r.get("budget") or {}
        out.update(
            metric_name="export",
            selected_model=params.get("model"),
            board=r.get("board"),
            flash_verdict=_g(b, "flash_data", "verdict_code"),
            ram_verdict=_g(b, "ram_working", "verdict_code"),
            flash_bytes=_num(_g(b, "flash_data", "used_bytes")),
            ram_bytes=_num(_g(b, "ram_working", "used_bytes")),
        )
    elif kind == "clean":
        sizes = _g(res, "split", "sizes") or {}
        out.update(
            metric_name="rows",
            value_val=_num(sizes.get("val")),
            value_test=_num(sizes.get("test")),
            split_sizes=sizes,
        )
    return out


def history_entry(workdir: Path, job: dict) -> dict:
    """Una fila del historial a partir de un ``job.json`` (o de ``JobManager.public``)."""
    jid = job["id"]
    d = _jdir(workdir, jid)
    disk = _read(d / "job.json") or {}
    kind = job.get("kind") or disk.get("kind")
    params = _read(d / "params.json") or {}
    recipe = job_recipe(workdir, jid)
    res: dict = {}
    state = job.get("state") or disk.get("state")
    if state == "succeeded":
        res = _read(d / "result.json") or {}
    created, started, finished = (job.get("created"), disk.get("started"), job.get("finished"))
    start = started or created
    files = [_basename(f) for f in ((recipe or {}).get("files") or [])]
    return {
        "id": jid,
        "kind": kind,
        "state": state,
        "created": created,
        "finished": finished,
        "duration_s": round(finished - start, 2) if finished and start else None,
        "files": files,
        "recipe_sha": recipe_sha(recipe),
        "family": res.get("family"),
        "models": _model_names(kind, res)[:MAX_MODELS_SHOWN],
        "n_models": len(_model_names(kind, res)),
        "headline": headline(kind, res, params) if res else None,
        "hw": disk.get("hw_summary") or job.get("hw_summary"),
        "error": job.get("error") or disk.get("error"),
    }


def _cached_entry(workdir: Path, job: dict) -> dict:
    d = _jdir(workdir, job["id"])
    try:
        sig = (str(d), (d / "job.json").stat().st_mtime_ns, (d / "result.json").stat().st_mtime_ns)
    except OSError:
        sig = (str(d), 0, 0)
    if sig not in _CACHE:
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.clear()
        _CACHE[sig] = history_entry(workdir, job)
    return _CACHE[sig]


def build_history(workdir: Path, jobs: list[dict], limit: int = HISTORY_LIMIT) -> list[dict]:
    """``jobs``: lista de ``JobManager.list()`` (ya más reciente primero)."""
    done = [
        j for j in jobs if j.get("state") in TERMINAL and not str(j.get("kind", "")).startswith("_")
    ]
    done.sort(key=lambda j: j.get("created") or 0, reverse=True)
    return [_cached_entry(workdir, j) for j in done[:limit]]


def job_params(workdir: Path, job: dict) -> dict:
    """Parámetros con los que se lanzó el trabajo (nada del servidor: solo lo que envió la UI)."""
    jid = job["id"]
    params = _read(_jdir(workdir, jid) / "params.json")
    if not isinstance(params, dict):
        raise JobError(404, tr("parámetros no disponibles", "parameters not available"))
    recipe = job_recipe(workdir, jid)
    return {
        "id": jid,
        "kind": job.get("kind"),
        "params": params,
        "files": [_basename(f) for f in ((recipe or {}).get("files") or [])],
        "recipe_sha": recipe_sha(recipe),
    }


# ------------------------------------------------------------------ comparación
def _row(label: str, jid: str, kind: str, metrics: dict) -> dict:
    return {
        "id": jid,
        "kind": kind,
        "label": label,
        "metrics": {k: _num(metrics.get(k)) for k in METRIC_KEYS},
    }


def _split_metrics(sp: Any, extra: Any = None) -> dict:
    """val/test de un modelo: acepta el bloque simple y el de calidad (mcc/mAP)."""
    out = dict(sp) if isinstance(sp, dict) else {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            out.setdefault(k, v)
    return out


def _vals(val: dict, test: dict, **extra) -> dict:
    return {
        "f1_macro_val": val.get("f1_macro"),
        "f1_macro_test": test.get("f1_macro"),
        "accuracy_test": test.get("accuracy"),
        "mcc_test": test.get("mcc"),
        "mAP_test": test.get("mAP"),
        "r2_test": test.get("r2"),
        "mae_test": test.get("mae"),
        **extra,
    }


def compare_rows(job: dict, res: dict) -> list[dict]:
    """Filas de métricas de un trabajo ya terminado (benchmark: una por modelo del ranking)."""
    jid, kind = job["id"], job.get("kind", "")
    r = res.get("results") or {}
    tag = f"{kind}·{jid[:6]}"
    if kind == "benchmark":
        rows = []
        for x in r.get("leaderboard", []):
            mt = x.get("metrics") or {}
            rows.append(
                _row(
                    f"{tag} · {x.get('name')}",
                    jid,
                    kind,
                    _vals(
                        mt.get("val") or {},
                        mt.get("test") or {},
                        size_bytes=x.get("size_bytes"),
                        fit_seconds=x.get("fit_seconds"),
                        predict_us_per_row=x.get("predict_us_per_row"),
                    ),
                )
            )
        return rows
    if kind == "train":
        best = _g(r, "selection", "best")
        m = (r.get("models") or {}).get(best) or {}
        qt = _g(m, "quality", "splits", "test")
        return [
            _row(
                f"{tag} · {best}",
                jid,
                kind,
                _vals(
                    _split_metrics(m.get("val")),
                    _split_metrics(m.get("test"), qt),
                    fit_seconds=m.get("fit_seconds"),
                    predict_us_per_row=m.get("predict_us_per_row_cpu_cloud"),
                ),
            )
        ]
    if kind in ("train_keras", "train_cnn1d"):
        val = r.get("val") or {}
        test = r.get("test") or {}
        qt = _g(r, "quality", "splits", "test")
        return [
            _row(
                f"{tag} · {res.get('family') or kind}",
                jid,
                kind,
                _vals(val, _split_metrics(test, qt), fit_seconds=r.get("fit_seconds")),
            )
        ]
    if kind in ("export_c", "export_tflite"):
        size = _g(r, "budget", "flash_data", "used_bytes")
        if size is None:
            size = _g(r, "sizes_bytes", "tflite_int8")
        return [
            _row(f"{tag} · {r.get('board') or ''}".strip(" ·"), jid, kind, {"size_bytes": size})
        ]
    return [_row(tag, jid, kind, {})]


def compare_jobs(
    workdir: Path, jobs: list[dict], results: dict[str, dict], lang_notes: bool = True
) -> dict:
    """Une filas + banderas de comparabilidad. ``jobs``: dicts públicos en el orden pedido."""
    rows: list[dict] = []
    for j in jobs:
        rows += compare_rows(j, results[j["id"]])
    if len(rows) > COMPARE_MAX_ROWS:
        raise JobError(
            422,
            tr(
                "demasiadas filas para comparar ({n}; máximo {m}): Elige menos modelos/trabajos",
                "too many rows to compare ({n}; maximum {m}): Pick fewer models/jobs",
                n=len(rows),
                m=COMPARE_MAX_ROWS,
            ),
        )
    entries = [history_entry(workdir, j) for j in jobs]
    file_sets = {tuple(sorted(e["files"])) for e in entries}
    shas = {e["recipe_sha"] for e in entries}
    same_dataset = len(file_sets) == 1 and () not in file_sets
    same_recipe = len(shas) == 1 and None not in shas
    notes: list[str] = []
    if not same_dataset:
        notes.append(
            tr(
                "Los trabajos usan datasets distintos: Las métricas no son directamente "
                "comparables",
                "The jobs use different datasets: The metrics are not directly comparable",
            )
        )
    elif not same_recipe:
        notes.append(
            tr(
                "Mismo dataset pero recetas distintas (limpieza, partición o columnas): "
                "compara con cuidado",
                "Same dataset but different recipes (cleaning, split or columns): Compare "
                "with care",
            )
        )
    tasks = {_g(results[j["id"]], "results", "task") for j in jobs} - {None}
    if len(tasks) > 1:
        notes.append(
            tr(
                "Hay tareas distintas ({t}): No mezcles F1 con R²",
                "Different tasks ({t}): Do not mix F1 with R²",
                t=", ".join(sorted(map(str, tasks))),
            )
        )
    tests = {_g(results[j["id"]], "results", "rows", "test") for j in jobs} - {None}
    if len(tests) > 1:
        notes.append(
            tr(
                "Los conjuntos de test tienen tamaños distintos ({t}): Las puntuaciones de test "
                "no son sobre las mismas filas",
                "The test sets have different sizes ({t}): Test scores are not on the same rows",
                t=", ".join(str(int(x)) for x in sorted(tests)),
            )
        )
    kinds = {j.get("kind") for j in jobs}
    if kinds & {"clean", "export_c", "export_tflite"}:
        notes.append(
            tr(
                "Limpieza y exportación no tienen métricas de modelo (solo tamaño en exportación)",
                "Cleaning and export jobs have no model metrics (only size for exports)",
            )
        )
    return {"rows": rows, "same_dataset": same_dataset, "same_recipe": same_recipe, "notes": notes}


# ------------------------------------------------------------------ rutas
def register_history_routes(
    app: FastAPI,
    jobs: Any,
    collect_files: Callable[[list[str]], list[str]],
) -> None:
    """Registra ``/history``, ``/history/compare``, ``/jobs/{id}/params`` y ``/recipe/transfer``.

    ``jobs`` es el ``JobManager``; ``collect_files`` valida y expande rutas (mismo código que
    ``/inspect``: raíces permitidas, carpetas -> *.csv).
    """

    @app.get("/history")
    def history() -> list[dict]:
        return build_history(jobs.workdir, jobs.list())

    @app.get("/jobs/{jid}/params")
    def params_of(jid: str) -> dict:
        return job_params(jobs.workdir, jobs.public(jid))

    @app.post("/history/compare")
    def compare(body: CompareBody, request: Request) -> dict:
        set_lang(body.lang or request.state.lang)
        ids = body.ids
        if not COMPARE_MIN <= len(ids) <= COMPARE_MAX or len(set(ids)) != len(ids):
            raise JobError(
                422,
                tr(
                    "indica entre {a} y {b} trabajos distintos",
                    "give between {a} and {b} different jobs",
                    a=COMPARE_MIN,
                    b=COMPARE_MAX,
                ),
            )
        pubs = [jobs.public(i) for i in ids]  # 404 si algún id no existe / es inválido
        results = {i: jobs.result(i) for i in ids}  # 409 si alguno no terminó bien
        return compare_jobs(jobs.workdir, pubs, results)

    @app.post("/recipe/transfer")
    def transfer(body: TransferBody, request: Request) -> dict:
        from tinyml_studio.dataset import inspect_files
        from tinyml_studio.dataset.transfer import transfer_recipe

        set_lang(body.lang or request.state.lang)
        files = collect_files(body.paths)
        try:
            report = inspect_files(files)
            out = transfer_recipe(body.old_recipe, report)
        except (ValueError, KeyError, TypeError) as e:
            raise JobError(422, str(e)) from e
        return {
            "report": report,
            "recipe": out["recipe"],
            "notes": out["notes"],
            "applied": out["applied"],
        }
