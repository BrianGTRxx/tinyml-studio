"""Ejecuta un trabajo en un subproceso: ``python -m tinyml_studio.api.jobrunner <carpeta>``."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from tinyml_studio.i18n import collected_pairs, set_lang, tr
from tinyml_studio.progress import emit

from .handlers import KINDS

KERAS_KINDS = {"train_keras", "train_cnn1d"}


def _setup_compute(kind: str, params: dict, ctx: dict) -> None:
    """Backend de Keras y dispositivo del trabajo; debe correr ANTES de importar keras."""
    device = params.get("device", "auto")
    backend = params.get("backend", "tensorflow")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    if (
        device == "gpu" and kind in KERAS_KINDS
    ):  # nunca falla: sin GPU avisa y sigue en CPU (sondeo en subprocesos)
        from tinyml_studio.train.backend import plan_compute

        plan = plan_compute(backend, device)
        backend = plan["backend"]
        ctx["compute_notes"] = plan["notes"]
        for note in plan["notes"]:
            emit("compute", None, note)
    ctx["backend"] = backend
    if backend == "torch" and kind in KERAS_KINDS:
        os.environ["KERAS_BACKEND"] = "torch"


def main(argv: list[str]) -> int:
    job_dir = Path(argv[1])
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    params = json.loads((job_dir / "params.json").read_text(encoding="utf-8"))
    os.environ["TINYML_JOB"] = "1"
    set_lang(params.get("lang"))
    model_cls, handler = KINDS[job["kind"]]
    ctx = {"out": job_dir / "out", "job_dir": job_dir}
    _setup_compute(job["kind"], params, ctx)
    ctx["out"].mkdir(parents=True, exist_ok=True)
    if params.get("run_job"):
        ctx["run_dir"] = Path(job["workdir"]) / "jobs" / params["run_job"] / "out"
    try:
        result = handler(model_cls(**params), ctx)
        if isinstance(result, dict):
            result["_i18n"] = collected_pairs()  # para cambiar de idioma sin repetir el trabajo
        (job_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8"
        )
        emit("done", 100, tr("Terminado", "Finished"))
        return 0
    except BaseException as e:  # noqa: BLE001
        (job_dir / "error.json").write_text(
            json.dumps(
                {
                    "type": type(e).__name__,
                    "message": str(e)[:2000],
                    "traceback": traceback.format_exc()[-3000:],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        emit("error", None, f"{type(e).__name__}: {str(e)[:300]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
