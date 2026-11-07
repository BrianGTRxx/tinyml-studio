"""Trabajos que se ejecutan en un subproceso (``jobrunner``). Cada uno devuelve un dict JSON.

Las rutas de entrada ya las validó el gestor de trabajos (archivos dentro de las raíces
permitidas, ``run_job`` = id de un trabajo anterior); las de salida son siempre ``<job>/out``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lang: str | None = None  # idioma de los mensajes ("es" | "en"); None = por defecto / env


class Base(_P):
    recipe: dict[str, Any]
    seed: int = 42
    exclude: list[str] = Field(default_factory=list)
    # 'gpu' = usar la GPU si algún framework de este entorno la ve; si no, CPU con aviso (no falla)
    device: Literal["auto", "cpu", "gpu"] = "auto"


class CleanParams(Base):
    max_columns: int = Field(default=24, ge=1, le=100)


class TrainParams(Base):
    models: list[str] | None = None
    params: dict[str, dict[str, Any]] | None = None


class GroupValidationParams(Base):
    model: str = "random_forest"
    group: str | None = None  # columna de sesión; por defecto la partición de la receta o 'source'


class TuneParams(Base):
    model: str = "random_forest"
    flash_kb: float | None = Field(default=None, gt=0)
    ram_kb: float | None = Field(default=None, gt=0)
    n_trials: int = Field(default=24, ge=1, le=200)


class AnomalyParams(Base):
    model: Literal["isolation_forest", "one_class_svm"] = "isolation_forest"
    normal_classes: list[str] = Field(default_factory=list)
    false_alarm: float = Field(default=0.01, gt=0, lt=0.5)


OptimizerName = Literal["adam", "adamw", "nadam", "rmsprop", "sgd", "adagrad"]
ScheduleName = Literal["constant", "cosine", "exp_decay"]


class _KerasTrain(Base):
    """Optimizador, programa de la tasa de aprendizaje y backend de los entrenadores Keras."""

    optimizer: OptimizerName = "adam"
    weight_decay: float = Field(default=0.004, ge=0, le=1)  # solo adamw
    nesterov: bool = False  # solo sgd (momentum fijo 0.9)
    lr_schedule: ScheduleName = "constant"
    lr_final_frac: float = Field(
        default=0.01, gt=0, le=1
    )  # tasa final / inicial (cosine, exp_decay)
    backend: Literal["tensorflow", "torch"] = "tensorflow"


def _opt_kwargs(p: _KerasTrain, ctx: dict) -> dict[str, Any]:
    return dict(
        optimizer=p.optimizer,
        weight_decay=p.weight_decay,
        nesterov=p.nesterov,
        lr_schedule=p.lr_schedule,
        lr_final_frac=p.lr_final_frac,
        backend=ctx.get("backend", p.backend),
        device=p.device,
        compute_notes=tuple(ctx.get("compute_notes", ())),
    )


class KerasParams(_KerasTrain):
    hidden: list[int] = Field(default_factory=lambda: [32, 16])
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=128, ge=1, le=65536)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    patience: int = Field(default=8, ge=1, le=200)
    activation: Literal["relu", "tanh", "sigmoid"] = "relu"
    dropout: float = Field(default=0.0, ge=0, le=0.5)
    l2: float = Field(default=0.0, ge=0, le=1)


class CnnParams(_KerasTrain):
    window: int = Field(default=16, ge=2, le=512)
    stride: int = Field(default=4, ge=1, le=512)
    filters: list[int] = Field(default_factory=lambda: [16, 32])
    kernel: int = Field(default=3, ge=1, le=15)
    head: Literal["gap", "flatten"] = "gap"
    epochs: int = Field(default=60, ge=1, le=1000)
    batch_size: int = Field(default=128, ge=1, le=65536)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    patience: int = Field(default=6, ge=1, le=200)




class KerasPreset(BaseModel):
    """Hiperparámetros de un MLP Keras dentro de un benchmark (acotados)."""

    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, max_length=32, pattern=r"^[A-Za-z0-9_\-]+$")
    hidden: list[int] = Field(default_factory=lambda: [32, 16], max_length=6)
    epochs: int = Field(default=60, ge=1, le=500)
    batch_size: int = Field(default=128, ge=1, le=65536)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    patience: int = Field(default=6, ge=1, le=100)
    activation: Literal["relu", "tanh", "sigmoid"] = "relu"
    dropout: float = Field(default=0.0, ge=0, le=0.5)
    l2: float = Field(default=0.0, ge=0, le=1)
    # opcionales: None = no se envían (el MLP usa Adam constante); ver KERAS_KEYS en train/bench.py
    optimizer: OptimizerName | None = None
    weight_decay: float | None = Field(default=None, ge=0, le=1)
    nesterov: bool | None = None
    lr_schedule: ScheduleName | None = None
    lr_final_frac: float | None = Field(default=None, gt=0, le=1)

    @field_validator("hidden")
    @classmethod
    def _hidden(cls, v: list[int]) -> list[int]:
        if not v or any(h < 1 or h > 1024 for h in v):
            raise ValueError("hidden: 1..6 capas de 1..1024 neuronas")
        return v


class BenchmarkParams(Base):
    models: list[str] | None = Field(default=None, max_length=40)
    keras: list[KerasPreset] = Field(default_factory=list, max_length=8)
    params: dict[str, dict[str, Any]] | None = None
    cv_folds: int = Field(default=0, ge=0, le=10)
    selection: Literal[
        "f1_macro",
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "mAP",
        "roc_auc_macro",
        "r2",
        "mae",
        "rmse",
    ] = "f1_macro"
    max_flash_kb: float | None = Field(default=None, gt=0)
    n_boot: int = Field(default=200, ge=20, le=2000)

    @field_validator("cv_folds")
    @classmethod
    def _cv(cls, v: int) -> int:
        if v == 1:
            raise ValueError("cv_folds debe ser 0 o 2..10")
        return v


class ExportCParams(_P):
    run_job: str
    # subcarpeta de un benchmark (p. ej. "classical" o "keras_mlp_h32-16"); None = raíz del trabajo
    run_subdir: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_:\-]+$")
    model: str | None = None
    board: str = "esp32-s3"
    flash_kb: float | None = Field(default=None, gt=0)
    ram_kb: float | None = Field(default=None, gt=0)
    device: Literal["auto", "cpu"] = "auto"
    compact: bool = False  # árboles/bosques: tablas compactas (opcional, ver docs/research)


class ExportTflParams(ExportCParams):
    int8_io: bool = False


class ExportOnnxParams(_P):
    run_job: str
    run_subdir: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_:\-]+$")
    model: str | None = None
    reject_below: float | None = Field(default=None, gt=0, lt=1)
    device: Literal["auto", "cpu"] = "auto"


class ImportanceParams(_P):
    run_job: str
    run_subdir: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_:\-]+$")
    model: str | None = None
    n_repeats: int = Field(default=5, ge=1, le=20)
    device: Literal["auto", "cpu"] = "auto"


class PredictParams(_P):
    """Probar un modelo clásico entrenado con valores crudos escritos por el usuario."""

    run_job: str
    run_subdir: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_:\-]+$")
    model: str | None = None
    # None = solo describir las entradas del modelo (para dibujar el formulario)
    values: dict[str, Any] | None = None

    @field_validator("values")
    @classmethod
    def _small(cls, v):
        if v is not None and (len(v) > 200 or any(len(str(x)) > 200 for x in v.values())):
            raise ValueError("demasiados valores")
        return v


class LatencyParams(_P):
    run_job: str
    run_subdir: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_:\-]+$")
    model: str | None = None
    board: str = "all"
    target_ms: float | None = Field(default=None, gt=0)
    device: Literal["auto", "cpu"] = "auto"


class RejectParams(_P):
    run_job: str
    run_subdir: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_:\-]+$")
    model: str | None = None
    target_accuracy: float = Field(default=0.95, gt=0.5, lt=1.0)
    holdout: bool = True
    device: Literal["auto", "cpu"] = "auto"


class SleepParams(_P):
    seconds: float = 5.0
    device: str = "auto"


def _summary(fn, res) -> str:
    return fn(res)


def h_clean(p: CleanParams, ctx: dict) -> dict:
    from tinyml_studio.clean import prepare_splits
    from tinyml_studio.clean.compare import (
        compare_stats,
        histogram_data,
        numeric_columns,
        unit_view,
    )

    emit(
        "prepare",
        10,
        tr("Organizando, partiendo y limpiando…", "Organizing, splitting and cleaning…"),
    )
    prep = prepare_splits(p.recipe, seed=p.seed)
    cols: list[str] = []
    for s in prep.pipeline.steps:
        cols += list(s.params.get("columns", []))
    before, after = unit_view(prep)  # mismas unidades: sin el escalado
    cols = numeric_columns(before, list(dict.fromkeys(cols)))[: p.max_columns]
    emit(
        "charts",
        70,
        tr("Calculando estadísticas antes/después…", "Computing before/after statistics…"),
    )
    sizes = {s: int(len(df)) for s, df in prep.clean.items()}
    dist = {}
    if prep.target and prep.target in prep.clean["train"].columns:
        for s, df in prep.clean.items():
            dist[s] = {str(k): int(v) for k, v in df[prep.target].value_counts().items()}
    return {
        "steps": prep.pipeline.to_list(),
        "columns": cols,
        "stats": compare_stats(before, after, cols),
        "histograms": {c: histogram_data(before[c], after[c]) for c in cols},
        "units_note": tr(
            "gráficas y estadísticas en unidades originales: El escalado no se incluye",
            "charts and statistics in original units: Scaling is not included",
        ),
        "split": {
            "sizes": sizes,
            "class_distribution": dist,
            "target": prep.target,
            "strategy": p.recipe.get("split", {}),
        },
        "raw_rows_before_cleaning": {s: int(len(df)) for s, df in prep.raw.items()},
    }


def h_train(p: TrainParams, ctx: dict) -> dict:
    from tinyml_studio.train import train_models
    from tinyml_studio.train.report import summary

    res = train_models(
        p.recipe,
        p.models,
        seed=p.seed,
        params=p.params,
        exclude=tuple(p.exclude),
        out_dir=ctx["out"],
    )
    return {
        "results": res,
        "summary_es": summary(res),
        "summary": summary(res),
        "family": "classical",
    }


def h_tune(p: TuneParams, ctx: dict) -> dict:
    from tinyml_studio.train.tune import tune

    res = tune(
        p.recipe,
        p.model,
        flash_kb=p.flash_kb,
        ram_kb=p.ram_kb,
        n_trials=p.n_trials,
        seed=p.seed,
        exclude=tuple(p.exclude),
    )
    return {"results": res, "summary": res["summary"], "family": "tuning"}


def h_anomaly(p: AnomalyParams, ctx: dict) -> dict:
    from tinyml_studio.train.anomaly import anomaly_detection

    res = anomaly_detection(
        p.recipe,
        model=p.model,
        normal_classes=p.normal_classes or None,
        false_alarm=p.false_alarm,
        seed=p.seed,
        exclude=tuple(p.exclude),
        out_dir=ctx["out"],
    )
    return {"results": res, "summary": res["summary"], "family": "anomaly"}


def h_validate_groups(p: GroupValidationParams, ctx: dict) -> dict:
    from tinyml_studio.train.groups import group_validation
    from tinyml_studio.train.groups import summary as group_summary

    res = group_validation(
        p.recipe, model=p.model, group=p.group, seed=p.seed, exclude=tuple(p.exclude)
    )
    return {"results": res, "summary": group_summary(res), "family": "group_validation"}


def h_train_keras(p: KerasParams, ctx: dict) -> dict:
    from tinyml_studio.train import keras_mlp as K
    from tinyml_studio.train import report as R

    common = dict(
        hidden=tuple(p.hidden),
        epochs=p.epochs,
        batch_size=p.batch_size,
        learning_rate=p.learning_rate,
        patience=p.patience,
        activation=p.activation,
        dropout=p.dropout,
        l2=p.l2,
        seed=p.seed,
        exclude=tuple(p.exclude),
        out_dir=ctx["out"],
        **_opt_kwargs(p, ctx),
    )
    res = K.train_keras_mlp(p.recipe, **common)
    return {
        "results": res,
        "summary_es": R.keras_summary(res),
        "summary": R.keras_summary(res),
        "family": "keras_mlp",
    }


def h_train_cnn1d(p: CnnParams, ctx: dict) -> dict:
    from tinyml_studio.train.cnn1d import train_keras_cnn1d
    from tinyml_studio.train.report import cnn1d_summary

    res = train_keras_cnn1d(
        p.recipe,
        window=p.window,
        stride=p.stride,
        filters=tuple(p.filters),
        kernel=p.kernel,
        head=p.head,
        epochs=p.epochs,
        batch_size=p.batch_size,
        learning_rate=p.learning_rate,
        patience=p.patience,
        seed=p.seed,
        exclude=tuple(p.exclude),
        out_dir=ctx["out"],
        **_opt_kwargs(p, ctx),
    )
    return {
        "results": res,
        "summary_es": cnn1d_summary(res),
        "summary": cnn1d_summary(res),
        "family": "keras_cnn1d",
    }




SUBDIR_RE = re.compile(r"^[A-Za-z0-9_:\-]+$")
INLINE_EXT = (".ino", ".txt", ".md")


def _arduino_variants(out, variants: list[str], name: str = "tinyml_model") -> list[dict]:
    """Generate the Arduino sketch folders and return what the UI shows (steps, code, checks)."""
    from tinyml_studio.export.arduino import build_arduino_sketch

    res = []
    for v in variants:
        try:
            r = build_arduino_sketch(
                out, out / "arduino", name if len(variants) == 1 else f"{name}_{v}", variant=v
            )
        except Exception as exc:  # noqa: BLE001 - a sketch problem must not lose the export
            res.append({"variant": v, "error": str(exc)})
            continue
        sd = out / "arduino" / r["name"]
        r["code"] = {
            f: (sd / f).read_text(encoding="utf-8", errors="replace")
            for f in r["files"]
            if f.endswith(INLINE_EXT) and (sd / f).stat().st_size < 60_000
        }
        r["sketch_dir"] = f"arduino/{r['name']}"
        res.append(r)
    if res and all("error" in r for r in res):
        res = res[:1]  # same reason for every variant: show it once
    return res


def _classic_graph(ctx: dict, rep: dict, model: str | None):
    import joblib

    from tinyml_studio.export.graph import sklearn_graph

    try:
        run = ctx["run_dir"]
        res = json.loads((run / "results.json").read_text(encoding="utf-8"))
        name = model or res["selection"]["best"]
        return sklearn_graph(
            joblib.load(run / f"model_{name}.joblib"), res["features_used"], res["classes"]
        )
    except Exception as exc:  # noqa: BLE001 - the graph is optional
        return {"error": str(exc)}


def _use_run_subdir(p: ExportCParams, ctx: dict) -> None:
    """Apunta ``ctx['run_dir']`` a una subcarpeta de un benchmark (validada, sin escapar)."""
    sub = p.run_subdir
    if sub is None:
        return
    base = Path(ctx["run_dir"]).resolve()
    target = (base / sub).resolve()
    if not SUBDIR_RE.fullmatch(sub) or target.parent != base or not target.is_dir():
        raise FileNotFoundError(
            tr(
                "run_subdir '{s}' no existe en el trabajo de origen",
                "run_subdir '{s}' does not exist in the source job",
                s=sub,
            )
        )
    ctx["run_dir"] = target


def h_benchmark(p: BenchmarkParams, ctx: dict) -> dict:
    from tinyml_studio.train.bench import run_benchmark

    res = run_benchmark(
        p.recipe,
        models=p.models,
        keras=[k.model_dump(exclude_none=True) for k in p.keras] or None,
        cv_folds=p.cv_folds,
        seed=p.seed,
        selection=p.selection,
        max_flash_kb=p.max_flash_kb,
        out_dir=ctx["out"],
        n_boot=p.n_boot,
        exclude=tuple(p.exclude),
        params=p.params,
    )
    res["out_dir"] = None  # ruta interna del servidor: la UI usa run_job + run_subdir
    return {
        "results": res,
        "summary_es": res["summary"],
        "summary": res["summary"],
        "family": "benchmark",
    }


def h_export_c(p: ExportCParams, ctx: dict) -> dict:
    from tinyml_studio.export import export_run
    from tinyml_studio.export.report import summary

    _use_run_subdir(p, ctx)
    rep = export_run(
        ctx["run_dir"],
        ctx["out"],
        p.model,
        flash_kb=p.flash_kb,
        ram_kb=p.ram_kb,
        board=p.board,
        compact=p.compact,
    )
    folder = tr("la carpeta de salida", "the output folder")
    text = summary(rep, folder)
    return {
        "results": rep,
        "summary_es": text,
        "summary": text,
        "family": "c",
        "graph": _classic_graph(ctx, rep, p.model),
        "arduino": _arduino_variants(ctx["out"], ["raw_c"]),
    }


def h_predict(p: PredictParams, ctx: dict) -> dict:
    from tinyml_studio.export.predict import describe, predict_values

    _use_run_subdir(p, ctx)
    if p.values is None:
        return {"results": describe(ctx["run_dir"], p.model), "family": "predict_schema"}
    return {"results": predict_values(ctx["run_dir"], p.values, p.model), "family": "predict"}


def h_export_onnx(p: ExportOnnxParams, ctx: dict) -> dict:
    from tinyml_studio.export.onnx_export import export_onnx

    _use_run_subdir(p, ctx)  # type: ignore[arg-type]
    rep = export_onnx(ctx["run_dir"], ctx["out"], p.model, p.reject_below)
    v = rep["verification"]
    text = tr(
        "ONNX listo: {m}, {k} KB; coincide con scikit-learn en {a:.2%} de {r} filas "
        "(diferencia máxima de probabilidad {d}).",
        "ONNX ready: {m}, {k} KB; matches scikit-learn on {a:.2%} of {r} rows "
        "(max probability difference {d}).",
        m=rep["model"],
        k=round(rep["size_bytes"] / 1024, 1),
        a=v["label_match"],
        r=v["rows"],
        d=v["max_abs_proba_diff"],
    )
    return {"results": rep, "summary_es": text, "summary": text, "family": "onnx"}


def h_importance(p: ImportanceParams, ctx: dict) -> dict:
    from tinyml_studio.train.importance import permutation_importance

    _use_run_subdir(p, ctx)  # type: ignore[arg-type]
    res = permutation_importance(ctx["run_dir"], p.model, n_repeats=p.n_repeats)
    return {"results": res, "summary": res["summary"], "family": "importance"}


def h_latency(p: LatencyParams, ctx: dict) -> dict:
    from tinyml_studio.export.latency import estimate_latency

    _use_run_subdir(p, ctx)  # type: ignore[arg-type]
    res = estimate_latency(ctx["run_dir"], p.model, p.board, p.target_ms)
    return {"results": res, "summary": res["summary"], "family": "latency"}


def h_reject(p: RejectParams, ctx: dict) -> dict:
    from tinyml_studio.train.reject import calibrate_rejection

    _use_run_subdir(p, ctx)  # type: ignore[arg-type]
    res = calibrate_rejection(
        ctx["run_dir"], p.model, target_accuracy=p.target_accuracy, holdout=p.holdout
    )
    return {"results": res, "summary": res["summary"], "family": "rejection"}


def h_export_tflite(p: ExportTflParams, ctx: dict) -> dict:
    from tinyml_studio.export.report import tflite_summary
    from tinyml_studio.export.tflite import export_tflite

    _use_run_subdir(p, ctx)
    rep = export_tflite(
        ctx["run_dir"],
        ctx["out"],
        int8_io=p.int8_io,
        flash_kb=p.flash_kb,
        ram_kb=p.ram_kb,
        board=p.board,
    )
    text = tflite_summary(rep, tr("la carpeta de salida", "the output folder"))
    from tinyml_studio.export.graph import tflite_graph

    graph = None
    try:
        graph = tflite_graph((ctx["out"] / "model_int8.tflite").read_bytes())
    except Exception as exc:  # noqa: BLE001
        graph = {"error": str(exc)}
    variants = ["tflm"] if p.int8_io else ["eloquent", "tflm"]
    return {
        "results": rep,
        "summary_es": text,
        "summary": text,
        "family": "tflite",
        "graph": graph,
        "arduino": _arduino_variants(ctx["out"], variants),
    }


def h_sleep(p: SleepParams, ctx: dict) -> dict:  # solo para pruebas (enable_test_jobs)
    t0 = time.time()
    while time.time() - t0 < p.seconds:
        emit("sleep", 100 * (time.time() - t0) / p.seconds, tr("durmiendo", "sleeping"))
        time.sleep(0.2)
    return {"slept": p.seconds}


KINDS: dict[str, tuple[type[_P], Any]] = {
    "clean": (CleanParams, h_clean),
    "train": (TrainParams, h_train),
    "validate_groups": (GroupValidationParams, h_validate_groups),
    "tune": (TuneParams, h_tune),
    "anomaly": (AnomalyParams, h_anomaly),
    "importance": (ImportanceParams, h_importance),
    "latency": (LatencyParams, h_latency),
    "reject": (RejectParams, h_reject),
    "train_keras": (KerasParams, h_train_keras),
    "train_cnn1d": (CnnParams, h_train_cnn1d),
    "benchmark": (BenchmarkParams, h_benchmark),
    "predict": (PredictParams, h_predict),
    "export_c": (ExportCParams, h_export_c),
    "export_onnx": (ExportOnnxParams, h_export_onnx),
    "export_tflite": (ExportTflParams, h_export_tflite),
    "_sleep": (SleepParams, h_sleep),
}
NEEDS_RUN = {
    "export_c",
    "export_tflite",
    "export_onnx",
    "importance",
    "latency",
    "reject",
    "predict",
}
NEEDS_RECIPE = {k for k, (m, _) in KINDS.items() if "recipe" in m.model_fields}
