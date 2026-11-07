"""Catálogo de modelos que ofrece la interfaz (qué se puede entrenar y con qué opciones)."""

from __future__ import annotations

from tinyml_studio.environment import environment
from tinyml_studio.i18n import tr
from tinyml_studio.train import models as M


def optimizers_info() -> list[dict]:
    """Optimizadores Keras con una nota corta (solo cambian el entrenamiento, no el .tflite)."""
    return [
        {
            "id": "adam",
            "default": True,
            "note": tr(
                "Por defecto; robusto en casi todo sin ajustar casi nada.",
                "Default; robust on almost everything with little tuning.",
            ),
        },
        {
            "id": "adamw",
            "params": ["weight_decay"],
            "note": tr(
                "Adam con decaimiento de pesos desacoplado; regulariza mejor que L2 con Adam.",
                "Adam with decoupled weight decay; regularizes better than L2 with Adam.",
            ),
        },
        {
            "id": "nadam",
            "note": tr(
                "Adam con momento de Nesterov; a veces converge algo antes.",
                "Adam with Nesterov momentum; sometimes converges a bit sooner.",
            ),
        },
        {
            "id": "rmsprop",
            "note": tr(
                "Tasa adaptativa por parámetro; útil en señales ruidosas o series.",
                "Per-parameter adaptive rate; useful on noisy signals or series.",
            ),
        },
        {
            "id": "sgd",
            "params": ["nesterov"],
            "note": tr(
                "Descenso por gradiente con momento 0.9; pide ajustar la tasa y más épocas.",
                "Gradient descent with momentum 0.9; needs rate tuning and more epochs.",
            ),
        },
        {
            "id": "adagrad",
            "note": tr(
                "Tasa que baja sola con el uso; pensado para datos dispersos.",
                "Self-decaying rate; meant for sparse data.",
            ),
        },
    ]


def lr_schedules_info() -> list[dict]:
    return [
        {
            "id": "constant",
            "default": True,
            "note": tr("Tasa fija (comportamiento actual).", "Fixed rate (current behavior)."),
        },
        {
            "id": "cosine",
            "params": ["lr_final_frac"],
            "note": tr(
                "Baja suave hasta tasa*lr_final_frac al final de las épocas máximas.",
                "Decays smoothly to rate*lr_final_frac at the end of the max epochs.",
            ),
        },
        {
            "id": "exp_decay",
            "params": ["lr_final_frac"],
            "note": tr(
                "Baja un factor constante por época hasta tasa*lr_final_frac.",
                "Decays by a constant factor per epoch down to rate*lr_final_frac.",
            ),
        },
    ]


def backends_info() -> dict:
    """Backends de Keras y valores de 'device'; todo sin verificar en GPU real (ver docs)."""
    return {
        "backends": [
            {
                "id": "tensorflow",
                "default": True,
                "note": tr(
                    "Comportamiento actual. GPU: NVIDIA (CUDA) en Linux/WSL2; AMD solo con "
                    "TensorFlow-ROCm en Linux.",
                    "Current behavior. GPU: NVIDIA (CUDA) on Linux/WSL2; AMD only with "
                    "TensorFlow-ROCm on Linux.",
                ),
            },
            {
                "id": "torch",
                "note": tr(
                    "Keras 3 sobre PyTorch (entrena con CUDA o ROCm). Los pesos se copian "
                    "después a un modelo TensorFlow y se verifican (<1e-4) para exportar TFLite.",
                    "Keras 3 on PyTorch (trains with CUDA or ROCm). Weights are then copied to "
                    "a TensorFlow model and verified (<1e-4) so TFLite export keeps working.",
                ),
            },
        ],
        "devices": [
            {"id": "auto", "default": True},
            {"id": "cpu"},
            {
                "id": "gpu",
                "note": tr(
                    "Usa la GPU si algún framework la ve; si no, CPU con aviso (no falla).",
                    "Uses the GPU if any framework sees one; otherwise CPU with a warning "
                    "(never fails).",
                ),
            },
        ],
        "tested": tr("solo CPU", "CPU only"),
    }


def get_catalog() -> dict:
    """Catálogo en el idioma activo."""
    return {
        "environment": environment(),
        "classical": {
            "label": tr("Modelos clásicos (scikit-learn)", "Classical models (scikit-learn)"),
            "job": "train",
            "tasks": ["classification", "regression"],
            "models": {t: list(M.available(t)) for t in ("classification", "regression")},
            "defaults": M.DEFAULTS,
            "mlp_solvers": list(M.MLP_SOLVERS),
            "meta": M.meta(),
            "export": "c",
            "notes": tr(
                "Exportables a C autónomo para microcontroladores (verificación Python↔C).",
                "Exportable to standalone C for microcontrollers (Python↔C verification).",
            ),
        },
        "keras_mlp": {
            "label": "MLP (Keras)",
            "job": "train_keras",
            "tasks": ["classification", "regression"],
            "defaults": {
                "hidden": [32, 16],
                "epochs": 100,
                "batch_size": 128,
                "learning_rate": 0.001,
                "patience": 8,
                "activation": "relu",
                "dropout": 0.0,
                "l2": 0.0,
                "optimizer": "adam",
                "lr_schedule": "constant",
                "weight_decay": 0.004,
                "nesterov": False,
                "lr_final_frac": 0.01,
                "backend": "tensorflow",
                "device": "auto",
            },
            "activations": ["relu", "tanh", "sigmoid"],
            "optimizers": optimizers_info(),
            "lr_schedules": lr_schedules_info(),
            "compute": backends_info(),
            "presets": [
                {
                    "id": "small",
                    "name": tr("Pequeña", "Small"),
                    "params": {"hidden": [16, 8], "dropout": 0.0, "activation": "relu"},
                },
                {
                    "id": "medium",
                    "name": tr("Mediana", "Medium"),
                    "params": {"hidden": [64, 32], "dropout": 0.0, "activation": "relu"},
                },
                {
                    "id": "deep",
                    "name": tr("Profunda", "Deep"),
                    "params": {"hidden": [128, 64, 32], "dropout": 0.1, "activation": "relu"},
                },
            ],
            "export": "tflite",
            "notes": tr(
                "Cuantización int8 y TFLite Micro; verificación Keras↔float↔int8↔TFLM.",
                "int8 quantization and TFLite Micro; Keras↔float↔int8↔TFLM verification.",
            ),
        },
        "keras_cnn1d": {
            "label": tr("1D-CNN (Keras) sobre ventanas", "1D-CNN (Keras) on windows"),
            "job": "train_cnn1d",
            "tasks": ["classification"],
            "defaults": {
                "window": 16,
                "stride": 4,
                "filters": [16, 32],
                "kernel": 3,
                "head": "gap",
                "epochs": 60,
                "batch_size": 128,
                "learning_rate": 0.001,
                "patience": 6,
                "optimizer": "adam",
                "lr_schedule": "constant",
                "weight_decay": 0.004,
                "nesterov": False,
                "lr_final_frac": 0.01,
                "backend": "tensorflow",
                "device": "auto",
            },
            "optimizers": optimizers_info(),
            "lr_schedules": lr_schedules_info(),
            "compute": backends_info(),
            "requires": tr(
                "partición temporal_blocks y columna de tiempo",
                "temporal_blocks split and a time column",
            ),
            "export": "tflite",
            "notes": tr(
                "Compara siempre contra un random forest con la última fila.",
                "Always compared against a random forest using the last row.",
            ),
        },
    }
