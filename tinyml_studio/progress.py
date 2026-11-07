"""Progreso de trabajos largos. Fuera de un trabajo de la API no hace nada.

Dentro de un trabajo (variable ``TINYML_JOB=1``) escribe en stdout líneas
``@@EVENT {json}`` que el gestor de trabajos reenvía por SSE a la interfaz.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from tinyml_studio.i18n import tr

PREFIX = "@@EVENT "


def emit(stage: str, pct: float | None = None, msg: str | None = None, **extra: Any) -> None:
    if os.environ.get("TINYML_JOB") != "1":
        return
    ev = {"stage": stage, "pct": None if pct is None else round(float(pct), 1), "msg": msg, **extra}
    sys.stdout.write(PREFIX + json.dumps(ev, default=str, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def keras_callback(epochs: int):
    """Callback de Keras que informa cada época (pérdida de train/val). None si no hay trabajo."""
    if os.environ.get("TINYML_JOB") != "1":
        return None
    from tinyml_studio.train.backend import keras_module  # sin cargar TF si el backend es torch

    class _Progress(keras_module().callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._t = time.monotonic()
            self._durs: list[float] = []

        def on_epoch_begin(self, epoch, logs=None):
            self._t = time.monotonic()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self._durs.append(time.monotonic() - self._t)
            recent = self._durs[-5:]  # las primeras épocas pesan más (compilación): promedio móvil
            per_epoch = sum(recent) / len(recent)
            left = max(epochs - (epoch + 1), 0)
            emit(
                "epoch",
                100.0 * (epoch + 1) / max(epochs, 1),
                tr("época {e}/{n}", "epoch {e}/{n}", e=epoch + 1, n=epochs),
                epoch=epoch + 1,
                epochs=epochs,
                loss=float(logs.get("loss", 0.0)),
                val_loss=float(logs.get("val_loss", 0.0)),
                epoch_s=round(self._durs[-1], 3),
                eta_s=round(
                    per_epoch * left, 1
                ),  # cota superior: la parada temprana puede acortarla
            )

    return _Progress()
