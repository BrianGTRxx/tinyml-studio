"""Callbacks comunes de Keras: parada temprana + progreso hacia la API."""

from __future__ import annotations

from tinyml_studio.progress import keras_callback


def fit_callbacks(patience: int, epochs: int) -> list:
    from .backend import keras_module

    cbs = [
        keras_module().callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]
    progress = keras_callback(epochs)
    if progress is not None:
        cbs.append(progress)
    return cbs


def history_dict(hist) -> dict[str, list[float]]:
    return {
        k: [float(v) for v in vals] for k, vals in hist.history.items() if k in ("loss", "val_loss")
    }
