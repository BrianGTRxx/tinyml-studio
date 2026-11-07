"""MLP en Keras (ADR-001) sobre los mismos splits y el mismo codificador que los modelos clásicos.

TensorFlow se importa de forma perezosa (3 s de import): solo al entrenar/exportar.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tinyml_studio.clean import prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit

from . import backend as B
from . import qual
from .callbacks import fit_callbacks, history_dict
from .features import FeatureEncoder
from .metrics import classification_metrics, regression_metrics
from .run import _sha, _versions, _warnings, feature_columns

ACTIVATIONS = ("relu", "tanh", "sigmoid")


def hidden_layers(
    hidden: tuple[int, ...], activation: str = "relu", dropout: float = 0.0, l2: float = 0.0
) -> list:
    """Capas ocultas Dense (+ Dropout opcional, que no existe en inferencia ni en el .tflite)."""
    K = B.keras_module()

    if activation not in ACTIVATIONS:
        raise ValueError(
            tr(
                "activación '{a}' no soportada: {ok}",
                "activation '{a}' not supported: {ok}",
                a=activation,
                ok=ACTIVATIONS,
            )
        )
    if not 0.0 <= dropout <= 0.5:
        raise ValueError(tr("dropout debe estar en [0, 0.5]", "dropout must be in [0, 0.5]"))
    reg = K.regularizers.l2(l2) if l2 > 0 else None
    out: list = []
    for h in hidden:
        out.append(K.layers.Dense(h, activation=activation, kernel_regularizer=reg))
        if dropout > 0:
            out.append(K.layers.Dropout(dropout))
    return out


def build_mlp(
    n_features: int,
    n_out: int,
    task: str,
    hidden: tuple[int, ...],
    activation: str = "relu",
    dropout: float = 0.0,
    l2: float = 0.0,
):
    K = B.keras_module()

    layers = [K.layers.Input(shape=(n_features,))]
    layers += hidden_layers(hidden, activation, dropout, l2)
    if task == "classification":
        layers.append(K.layers.Dense(n_out, activation="softmax"))
    else:
        layers.append(K.layers.Dense(1))
    return K.Sequential(layers)




def _balanced_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return {k: float(len(y) / (n_classes * counts[k])) for k in range(n_classes) if counts[k] > 0}


def train_keras_mlp(
    recipe: dict[str, Any],
    *,
    hidden: tuple[int, ...] = (32, 16),
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    patience: int = 8,
    activation: str = "relu",
    dropout: float = 0.0,
    l2: float = 0.0,
    seed: int = 42,
    exclude: tuple[str, ...] = (),
    out_dir: str | Path | None = None,
    optimizer: str = "adam",
    weight_decay: float = 0.004,
    nesterov: bool = False,
    lr_schedule: str = "constant",
    lr_final_frac: float = 0.01,
    backend: str = "tensorflow",
    device: str = "auto",
    compute_notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    plan = B.begin(backend, device, compute_notes)
    if plan["delegate"]:
        return B.delegate_training(
            "mlp",
            recipe,
            dict(
                hidden=list(hidden), epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, patience=patience, activation=activation,
                dropout=dropout, l2=l2, seed=seed, exclude=list(exclude), optimizer=optimizer,
                weight_decay=weight_decay, nesterov=nesterov, lr_schedule=lr_schedule,
                lr_final_frac=lr_final_frac, device=device, compute_notes=plan["notes"],
            ),
            out_dir,
        )  # fmt: skip
    K = B.keras_module()

    task = recipe["task"]
    if task not in ("classification", "regression"):
        raise ValueError(
            tr(
                "tarea no entrenable con un MLP supervisado: {task}",
                "task cannot be trained with a supervised MLP: {task}",
                task=task,
            )
        )
    prep = prepare_splits(recipe, seed=seed)
    target = prep.target
    if target is None:
        raise ValueError(
            tr(
                "la receta no tiene columna objetivo (ni label_from_file)",
                "the recipe has no target column (nor label_from_file)",
            )
        )
    feats = feature_columns(recipe, prep.clean["train"], exclude)
    enc = FeatureEncoder.fit(prep.clean["train"], feats, target, task, recipe.get("encoding"))
    X = {s: enc.transform(df) for s, df in prep.clean.items()}
    if task == "classification":
        y = {s: enc.encode_target(df[target]) for s, df in prep.clean.items()}
    else:
        y = {s: pd.to_numeric(df[target]).to_numpy(dtype="float32") for s, df in prep.clean.items()}
    n_classes = len(enc.classes)

    emit("fit", 8, tr("Entrenando la red…", "Training the network…"))
    K.utils.set_random_seed(seed)
    model = build_mlp(X["train"].shape[1], n_classes, task, hidden, activation, dropout, l2)
    model.compile(
        optimizer=B.make_optimizer(
            optimizer,
            learning_rate,
            weight_decay=weight_decay,
            nesterov=nesterov,
            lr_schedule=lr_schedule,
            lr_final_frac=lr_final_frac,
            steps_per_epoch=-(-len(X["train"]) // batch_size),
            epochs=epochs,
        ),
        loss="sparse_categorical_crossentropy" if task == "classification" else "mse",
    )
    cw = _balanced_weights(y["train"], n_classes) if task == "classification" else None
    # regresión: se entrena con el objetivo estandarizado y luego se "pliega" la transformación
    # en la última capa; así el modelo exportado devuelve unidades originales sin postproceso
    y_mu, y_sd = 0.0, 1.0
    if task == "regression":
        y_mu, y_sd = float(y["train"].mean()), float(y["train"].std()) or 1.0
    fit_y = {
        s: ((v - y_mu) / y_sd).astype("float32") if task == "regression" else v
        for s, v in y.items()
    }
    t0 = time.perf_counter()
    hist = model.fit(
        X["train"],
        fit_y["train"],
        validation_data=(X["val"], fit_y["val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        verbose=0,
        callbacks=fit_callbacks(patience, epochs),
    )
    fit_s = time.perf_counter() - t0
    if task == "regression":
        w, b = model.layers[-1].get_weights()
        model.layers[-1].set_weights([w * y_sd, b * y_sd + y_mu])

    def predict(split: str) -> np.ndarray:
        return model.predict(X[split], verbose=0)

    out_metrics = {}
    for s in ("val", "test"):
        p = predict(s)
        out_metrics[s] = (
            classification_metrics(y[s], p.argmax(1), enc.classes)
            if task == "classification"
            else regression_metrics(y[s], p[:, 0])
        )
    key = "f1_macro" if task == "classification" else "r2"
    base = np.bincount(y["train"]).argmax() if task == "classification" else y["train"].mean()
    baseline = {}
    for s in ("val", "test"):
        pred = np.full(len(y[s]), base)
        m = (
            classification_metrics(y[s], pred.astype(int), enc.classes)
            if task == "classification"
            else regression_metrics(y[s], pred)
        )
        baseline[s] = {k: m[k] for k in ("accuracy", "f1_macro", "r2", "mae") if k in m}
    result = {
        "kind": "keras_mlp",
        "task": task,
        "target": target,
        "classes": enc.classes,
        "features_used": enc.feature_names,
        "features_skipped": enc.skipped,
        "features_excluded_by_user": list(exclude),
        "rows": {s: int(len(y[s])) for s in y},
        "hyperparameters": {
            "hidden": list(hidden),
            "epochs_max": epochs,
            "epochs_run": len(hist.history["loss"]),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "activation": activation,
            "dropout": dropout,
            "l2": l2,
            "class_weight": "balanced" if cw else None,
            **B.optimizer_record(optimizer, weight_decay, nesterov, lr_schedule, lr_final_frac),
        },
        "parameters": int(model.count_params()),
        "fit_seconds": round(fit_s, 1),
        "history": history_dict(hist),
        "quality": (
            qual.classification_block(y, {s: predict(s) for s in y}, enc.classes)
            if task == "classification"
            else qual.regression_block(y, {s: predict(s)[:, 0] for s in y})
        ),
        "baseline": baseline,
        "val": out_metrics["val"],
        "test": out_metrics["test"],
        "selection": {
            "metric": key,
            "split": "val",
            "note": tr("early stopping con val_loss", "early stopping on val_loss"),
        },
        "reproducibility": {
            "seed": seed,
            "python": platform.python_version(),
            "libraries": {
                **_versions(),
                **B.framework_versions(),
            },
            "recipe_sha256": hashlib.sha256(
                json.dumps(recipe, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "data_sha256": {Path(f).name: _sha(f) for f in recipe["files"]},
        },
        "warnings": _warnings(
            {"keras_mlp": {"val": out_metrics["val"], "test": out_metrics["test"]}},
            baseline,
            task,
            key,
            recipe.get("split"),
        ),
    }
    transfer = None
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save(out / "model.keras")
        if plan["backend"] == "torch":
            arch = dict(
                kind="mlp", n_features=int(X["train"].shape[1]), n_out=n_classes, task=task,
                hidden=list(hidden), activation=activation, dropout=dropout, l2=l2,
            )  # fmt: skip
            B.save_transfer(out, arch, model, X["test"])
            transfer = B.finalize_transfer(out)
    result["compute"] = B.compute_record(plan, transfer)
    result["warnings"] = [*result["warnings"], *plan["notes"]]
    if out_dir is not None:
        (out / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "recipe_fitted.json").write_text(
            json.dumps(
                {**recipe, "cleaning": prep.pipeline.to_list()}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        (out / "encoder.json").write_text(
            json.dumps(enc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result

