"""1D-CNN (Keras) sobre ventanas de filas consecutivas, con exportación a TFLite Micro.

Necesita orden temporal: la receta debe usar partición ``temporal_blocks`` (el test es la parte
final de cada sesión) y las ventanas se forman dentro de cada split y de cada archivo/grupo.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio.clean import prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit
from tinyml_studio.window import WindowConfig, make_windows

from . import backend as B
from . import qual
from .callbacks import fit_callbacks, history_dict
from .features import FeatureEncoder
from .metrics import classification_metrics
from .run import _sha, _versions, feature_columns


@dataclass
class WinData:
    enc: FeatureEncoder
    cfg: WindowConfig
    X: dict[str, np.ndarray]  # (m, W, F)
    y: dict[str, np.ndarray]
    idx: dict[str, np.ndarray]  # posiciones de fila (en prep.clean[split]) de cada ventana
    stats: dict[str, dict[str, int]]
    rows_X: dict[str, np.ndarray]  # features por fila, para baselines
    prep: Any
    notes: list[str] = field(default_factory=list)


def prepare_windows(
    recipe: dict[str, Any], cfg: WindowConfig, seed: int, exclude: tuple[str, ...] = ()
) -> WinData:
    split = recipe.get("split", {})
    if split.get("strategy") != "temporal_blocks":
        raise ValueError(
            tr(
                "la 1D-CNN necesita partición 'temporal_blocks' (la receta usa "
                "'{s}'): Con filas barajadas no hay ventanas consecutivas. "
                "Cambia recipe['split'] o añade una columna de tiempo.",
                "the 1D-CNN needs a 'temporal_blocks' split (the recipe uses "
                "'{s}'): With shuffled rows there are no consecutive windows. "
                "Change recipe['split'] or add a time column.",
                s=split.get("strategy"),
            )
        )
    if recipe.get("task") not in ("classification",) and not recipe.get("label_from_file", {}).get(
        "enabled"
    ):
        raise ValueError(
            tr(
                "la 1D-CNN v1 solo soporta clasificación (una etiqueta por ventana)",
                "the 1D-CNN v1 only supports classification (one label per window)",
            )
        )
    prep = prepare_splits(recipe, seed=seed)
    target = prep.target
    feats = feature_columns(recipe, prep.clean["train"], exclude)
    enc = FeatureEncoder.fit(
        prep.clean["train"], feats, target, "classification", recipe.get("encoding")
    )
    gcol, tcol = split.get("group_column"), split.get("time_column")
    X, y, idx, stats, rows_X = {}, {}, {}, {}, {}
    for s in ("train", "val", "test"):
        df = prep.clean[s]
        rows = enc.transform(df)
        rows_X[s] = rows
        lab = enc.encode_target(df[target])
        group = df[gcol] if gcol else np.zeros(len(df), dtype=int)
        tm = df[tcol] if tcol else None
        X[s], y[s], idx[s], stats[s] = make_windows(
            rows, lab, group, tm, cfg, stride=None if s == "train" else cfg.window
        )
    notes = []
    if not tcol:
        notes.append(
            tr(
                "sin columna de tiempo: Se usa el orden de las filas dentro de cada grupo",
                "no time column: Row order within each group is used",
            )
        )
    return WinData(enc, cfg, X, y, idx, stats, rows_X, prep, notes)


def build_cnn1d(window: int, n_features: int, n_classes: int, filters, kernel: int, head: str):
    K = B.keras_module()

    length = window - len(filters) * (kernel - 1)
    if length < 1:
        raise ValueError(
            tr(
                "ventana {w} demasiado corta para {n} capas con kernel {k}",
                "window {w} too short for {n} layers with kernel {k}",
                w=window,
                n=len(filters),
                k=kernel,
            )
        )
    x = inp = K.layers.Input(shape=(window, n_features))
    for f in filters:
        x = K.layers.Conv1D(f, kernel, activation="relu")(x)  # padding 'valid'
    x = K.layers.GlobalAveragePooling1D()(x) if head == "gap" else K.layers.Flatten()(x)
    out = K.layers.Dense(n_classes, activation="softmax")(x)
    return K.Model(inp, out)


def _balanced(y: np.ndarray, n: int) -> dict[int, float]:
    c = np.bincount(y, minlength=n).astype(float)
    return {k: float(len(y) / (n * c[k])) for k in range(n) if c[k] > 0}


def train_keras_cnn1d(
    recipe: dict[str, Any],
    *,
    window: int = 16,
    stride: int = 4,
    filters: tuple[int, ...] = (16, 32),
    kernel: int = 3,
    head: str = "gap",
    epochs: int = 60,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    patience: int = 6,
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
    if head not in ("gap", "flatten"):
        raise ValueError(tr("head debe ser 'gap' o 'flatten'", "head must be 'gap' or 'flatten'"))
    plan = B.begin(backend, device, compute_notes)
    if plan["delegate"]:
        return B.delegate_training(
            "cnn1d",
            recipe,
            dict(
                window=window, stride=stride, filters=list(filters), kernel=kernel, head=head,
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                patience=patience, seed=seed, exclude=list(exclude), optimizer=optimizer,
                weight_decay=weight_decay, nesterov=nesterov, lr_schedule=lr_schedule,
                lr_final_frac=lr_final_frac, device=device, compute_notes=plan["notes"],
            ),
            out_dir,
        )  # fmt: skip
    K = B.keras_module()
    cfg = WindowConfig(window=window, stride=stride)
    d = prepare_windows(recipe, cfg, seed, exclude)
    classes = d.enc.classes
    n_classes = len(classes)
    emit("fit", 8, tr("Entrenando la 1D-CNN…", "Training the 1D-CNN…"))
    K.utils.set_random_seed(seed)
    model = build_cnn1d(window, d.X["train"].shape[2], n_classes, filters, kernel, head)
    model.compile(
        optimizer=B.make_optimizer(
            optimizer,
            learning_rate,
            weight_decay=weight_decay,
            nesterov=nesterov,
            lr_schedule=lr_schedule,
            lr_final_frac=lr_final_frac,
            steps_per_epoch=-(-len(d.X["train"]) // batch_size),
            epochs=epochs,
        ),
        loss="sparse_categorical_crossentropy",
    )
    t0 = time.perf_counter()
    hist = model.fit(
        d.X["train"],
        d.y["train"],
        validation_data=(d.X["val"], d.y["val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=_balanced(d.y["train"], n_classes),
        verbose=0,
        callbacks=fit_callbacks(patience, epochs),
    )
    fit_s = time.perf_counter() - t0

    def metr(split):
        return classification_metrics(
            d.y[split], model.predict(d.X[split], verbose=0).argmax(1), classes
        )

    # referencias: clase mayoritaria y un random forest con SOLO la última fila de cada ventana
    major = int(np.bincount(d.y["train"]).argmax())
    base = {
        s: classification_metrics(d.y[s], np.full(len(d.y[s]), major), classes)
        for s in ("val", "test")
    }
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
    )
    rf.fit(d.X["train"][:, -1, :], d.y["train"])
    rf_last = {
        s: classification_metrics(d.y[s], rf.predict(d.X[s][:, -1, :]), classes)
        for s in ("val", "test")
    }
    warnings = [
        tr(
            "el test son las ventanas finales de cada sesión (bloques temporales): Comparte "
            "sesión con el train y NO mide la generalización a sesiones, hogares o aparatos "
            "nuevos",
            "the test set is the final windows of each session (temporal blocks): It shares "
            "sessions with train and does NOT measure generalization to new sessions, "
            "households or appliances",
        ),
        tr(
            "ventanas de entrenamiento solapadas (stride < window): Son casi iguales entre sí; "
            "val/test usan ventanas sin solape",
            "overlapping training windows (stride < window): They are nearly identical to each "
            "other; val/test use non-overlapping windows",
        ),
        *d.notes,
    ]
    small = [c for c, v in metr("test")["per_class"].items() if 0 < v["support"] < 20]
    if small:
        warnings.append(
            tr(
                "clases con < 20 ventanas en test (métricas poco fiables): {small}",
                "classes with < 20 windows in test (unreliable metrics): {small}",
                small=", ".join(map(str, small)),
            )
        )
    if metr("test")["f1_macro"] <= rf_last["test"]["f1_macro"] + 0.005:
        warnings.append(
            tr(
                "la 1D-CNN no supera a un random forest con solo la última fila de cada "
                "ventana: El contexto temporal no aporta aquí",
                "the 1D-CNN does not beat a random forest using only the last row of each "
                "window: Temporal context adds nothing here",
            )
        )
    result = {
        "kind": "keras_cnn1d",
        "task": "classification",
        "target": d.prep.target,
        "classes": classes,
        "features_used": d.enc.feature_names,
        "features_excluded_by_user": list(exclude),
        "window": cfg.to_dict(),
        "window_stats": d.stats,
        "windows": {s: int(len(d.y[s])) for s in d.y},
        "architecture": {"filters": list(filters), "kernel": kernel, "head": head},
        "hyperparameters": {
            "epochs_max": epochs,
            "epochs_run": len(hist.history["loss"]),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "class_weight": "balanced",
            **B.optimizer_record(optimizer, weight_decay, nesterov, lr_schedule, lr_final_frac),
        },
        "parameters": int(model.count_params()),
        "fit_seconds": round(fit_s, 1),
        "history": history_dict(hist),
        "quality": qual.classification_block(
            d.y, {s: model.predict(d.X[s], verbose=0) for s in d.y}, classes
        ),
        "baseline": {
            s: {k: v for k, v in m.items() if k in ("accuracy", "f1_macro")}
            for s, m in base.items()
        },
        "reference_rf_last_row": {
            s: {k: v for k, v in m.items() if k in ("accuracy", "f1_macro")}
            for s, m in rf_last.items()
        },
        "val": metr("val"),
        "test": metr("test"),
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
        "warnings": warnings,
    }
    transfer = None
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save(out / "model.keras")
        if plan["backend"] == "torch":
            arch = dict(
                kind="cnn1d", window=window, n_features=int(d.X["train"].shape[2]),
                n_out=n_classes, filters=list(filters), kernel=kernel, head=head,
            )  # fmt: skip
            B.save_transfer(out, arch, model, d.X["test"])
            transfer = B.finalize_transfer(out)
    result["compute"] = B.compute_record(plan, transfer)
    result["warnings"] = [*result["warnings"], *plan["notes"]]
    if out_dir is not None:
        (out / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "recipe_fitted.json").write_text(
            json.dumps(
                {**recipe, "cleaning": d.prep.pipeline.to_list(), "cnn1d": {**cfg.to_dict()}},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (out / "encoder.json").write_text(
            json.dumps(d.enc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result
