"""Backend de Keras (TensorFlow | PyTorch), dispositivo y optimizadores de los entrenadores Keras.

Solo el entrenamiento puede usar el backend ``torch`` (útil para GPU AMD con ROCm, que
TensorFlow solo ofrece en Linux). La exportación a TFLite necesita TensorFlow, así que después
de entrenar en torch los pesos se copian (numpy) a un modelo equivalente construido con el
mismo constructor en un proceso con ``KERAS_BACKEND=tensorflow`` y se comprueba que las
predicciones coinciden (|diff| < 1e-4) antes de reemplazar ``model.keras``.

Nada de esto se ha probado con una GPU real: solo CPU (ver docs/GPU_AMD.md).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio.i18n import tr
from tinyml_studio.pyexec import python_argv

BACKENDS = ("tensorflow", "torch")
DEVICES = ("auto", "cpu", "gpu")
OPTIMIZERS = ("adam", "adamw", "nadam", "rmsprop", "sgd", "adagrad")
SCHEDULES = ("constant", "cosine", "exp_decay")
TRANSFER_TOL = 1e-4
CHECK_ROWS = 2048
SUBPROCESS_TIMEOUT = 3600


# ----------------------------------------------------------------------------- keras / backend


def active_backend() -> str | None:
    """Backend de Keras ya cargado en este proceso (None si Keras aún no se importó)."""
    k = sys.modules.get("keras")
    if k is None:
        return None
    try:
        return str(k.backend.backend())
    except Exception:  # noqa: BLE001
        return None


def keras_module():
    """``tf.keras`` (comportamiento actual) o ``keras`` a secas si el backend es torch.

    Con ``KERAS_BACKEND=torch`` no se importa TensorFlow: tarda y podría reservar la GPU.
    """
    if os.environ.get("KERAS_BACKEND", "").lower() == "torch":
        import keras

        if int(str(keras.__version__).split(".")[0]) < 3:
            raise RuntimeError(
                tr(
                    "el backend torch necesita Keras 3 (instalado: {v})",
                    "the torch backend needs Keras 3 (installed: {v})",
                    v=keras.__version__,
                )
            )
        return keras
    import tensorflow as tf

    return tf.keras


def framework_versions() -> dict[str, str]:
    k = keras_module()
    out = {"keras": str(k.__version__)}
    if active_backend() == "torch" or os.environ.get("KERAS_BACKEND", "").lower() == "torch":
        import torch

        out["torch"] = torch.__version__
    else:
        import tensorflow as tf

        out["tensorflow"] = tf.__version__
    return out


def check_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(
            tr(
                "backend '{b}' no soportado: {ok}",
                "backend '{b}' not supported: {ok}",
                b=backend,
                ok=BACKENDS,
            )
        )
    return backend


def torch_active() -> bool:
    return os.environ.get("KERAS_BACKEND", "").lower() == "torch" and active_backend() in (
        None,
        "torch",
    )


# ----------------------------------------------------------------------------- planificación


def plan_compute(backend: str, device: str) -> dict[str, Any]:
    """Decide backend y dispositivo; NUNCA falla por falta de GPU (avisa y usa CPU).

    ``device='gpu'`` = usar la GPU si algún framework de este entorno la ve; si el backend
    pedido no la ve pero el otro sí, se cambia de backend (con aviso).
    """
    check_backend(backend)
    if device not in DEVICES:
        raise ValueError(
            tr(
                "device '{d}' no soportado: {ok}",
                "device '{d}' not supported: {ok}",
                d=device,
                ok=DEVICES,
            )
        )
    plan: dict[str, Any] = {
        "backend": backend,
        "backend_requested": backend,
        "device_requested": device,
        "device": "cpu" if device == "cpu" else "auto",
        "gpu_seen_by": [],
        "notes": [],
    }
    if device != "gpu":
        return plan
    from tinyml_studio.hw.detect import probe_framework

    order = [backend] + [b for b in BACKENDS if b != backend]
    seen: dict[str, bool] = {}
    for fw in order:
        info = probe_framework(fw)
        seen[fw] = bool(info.get("gpu_count"))
        if seen[fw]:
            break
    plan["gpu_seen_by"] = [fw for fw, ok in seen.items() if ok]
    if not plan["gpu_seen_by"]:
        plan["device"] = "cpu"
        plan["notes"].append(
            tr(
                "Se pidió GPU pero ningún framework de este entorno ve una (probado: {fw}): Se "
                "entrena en CPU. Revisa la guía de instalación en docs/GPU_AMD.md.",
                "A GPU was requested but no framework in this environment sees one (probed: "
                "{fw}): Training on the CPU. See the install guide in docs/GPU_AMD.md.",
                fw=", ".join(seen),
            )
        )
    elif backend not in plan["gpu_seen_by"]:
        plan["backend"] = plan["gpu_seen_by"][0]
        plan["device"] = "gpu"
        plan["notes"].append(
            tr(
                "El backend '{b}' no ve ninguna GPU pero '{o}' sí: Se entrena con '{o}'.",
                "The '{b}' backend sees no GPU but '{o}' does: Training with '{o}'.",
                b=backend,
                o=plan["backend"],
            )
        )
    else:
        plan["device"] = "gpu"
    return plan


def begin(backend: str, device: str, notes: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """Plan de cómputo de un entrenamiento. ``plan['delegate']``: hay que correr en un
    subproceso torch porque este proceso aún no tiene (ni puede cambiar a) ese backend."""
    check_backend(backend)
    if backend == "torch" and torch_active():
        plan = {
            "backend": "torch",
            "backend_requested": backend,
            "device_requested": device,
            "device": "cpu" if device == "cpu" else "auto",
            "gpu_seen_by": [],
            "notes": [],
        }
    else:
        plan = plan_compute(backend, device)
    plan["notes"] = list(dict.fromkeys([*notes, *plan["notes"]]))
    plan["delegate"] = plan["backend"] == "torch" and not torch_active()
    return plan


def runtime_device() -> str:
    """Dispositivo que ESTE proceso puede usar de verdad ('gpu' si su framework ve una)."""
    try:
        if active_backend() == "torch":
            import torch

            return "gpu" if torch.cuda.is_available() else "cpu"
        import tensorflow as tf

        return "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def compute_record(plan: dict[str, Any], transfer: dict | None = None) -> dict[str, Any]:
    rec = {
        "backend": plan["backend"],
        "backend_requested": plan["backend_requested"],
        "device_requested": plan["device_requested"],
        "device_seen_by_framework": runtime_device(),
        "frameworks": framework_versions(),
        "torch_to_tf_transfer": transfer,
    }
    if plan["notes"]:
        rec["notes"] = list(plan["notes"])
    return rec


# ----------------------------------------------------------------------------- optimizadores


def make_optimizer(
    name: str = "adam",
    learning_rate: float = 1e-3,
    *,
    weight_decay: float = 0.004,
    nesterov: bool = False,
    lr_schedule: str = "constant",
    lr_final_frac: float = 0.01,
    steps_per_epoch: int = 1,
    epochs: int = 1,
):
    """Optimizador de Keras. ``adam`` + ``constant`` == el ``Adam(learning_rate)`` de siempre.

    ``lr_final_frac``: fracción de la tasa inicial a la que llega al final de ``epochs``
    (coseno: ``alpha``; exponencial: ``rate = frac**(1/epochs)`` por época).
    """
    if name not in OPTIMIZERS:
        raise ValueError(
            tr(
                "optimizador '{o}' no soportado: {ok}",
                "optimizer '{o}' not supported: {ok}",
                o=name,
                ok=OPTIMIZERS,
            )
        )
    if lr_schedule not in SCHEDULES:
        raise ValueError(
            tr(
                "lr_schedule '{s}' no soportado: {ok}",
                "lr_schedule '{s}' not supported: {ok}",
                s=lr_schedule,
                ok=SCHEDULES,
            )
        )
    if not 0 < lr_final_frac <= 1:
        raise ValueError(
            tr("lr_final_frac debe estar en (0, 1]", "lr_final_frac must be in (0, 1]")
        )
    K = keras_module()
    lr: Any = learning_rate
    total = max(1, int(steps_per_epoch) * int(epochs))
    if lr_schedule == "cosine":
        lr = K.optimizers.schedules.CosineDecay(learning_rate, total, alpha=lr_final_frac)
    elif lr_schedule == "exp_decay":
        lr = K.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=max(1, int(steps_per_epoch)),
            decay_rate=float(lr_final_frac) ** (1.0 / max(1, int(epochs))),
        )
    opt = K.optimizers
    if name == "adam":
        return opt.Adam(lr)
    if name == "adamw":
        return opt.AdamW(lr, weight_decay=weight_decay)
    if name == "nadam":
        return opt.Nadam(lr)
    if name == "rmsprop":
        return opt.RMSprop(lr)
    if name == "sgd":
        return opt.SGD(lr, momentum=0.9, nesterov=nesterov)
    return opt.Adagrad(lr)


def optimizer_record(
    name: str, weight_decay: float, nesterov: bool, lr_schedule: str, lr_final_frac: float
) -> dict[str, Any]:
    """Lo que se anota en ``hyperparameters`` (solo lo que aplica a ese optimizador)."""
    rec: dict[str, Any] = {"optimizer": name, "lr_schedule": lr_schedule}
    if name == "adamw":
        rec["weight_decay"] = weight_decay
    if name == "sgd":
        rec["momentum"] = 0.9
        rec["nesterov"] = nesterov
    if lr_schedule != "constant":
        rec["lr_final_frac"] = lr_final_frac
    return rec


# ----------------------------------------------------------------------------- subprocesos


def _env(backend: str, device: str) -> dict[str, str]:
    env = dict(os.environ)
    env["KERAS_BACKEND"] = backend
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["HIP_VISIBLE_DEVICES"] = "-1"
    return env


def _run_module(args: list[str], backend: str, device: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        python_argv(module="tinyml_studio.train.backend", args=args),
        capture_output=True,
        text=True,
        env=_env(backend, device),
        timeout=SUBPROCESS_TIMEOUT,
    )


def _fail(what: str, p: subprocess.CompletedProcess) -> RuntimeError:
    tail = (p.stderr or p.stdout or "")[-1500:]
    return RuntimeError(f"{what}: {tail}")


def delegate_training(
    kind: str, recipe: dict, kwargs: dict, out_dir: str | Path | None, mix: dict | None = None
) -> dict[str, Any]:
    """Entrena en un subproceso con KERAS_BACKEND=torch (este proceso ya tiene otro backend)."""
    with tempfile.TemporaryDirectory(prefix="tinyml_bk_") as tmp:
        spec = Path(tmp) / "spec.json"
        res = Path(tmp) / "result.json"
        target = Path(out_dir) if out_dir is not None else Path(tmp) / "out"
        spec.write_text(
            json.dumps(
                {
                    "kind": kind,
                    "recipe": recipe,
                    "kwargs": kwargs,
                    "out_dir": str(target),
                    "mix": mix,
                }
            ),
            encoding="utf-8",
        )
        p = _run_module(["train", str(spec), str(res)], "torch", kwargs.get("device", "auto"))
        if p.returncode != 0:
            raise _fail(tr("falló el entrenamiento con torch", "torch training failed"), p)
        return json.loads(res.read_text(encoding="utf-8"))


def save_transfer(out: Path, arch: dict, model, X_check: np.ndarray) -> None:
    """Guarda lo necesario para reconstruir el modelo en TF: arquitectura, pesos y una muestra
    del split de test con las predicciones del modelo entrenado (para comparar)."""
    X = np.asarray(X_check, dtype="float32")[:CHECK_ROWS]
    pred = np.asarray(model.predict(X, verbose=0), dtype="float32")
    (out / "transfer_arch.json").write_text(json.dumps(arch), encoding="utf-8")
    np.savez(
        out / "transfer_weights.npz",
        **{f"w{i}": np.asarray(w) for i, w in enumerate(model.get_weights())},
    )
    np.savez(out / "transfer_check.npz", X=X, pred=pred)


def finalize_transfer(out: Path) -> dict[str, Any]:
    """Convierte el modelo entrenado en torch en uno TF (subproceso) y verifica los números."""
    p = _run_module(["transfer", str(out)], "tensorflow", "cpu")
    if p.returncode != 0:
        raise _fail(
            tr(
                "no se pudo pasar el modelo torch a TensorFlow (la exportación TFLite lo necesita)",
                "could not transfer the torch model to TensorFlow (TFLite export needs it)",
            ),
            p,
        )
    rep = json.loads((out / "transfer_report.json").read_text(encoding="utf-8"))
    for f in (
        "transfer_arch.json",
        "transfer_weights.npz",
        "transfer_check.npz",
        "transfer_report.json",
    ):
        (out / f).unlink(missing_ok=True)
    return rep


def _build(arch: dict):
    if arch["kind"] == "mlp":
        from . import keras_mlp as M

        return M.build_mlp(
            arch["n_features"],
            arch["n_out"],
            arch["task"],
            tuple(arch["hidden"]),
            arch["activation"],
            arch["dropout"],
            arch["l2"],
        )
    if arch["kind"] == "cnn1d":
        from .cnn1d import build_cnn1d

        return build_cnn1d(
            arch["window"],
            arch["n_features"],
            arch["n_out"],
            tuple(arch["filters"]),
            arch["kernel"],
            arch["head"],
        )
    raise ValueError(
        tr(
            "tipo de modelo sin transferencia torch→TF: {k}",
            "model type without torch→TF transfer: {k}",
            k=arch["kind"],
        )
    )


def _transfer(out: Path) -> dict[str, Any]:
    arch = json.loads((out / "transfer_arch.json").read_text(encoding="utf-8"))
    w = np.load(out / "transfer_weights.npz")
    chk = np.load(out / "transfer_check.npz")
    model = _build(arch)
    weights = [w[f"w{i}"] for i in range(len(w.files))]
    model.set_weights(weights)  # falla si las formas no coinciden
    pred = np.asarray(model.predict(chk["X"], verbose=0), dtype="float32")
    diff = float(np.max(np.abs(pred - chk["pred"]))) if pred.size else 0.0
    if not diff < TRANSFER_TOL:
        raise RuntimeError(
            tr(
                "la copia TF del modelo torch difiere ({d:.2e} >= {t}); no se reemplaza "
                "model.keras",
                "the TF copy of the torch model differs ({d:.2e} >= {t}); model.keras is not "
                "replaced",
                d=diff,
                t=TRANSFER_TOL,
            )
        )
    model.save(out / "model.keras")
    import tensorflow as tf

    rep = {
        "max_abs_diff": diff,
        "tolerance": TRANSFER_TOL,
        "rows_checked": int(len(chk["X"])),
        "split": "test",
        "tensorflow": tf.__version__,
    }
    (out / "transfer_report.json").write_text(json.dumps(rep), encoding="utf-8")
    return rep


def _train(spec_path: Path, result_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    kw = dict(spec["kwargs"], backend="torch", out_dir=spec["out_dir"])
    if "hidden" in kw:
        kw["hidden"] = tuple(kw["hidden"])
    if "filters" in kw:
        kw["filters"] = tuple(kw["filters"])
    if "exclude" in kw:
        kw["exclude"] = tuple(kw["exclude"])
    kind = spec["kind"]
    if kind == "mlp":
        from .keras_mlp import train_keras_mlp as fn

        res = fn(spec["recipe"], **kw)
    else:
        from .cnn1d import train_keras_cnn1d as fn

        res = fn(spec["recipe"], **kw)
    result_path.write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[1] == "train":
        _train(Path(argv[2]), Path(argv[3]))
        return 0
    if len(argv) >= 3 and argv[1] == "transfer":
        _transfer(Path(argv[2]))
        return 0
    print("uso: python -m tinyml_studio.train.backend train <spec> <result> | transfer <dir>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
