import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

pytest.importorskip("tensorflow")

from tinyml_studio.api.catalog import get_catalog  # noqa: E402
from tinyml_studio.api.handlers import CnnParams, KerasParams, KerasPreset  # noqa: E402
from tinyml_studio.dataset import inspect_files  # noqa: E402
from tinyml_studio.export.tflite import export_tflite  # noqa: E402
from tinyml_studio.train import backend as B  # noqa: E402
from tinyml_studio.train import models as M  # noqa: E402
from tinyml_studio.train.keras_mlp import train_keras_mlp  # noqa: E402

RNG = np.random.default_rng(7)


def _recipe(tmp, n=600, k=200):
    base = np.repeat([0.0, 6.0, 12.0], k)
    pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1.0, n),
            "f2": RNG.normal(0, 1, n) + base / 2,
            "label": np.repeat(["a", "b", "c"], k),
        }
    ).to_csv(tmp / "d.csv", index=False)
    return inspect_files([tmp / "d.csv"])["recipe"]


def test_default_is_plain_adam():
    opt = B.make_optimizer("adam", 0.002)
    assert type(opt).__name__ == "Adam" and abs(float(opt.learning_rate) - 0.002) < 1e-9


@pytest.mark.parametrize("name", B.OPTIMIZERS)
@pytest.mark.parametrize("sched", B.SCHEDULES)
def test_every_optimizer_and_schedule_builds(name, sched):
    opt = B.make_optimizer(
        name, 0.01, lr_schedule=sched, lr_final_frac=0.1, steps_per_epoch=5, epochs=4
    )
    assert opt is not None
    if sched != "constant":
        assert abs(float(opt.learning_rate) - 0.01) < 1e-9
        opt.iterations.assign(20)  # 5 pasos/época * 4 épocas
        assert abs(float(opt.learning_rate) - 0.001) < 1e-6  # al final: lr * lr_final_frac


def test_sgd_momentum_and_adamw_decay():
    sgd = B.make_optimizer("sgd", 0.1, nesterov=True)
    assert float(sgd.momentum) == 0.9 and sgd.nesterov
    assert abs(float(B.make_optimizer("adamw", 0.1, weight_decay=0.05).weight_decay) - 0.05) < 1e-9
    with pytest.raises(ValueError, match="no soportado"):
        B.make_optimizer("lion")
    with pytest.raises(ValueError):
        B.make_optimizer("adam", lr_schedule="step")


@pytest.mark.parametrize(
    "kw,rec",
    [
        (
            {"optimizer": "adamw", "weight_decay": 0.01},
            {"optimizer": "adamw", "weight_decay": 0.01},
        ),
        (
            {"optimizer": "sgd", "nesterov": True, "learning_rate": 0.05, "epochs": 80},
            {"optimizer": "sgd", "nesterov": True, "momentum": 0.9},
        ),
    ],
)
def test_int8_export_still_works(tmp_path, kw, rec):
    r = _recipe(tmp_path)
    run = tmp_path / "run"
    base = {"hidden": (8,), "epochs": 60, "batch_size": 32, "seed": 0, "learning_rate": 0.01}
    res = train_keras_mlp(r, out_dir=run, lr_schedule="cosine", **{**base, **kw})
    hp = res["hyperparameters"]
    assert hp["lr_schedule"] == "cosine" and hp["lr_final_frac"] == 0.01
    assert all(hp[k] == v for k, v in rec.items())
    assert res["test"]["f1_macro"] > 0.8
    rep = export_tflite(run, tmp_path / "x")
    assert rep["comparison"]["float_vs_keras_outputs"]["max_abs"] < 1e-4
    assert rep["metrics_test"]["tflite_int8"]["f1_macro"] > 0.8


@pytest.mark.parametrize("opt", ["nadam", "rmsprop", "adagrad"])
def test_other_optimizers_train(tmp_path, opt):
    res = train_keras_mlp(
        _recipe(tmp_path), hidden=(4,), epochs=3, seed=0, optimizer=opt, lr_schedule="exp_decay"
    )
    assert res["hyperparameters"]["optimizer"] == opt


def test_default_hyperparameters_record_adam_constant(tmp_path):
    res = train_keras_mlp(_recipe(tmp_path), hidden=(4,), epochs=2, seed=0)
    hp = res["hyperparameters"]
    assert hp["optimizer"] == "adam" and hp["lr_schedule"] == "constant"
    assert "weight_decay" not in hp and "lr_final_frac" not in hp


def test_handler_validation():
    p = KerasParams(recipe={})
    assert (p.optimizer, p.lr_schedule, p.backend, p.device) == (
        "adam",
        "constant",
        "tensorflow",
        "auto",
    )
    assert KerasParams(recipe={}, optimizer="adamw", device="gpu", backend="torch").device == "gpu"
    assert CnnParams(recipe={}, optimizer="sgd", nesterov=True).nesterov
    for bad in (
        {"optimizer": "lion"},
        {"lr_schedule": "step"},
        {"backend": "jax"},
        {"device": "tpu"},
        {"weight_decay": -1},
        {"lr_final_frac": 0},
        {"momentum": 0.5},  # extra='forbid'
    ):
        with pytest.raises(ValidationError):
            KerasParams(recipe={}, **bad)
        with pytest.raises(ValidationError):
            CnnParams(recipe={}, **bad)


def test_benchmark_preset_fields_are_optional_and_bounded():
    # sin tocar nada, el preset sigue volcando solo las claves de siempre (bench.py las conoce)
    from tinyml_studio.train.bench import KERAS_KEYS

    assert set(KerasPreset().model_dump(exclude_none=True)) <= KERAS_KEYS
    assert KerasPreset(optimizer="rmsprop", lr_schedule="cosine").optimizer == "rmsprop"
    with pytest.raises(ValidationError):
        KerasPreset(optimizer="lion")
    with pytest.raises(ValidationError):
        KerasPreset(lr_final_frac=2)


def test_catalog_lists_optimizers_with_bilingual_notes():
    from tinyml_studio.i18n import set_lang

    for lang in ("es", "en"):
        set_lang(lang)
        cat = get_catalog()
        for fam in ("keras_mlp", "keras_cnn1d"):
            ids = [o["id"] for o in cat[fam]["optimizers"]]
            assert ids == list(B.OPTIMIZERS) and all(o["note"] for o in cat[fam]["optimizers"])
            assert cat[fam]["defaults"]["optimizer"] == "adam"
            assert [s["id"] for s in cat[fam]["lr_schedules"]] == list(B.SCHEDULES)
        assert cat["classical"]["mlp_solvers"] == ["adam", "sgd", "lbfgs"]
    set_lang(None)
    es = get_catalog()["keras_mlp"]["optimizers"][0]["note"]
    set_lang("en")
    assert get_catalog()["keras_mlp"]["optimizers"][0]["note"] != es
    set_lang(None)


@pytest.mark.parametrize("solver", ["adam", "sgd", "lbfgs"])
def test_sklearn_mlp_solver(solver):
    assert M.DEFAULTS["mlp"]["solver"] == "adam"
    est, eff = M.build("mlp", "classification", 0, {"solver": solver})
    assert est.solver == solver and eff["solver"] == solver
    x = RNG.normal(size=(60, 3))
    est.fit(x, (x[:, 0] > 0).astype(int))
    with pytest.raises(ValueError, match="solver"):
        M.build("mlp", "classification", 0, {"solver": "newton"})
