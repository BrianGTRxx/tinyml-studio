import shutil

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.dataset import inspect_files
from tinyml_studio.export import export_run
from tinyml_studio.export.arduino import build_arduino_sketch
from tinyml_studio.train import train_models

RNG = np.random.default_rng(21)


def _clf_csv(path, n=600):
    k = n // 3
    base = np.repeat([0.0, 5.0, 10.0], k)
    pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1.0, n),
            "f2": RNG.normal(0, 1, n) + base / 2,
            "tipo": RNG.choice(["u", "v", "w"], n),
            "label": np.repeat(["a", "b", "c"], k),
        }
    ).to_csv(path, index=False)


@pytest.fixture(scope="module")
def classic_export(tmp_path_factory):
    if shutil.which("cc") is None:
        pytest.skip("sin compilador C")
    tmp = tmp_path_factory.mktemp("classic")
    _clf_csv(tmp / "d.csv")
    recipe = inspect_files([tmp / "d.csv"])["recipe"]
    train_models(recipe, ["decision_tree"], seed=0, out_dir=tmp / "run")
    export_run(tmp / "run", tmp / "c")
    return tmp / "c"


@pytest.fixture(scope="module")
def tflite_export(tmp_path_factory):
    pytest.importorskip("tensorflow")
    pytest.importorskip("tflite_micro")
    from tinyml_studio.export.tflite import export_tflite
    from tinyml_studio.train.keras_mlp import train_keras_mlp

    tmp = tmp_path_factory.mktemp("tfl")
    _clf_csv(tmp / "d.csv", 900)
    recipe = inspect_files([tmp / "d.csv"])["recipe"]
    train_keras_mlp(recipe, out_dir=tmp / "run", hidden=(8,), epochs=40, batch_size=32, seed=0)
    export_tflite(tmp / "run", tmp / "x")
    return tmp / "x"


def test_raw_c_sketch_compiles_and_predicts(classic_export, tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("sin g++")
    r = build_arduino_sketch(classic_export, tmp_path, "mi-modelo", "raw_c")
    sk = tmp_path / "mi_modelo"
    assert r["name"] == "mi_modelo" and r["variant"] == "raw_c"
    assert set(r["files"]) == {
        "mi_modelo.ino",
        "tm_model.c",
        "tm_model.h",
        "model_config.h",
        "README.txt",
    }
    assert all((sk / f).exists() for f in r["files"])
    assert r["verification"]["level"] == "host_compiled", r["verification"]
    ino = (sk / "mi_modelo.ino").read_text()
    assert "Serial.begin(115200)" in ino and "tm_predict(" in ino and "password" not in ino.lower()
    names = [(t["array"], t["name"]) for t in r["input_table"]]
    assert ("num", "f1") in names and ("cat", "tipo") in names
    assert len(r["steps"]) >= 5 and "README" not in "".join(r["steps"][:1])
    assert "ESP32S3 Dev Module" in (sk / "README.txt").read_text()
    assert r["variant"] == build_arduino_sketch(classic_export, tmp_path, "x2")["variant"]


def test_raw_c_detects_wrong_expectation(classic_export, tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("sin g++")
    bad = tmp_path / "bad"
    shutil.copytree(classic_export, bad)
    lines = (bad / "test_vectors.csv").read_text().splitlines()
    cells = lines[1].split(",")
    cells[-2] = str((int(float(cells[-2])) + 1) % 3)
    lines[1] = ",".join(cells)
    (bad / "test_vectors.csv").write_text("\n".join(lines) + "\n")
    r = build_arduino_sketch(bad, tmp_path / "o", "bad")
    assert r["verification"]["level"] == "not_compiled"
    assert "NO" in r["verification"]["detail"] or "NOT" in r["verification"]["detail"]


def test_variant_mismatch_and_missing(classic_export, tmp_path):
    with pytest.raises(ValueError):
        build_arduino_sketch(classic_export, tmp_path, "m", "eloquent")
    with pytest.raises(FileNotFoundError):
        build_arduino_sketch(tmp_path, tmp_path / "o", "m")
    with pytest.raises(ValueError):
        build_arduino_sketch(classic_export, tmp_path, "m", "nope")


def test_eloquent_sketch(tflite_export, tmp_path):
    r = build_arduino_sketch(tflite_export, tmp_path, "tfl", units={"f1": "A"})
    assert r["variant"] == "eloquent"
    assert r["verification"]["level"] == "not_compiled"
    assert set(r["files"]) == {
        "tfl.ino",
        "model.h",
        "model_config.h",
        "tm_prepare.c",
        "tm_prepare.h",
        "README.txt",
    }
    sk = tmp_path / "tfl"
    ino = (sk / "tfl.ino").read_text()
    for s in (
        "#include <tflm_esp32.h>",
        "#include <eloquent_tinyml.h>",
        "Eloquent::TF::Sequential<NUM_OPS, ARENA_SIZE> tf;",
        "tf.setNumInputs(",
        "tf.setNumOutputs(",
        "tf.resolver.AddFullyConnected();",
        "tf.begin(model_data).isOk()",
        "tf.predict(x, y)",
    ):
        assert s in ino, s
    assert f"#define NUM_OPS   {len(r['ops'])}" in ino
    assert "const unsigned char model_data[]" in (sk / "model.h").read_text()
    assert any(t.get("unit") == "A" for t in r["input_table"])
    assert any("EloquentTinyML" in s for s in r["steps"])
    readme = (sk / "README.txt").read_text()
    assert "ARENA_SIZE" not in readme or "arena" in readme.lower()
    assert "ssid" not in ino.lower() and "wifi" not in ino.lower()


def test_tflm_sketch(tflite_export, tmp_path):
    r = build_arduino_sketch(tflite_export, tmp_path, "raw", "tflm", arena_size=12345)
    ino = (tmp_path / "raw" / "raw.ino").read_text()
    for s in (
        "MicroMutableOpResolver<NUM_OPS>",
        "MicroInterpreter",
        "input->params.scale",
        "input->params.zero_point",
        "resolver.AddFullyConnected();",
        "#define ARENA_SIZE 12345",
    ):
        assert s in ino, s
    assert r["verification"]["level"] == "not_compiled" and r["arena_size"] == 12345
