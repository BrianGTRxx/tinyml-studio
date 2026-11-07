"""El callback de Keras informa el ritmo por época y una cota del tiempo restante."""

import json

import numpy as np
import pytest

pytest.importorskip("keras")


def test_epoch_events_carry_pace_and_eta(monkeypatch, capsys):
    monkeypatch.setenv("TINYML_JOB", "1")
    import keras

    from tinyml_studio.progress import keras_callback

    model = keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(2)])
    model.compile("sgd", "mse")
    cb = keras_callback(4)
    x = np.random.default_rng(0).normal(size=(64, 3)).astype("float32")
    model.fit(x, x[:, :2], epochs=4, verbose=0, callbacks=[cb])
    out = capsys.readouterr().out.splitlines()
    evs = [json.loads(line[len("@@EVENT ") :]) for line in out if line.startswith("@@EVENT ")]
    epochs = [e for e in evs if e.get("stage") == "epoch"]
    assert len(epochs) == 4 and epochs[-1]["eta_s"] == 0
    assert all(e["epoch_s"] > 0 and e["epochs"] == 4 for e in epochs)
    assert epochs[0]["eta_s"] > 0
