import json

import numpy as np
import pytest

from tinyml_studio.export.graph import keras_graph, sklearn_graph, tflite_graph

LIMIT = 300_000
FEATS = ["sepal_l", "sepal_w", "petal_l", "petal_w"]
CLASSES = ["a", "b", "c"]


def _data(n=300):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], n // 3)
    X = rng.normal(0, 1, (n, 4)) + y[:, None] * 2.5
    return X, y


def _size(g):
    return len(json.dumps(g))


@pytest.fixture(scope="module")
def keras_model():
    tf = pytest.importorskip("tensorflow")
    X, y = _data()
    m = tf.keras.Sequential(
        [
            tf.keras.Input((4,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    m.compile("adam", "sparse_categorical_crossentropy")
    m.fit(X, y, epochs=2, verbose=0)
    return m, X


def _tfl(m, X, int8):
    import tensorflow as tf

    c = tf.lite.TFLiteConverter.from_keras_model(m)
    if int8:
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = lambda: ([X[i : i + 1].astype("f")] for i in range(50))
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.int8
        c.inference_output_type = tf.int8
    return c.convert()


def test_tflite_int8(keras_model):
    pytest.importorskip("tflite_micro")
    m, X = keras_model
    g = tflite_graph(_tfl(m, X, True))
    assert g["format"] == "tflite" and _size(g) < LIMIT
    json.dumps(g)
    ops = [n["op"] for n in g["nodes"]]
    assert ops.count("FULLY_CONNECTED") == 2 and "SOFTMAX" in ops
    t = {x["id"]: x for x in g["tensors"]}
    i, o = g["inputs"][0], g["outputs"][0]
    assert t[i]["shape"] == [1, 4] and t[i]["dtype"] == "INT8" and t[i]["quant"]["scale"] > 0
    assert t[o]["shape"] == [1, 3]
    consts = [x for x in g["tensors"] if x["is_const"]]
    assert consts and all(len(x["data"]["hist"]["counts"]) == 20 for x in consts)
    assert any("values" in x["data"] for x in consts)  # bias de 3 elementos
    assert g["stats"]["op_counts"]["FULLY_CONNECTED"] == 2
    assert g["stats"]["param_count"] >= 4 * 8 + 8 * 3
    assert g["summary"]["quantized"] is True
    for n in g["nodes"]:
        assert all(k in t for k in n["inputs"] + n["outputs"])


def test_tflite_float(keras_model):
    pytest.importorskip("tflite_micro")
    m, X = keras_model
    g = tflite_graph(_tfl(m, X, False))
    assert g["summary"]["quantized"] is False
    assert g["stats"]["total_const_bytes"] > 0


def test_keras(keras_model):
    m, _ = keras_model
    g = keras_graph(m)
    json.dumps(g)
    assert [n["op"] for n in g["nodes"]] == ["Dense", "Dense"]
    assert g["stats"]["param_count"] == 4 * 8 + 8 + 8 * 3 + 3


def test_decision_tree():
    from sklearn.tree import DecisionTreeClassifier

    X, y = _data()
    est = DecisionTreeClassifier(random_state=0).fit(X, y)
    g = sklearn_graph(est, FEATS, CLASSES, max_tree_depth=2)
    json.dumps(g)
    root = g["model"]["root"]
    assert root["feature"] in FEATS and "threshold" in root and root["samples"] == len(y)
    assert abs(sum(root["distribution"]) - 1) < 1e-9
    assert g["model"]["depth"] > 2 and g["model"]["truncated"] is True
    assert root["left"]["left"]["truncated"] is True or root["left"]["left"]["leaf"]
    assert _size(g) < LIMIT


def test_forest():
    from sklearn.ensemble import RandomForestClassifier

    X, y = _data()
    est = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    g = sklearn_graph(est, FEATS, CLASSES)
    json.dumps(g)
    assert g["model"]["n_trees"] == 20 and len(g["model"]["per_tree"]) == 20
    assert g["model"]["first_tree"]["root"]["feature"] in FEATS
    assert _size(g) < LIMIT


def test_logreg_and_svm_and_mlp():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    X, y = _data()
    g = sklearn_graph(LogisticRegression(max_iter=300).fit(X, y), FEATS, CLASSES)
    rows = g["model"]["classes"]
    assert [r["class"] for r in rows] == CLASSES
    assert rows[0]["coefficients"][0]["feature"] == "sepal_l"

    g = sklearn_graph(SVC(kernel="rbf").fit(X, y), FEATS, CLASSES)
    json.dumps(g)
    assert g["model"]["kernel"] == "rbf" and sum(g["model"]["n_support"]) > 0

    g = sklearn_graph(MLPClassifier((8,), max_iter=200, random_state=0).fit(X, y), FEATS, CLASSES)
    json.dumps(g)
    assert [n["op"] for n in g["nodes"]] == ["Dense", "Dense"]
    assert g["stats"]["param_count"] == 4 * 8 + 8 + 8 * 3 + 3
    assert _size(g) < LIMIT
