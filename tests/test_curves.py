import json

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tinyml_studio.i18n import set_lang
from tinyml_studio.train.curves import (
    classification_quality,
    pr_curve_points,
    regression_quality,
    roc_curve_points,
    split_comparison,
)


def _data(n=400, k=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k)) + 2.0 * np.eye(k)[y]
    # round to force ties
    p = np.exp(logits.round(1))
    p /= p.sum(1, keepdims=True)
    return y, p


def _strict_json(d):
    return json.loads(json.dumps(d, allow_nan=False))


def test_classification_matches_sklearn():
    y, p = _data()
    pred = p.argmax(1)
    names = list("abcd")
    q = classification_quality(y, pred, p, names)
    _strict_json(q)
    yo = np.eye(4)[y]
    assert q["accuracy"] == pytest.approx((y == pred).mean())
    assert q["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    for avg in ("macro", "weighted"):
        assert q[f"f1_{avg}"] == pytest.approx(f1_score(y, pred, average=avg))
        assert q[f"precision_{avg}"] == pytest.approx(precision_score(y, pred, average=avg))
        assert q[f"recall_{avg}"] == pytest.approx(recall_score(y, pred, average=avg))
    assert q["roc_auc_macro"] == pytest.approx(roc_auc_score(y, p, multi_class="ovr"), abs=1e-9)
    assert q["mAP"] == pytest.approx(average_precision_score(yo, p, average="macro"), abs=1e-9)
    for i, c in enumerate(names):
        assert q["per_class"][c]["auc"] == pytest.approx(roc_auc_score(yo[:, i], p[:, i]))
        assert q["per_class"][c]["ap"] == pytest.approx(average_precision_score(yo[:, i], p[:, i]))
    cm = confusion_matrix(y, pred)
    assert np.array_equal(q["confusion_matrix"], cm)
    assert np.allclose(np.sum(q["confusion_matrix_norm"], 1), 1.0)
    assert any("mAP50" in n for n in q["notes"])


def test_notes_english():
    y, p = _data(50)
    set_lang("en")
    try:
        q = classification_quality(y, p.argmax(1), p, list("abcd"))
    finally:
        set_lang(None)
    assert "bounding boxes" in q["notes"][0]


def test_roc_pr_curves_and_downsample():
    y, p = _data(2000)
    roc = roc_curve_points(y, p, max_points=50)
    pr = pr_curve_points(np.eye(4)[y], p, max_points=50)
    for i in range(4):
        c = roc[str(i)]
        assert len(c["fpr"]) <= 50
        assert c["fpr"][0] == 0 and c["fpr"][-1] == 1 and c["tpr"][-1] == 1
        assert c["auc"] == pytest.approx(roc_auc_score(y == i, p[:, i]))
        assert pr[str(i)]["ap"] == pytest.approx(average_precision_score(y == i, p[:, i]))
        assert pr[str(i)]["recall"][0] == 0 and pr[str(i)]["recall"][-1] == 1
    assert roc == roc_curve_points(y, p, max_points=50)  # deterministic
    _strict_json(roc), _strict_json(pr)


def test_absent_class_binary_and_constant():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    q = classification_quality(y, (p > 0.5).astype(int), p, ["neg", "pos"])
    assert q["per_class"]["pos"]["auc"] == pytest.approx(roc_auc_score(y, p))
    assert q["per_class"]["pos"]["ap"] == pytest.approx(average_precision_score(y, p))
    # absent class 2
    p3 = np.full((6, 3), 1 / 3)
    q = classification_quality(y, np.zeros(6, int), p3, ["a", "b", "c"])
    assert q["per_class"]["c"]["auc"] is None and q["per_class"]["c"]["ap"] is None
    assert q["per_class"]["a"]["auc"] == pytest.approx(0.5)  # constant scores
    assert len(q["notes"]) >= 2
    assert roc_curve_points(y, p3, classes=["a", "b", "c"])["c"] == {
        "fpr": [],
        "tpr": [],
        "auc": None,
    }
    _strict_json(q)




def test_regression_quality():
    rng = np.random.default_rng(2)
    y = rng.normal(size=1000)
    p = y + rng.normal(scale=0.3, size=1000)
    q = regression_quality(y, p, max_points=100)
    _strict_json(q)
    assert q["mae"] == pytest.approx(np.abs(y - p).mean())
    assert q["rmse"] == pytest.approx(np.sqrt(((y - p) ** 2).mean()))
    from sklearn.metrics import r2_score

    assert q["r2"] == pytest.approx(r2_score(y, p))
    assert sum(q["residual_hist"]["counts"]) == 1000
    assert len(q["residual_hist"]["bins"]) == len(q["residual_hist"]["counts"]) + 1
    assert len(q["scatter"]["y_true"]) == 100
    assert q == regression_quality(y, p, max_points=100)
    q = regression_quality([1.0, 1.0, np.nan], [1.0, 2.0, 3.0])
    assert q["n"] == 2 and q["r2"] is None
    _strict_json(q)


def test_split_comparison():
    s = split_comparison(
        {
            "train": {"accuracy": 0.99, "f1_macro": 0.98, "n": 80, "mae": 0.1},
            "val": {"accuracy": 0.80, "f1_macro": 0.79, "n": 10, "mae": 0.3},
        }
    )
    assert s["metrics"]["accuracy"] == {"train": 0.99, "val": 0.80}
    assert s["gaps"]["accuracy"]["val"] == pytest.approx(0.19)
    assert "mae" not in s["gaps"] and "n" not in s["metrics"]
    assert s["notes"]
    _strict_json(s)

