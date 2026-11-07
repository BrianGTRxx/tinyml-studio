"""Validación por sesiones: qué se puede medir y qué no, y el detector de columnas de sesión."""

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.dataset import inspect_files
from tinyml_studio.train.groups import group_validation, session_columns

RNG = np.random.default_rng(3)


def _make(tmp_path, sessions):
    """sessions: lista de (archivo, clase, canal); dos características que separan las clases."""
    centers = {"a": (0.0, 0.0), "b": (6.0, 0.0), "c": (0.0, 6.0)}
    paths = []
    for name, cls, canal in sessions:
        cx, cy = centers[cls]
        df = pd.DataFrame(
            {
                "x1": RNG.normal(cx, 0.6, 120),
                "x2": RNG.normal(cy, 0.6, 120),
                "canal": canal,
                "clase": cls,
            }
        )
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    rec = inspect_files(paths)["recipe"]
    rec["task"] = "classification"
    rec["target"] = "clase"
    rec["label_from_file"] = {"enabled": False}
    rec["roles"] = {"x1": "feature", "x2": "feature", "canal": "feature", "clase": "target"}
    return rec


def test_group_validation_measures_only_classes_seen_in_two_sessions(tmp_path):
    rec = _make(
        tmp_path,
        [
            ("a1", "a", "k1"),
            ("a2", "a", "k2"),
            ("b1", "b", "k1"),
            ("b2", "b", "k2"),
            ("c1", "c", "k3"),
        ],
    )
    r = group_validation(rec, model="random_forest", seed=1)
    assert r["verdict"] == "partial"
    assert r["with_all_features"]["classes_measurable"] == ["a", "b"]
    assert r["with_all_features"]["classes_not_measurable"] == ["c"]
    assert r["with_all_features"]["skipped_groups"] == [
        "c1.csv"
    ]  # su clase no existe en otra sesión
    assert len(r["with_all_features"]["folds"]) == 4
    # la columna 'canal' (un valor por sesión) confunde al modelo: sin ella se separan bien
    assert r["session_columns"] == ["canal"]
    assert r["without_session_columns"]["mean_accuracy"] > 0.95
    assert r["with_all_features"]["mean_accuracy"] < r["without_session_columns"]["mean_accuracy"]
    assert any("2 sesiones" in w for w in r["warnings"])


def test_session_column_detected_and_dropped_in_second_run(tmp_path):
    rec = _make(
        tmp_path, [("a1", "a", "k1"), ("a2", "a", "k2"), ("b1", "b", "k1"), ("b2", "b", "k2")]
    )
    r = group_validation(rec, model="decision_tree", seed=1)
    assert r["session_columns"] == ["canal"]
    assert r["without_session_columns"] is not None
    assert r["verdict"] == "measurable"
    assert any("canal" in w for w in r["warnings"])


def test_nothing_measurable_with_one_session_per_class(tmp_path):
    rec = _make(tmp_path, [("a1", "a", "k1"), ("b1", "b", "k2"), ("c1", "c", "k3")])
    r = group_validation(rec, model="decision_tree", seed=1)
    assert r["verdict"] == "not_measurable" and r["with_all_features"]["mean_accuracy"] is None


def test_session_columns_ignores_columns_that_vary_inside_sessions():
    df = pd.DataFrame(
        {
            "source": ["f1"] * 40 + ["f2"] * 40,
            "const": ["u"] * 40 + ["v"] * 40,
            "mixed": (["u", "v"] * 20) + (["u", "v"] * 20),
        }
    )
    assert session_columns(df, "source", ["const", "mixed"]) == ["const"]


def test_regression_is_rejected(tmp_path):
    rec = _make(tmp_path, [("a1", "a", "k1"), ("a2", "a", "k2")])
    rec["task"] = "regression"
    with pytest.raises(ValueError):
        group_validation(rec)


def test_duplicated_session_is_not_counted_as_generalization(tmp_path):
    rec = _make(
        tmp_path, [("a1", "a", "k1"), ("a2", "a", "k2"), ("b1", "b", "k1"), ("b2", "b", "k2")]
    )
    (tmp_path / "a2.csv").write_text((tmp_path / "a1.csv").read_text())  # mismo contenido
    rec = inspect_files([tmp_path / f"{n}.csv" for n in ("a1", "a2", "b1", "b2")])["recipe"] | {
        "task": "classification",
        "target": "clase",
        "label_from_file": {"enabled": False},
        "roles": {"x1": "feature", "x2": "feature", "canal": "feature", "clase": "target"},
    }
    r = group_validation(rec, model="decision_tree", seed=1)
    full = r["with_all_features"]
    assert set(full["duplicate_groups"]) == {"a1.csv", "a2.csv"}
    assert full["classes_measurable"] == ["b"] and full["classes_not_measurable"] == ["a"]
    assert any("copias" in w for w in r["warnings"])
