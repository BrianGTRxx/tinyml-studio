"""Motor de comprensión de datasets: casos sintéticos tipo Kaggle + dataset real de ejemplo."""

import json

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.dataset import apply_recipe, inspect_files
from tinyml_studio.dataset.harmonize import harmonize
from tinyml_studio.dataset.load import coerce_frame, load_table

RNG = np.random.default_rng(0)


def col(rep, name):
    return next(c for c in rep["columns"] if c["name"] == name)


def codes(rep):
    return {a["code"] for a in rep["alerts"]}


def titanic_like(path, n=300):
    surv = RNG.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "PassengerId": np.arange(1, n + 1),
            "Survived": surv,
            "Pclass": RNG.integers(1, 4, n),
            "Name": [f"Persona Numero {i}, Sr. Apellido{i}" for i in range(n)],
            "Sex": RNG.choice(["male", "female"], n),
            "Age": np.where(RNG.random(n) < 0.2, np.nan, RNG.normal(30, 10, n).round(1)),
            "Fare": RNG.exponential(30, n).round(2),
            "Cabin": ["?"] * n,
        }
    )
    df.to_csv(path, index=False)
    return df




def test_single_file_classification(tmp_path):
    titanic_like(tmp_path / "train.csv")
    rep = inspect_files([tmp_path / "train.csv"])
    assert rep["task"]["task"] == "classification"
    assert rep["task"]["target"] == "survived"
    assert col(rep, "passengerid")["role"] == "drop" and col(rep, "passengerid")["kind"] == "id"
    assert col(rep, "name")["role"] == "drop"
    assert col(rep, "cabin")["role"] == "drop"  # solo "?" -> nulos -> vacía
    assert col(rep, "sex")["kind"] == "binary"
    assert col(rep, "age")["role"] == "feature"
    assert rep["split"]["strategy"] == "stratified"
    json.dumps(rep)  # serializable
    assert "Tarea sugerida" in rep["summary_es"]


def test_latin1_semicolon_decimal_comma(tmp_path):
    f = tmp_path / "datos.csv"
    rows = ["Temperatura;Humedad;Estado"] + [
        f"{20 + i % 7},{i % 10};{50 + i % 9},5;{'alto' if i % 2 else 'bajo'}" for i in range(60)
    ]
    f.write_bytes("\n".join(rows).replace("alto", "álto").encode("latin-1"))
    t = load_table(f)
    assert t.encoding == "latin-1" and t.separator == ";"
    assert pd.api.types.is_float_dtype(t.df["Temperatura"])
    assert t.df["Humedad"].iloc[0] == pytest.approx(50.5)
    assert any(n["action"] == "numeric" and n["decimal_comma"] for n in t.coercions)


def test_coerce_bool_datetime_na():
    df = pd.DataFrame(
        {
            "a": ["yes", "no", "yes", "no"],
            "d": ["2024-01-01 10:00", "2024-01-02 10:00", "2024-01-03 10:00", "2024-01-04 10:00"],
            "n": ["1", "2", "NA", "4"],
        }
    )
    out, notes = coerce_frame(df)
    assert str(out["a"].dtype) == "boolean"
    assert pd.api.types.is_datetime64_any_dtype(out["d"])
    assert out["n"].isna().sum() == 1
    assert {n["action"] for n in notes} >= {"boolean", "datetime", "numeric", "na_tokens"}


def test_multi_file_different_columns(tmp_path):
    """Tres 'versiones' de un mismo dataset con nombres distintos, como en Kaggle."""
    n = 80
    for fname, names, species in [
        ("setosa.csv", ["sepal_length", "sepal_width", "petal_length"], "setosa"),
        ("versi.csv", ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)"], "versi"),
        ("virg.csv", ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm"], "virg"),
    ]:
        base = {"setosa": 5.0, "versi": 6.0, "virg": 6.5}[species]
        pd.DataFrame(
            {
                names[0]: RNG.normal(base, 0.3, n),
                names[1]: RNG.normal(3, 0.2, n),
                names[2]: RNG.normal(base - 3, 0.2, n),
            }
        ).to_csv(tmp_path / fname, index=False)
    tables = [load_table(tmp_path / f) for f in ("setosa.csv", "versi.csv", "virg.csv")]
    groups = harmonize(tables)
    assert len(groups) == 3 and all(len(g.members) == 3 for g in groups)
    rep = inspect_files([tmp_path])
    assert rep["task"]["label_from_file"] is True  # sin columna objetivo: etiqueta = archivo
    assert rep["rows_total"] == 3 * n
    assert not [a for a in rep["alerts"] if a["code"] == "missing_column"]
    df = apply_recipe(rep["recipe"])
    assert set(df["label"]) == {"setosa", "versi", "virg"}
    assert {"sepal_length", "sepal_width", "petal_length"} <= set(df.columns)


def test_incompatible_columns_not_merged(tmp_path):
    pd.DataFrame({"valor": RNG.normal(0, 1, 50)}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"valor": ["x", "y"] * 25}).to_csv(tmp_path / "b.csv", index=False)
    g = harmonize([load_table(tmp_path / "a.csv"), load_table(tmp_path / "b.csv")])
    assert len(g) == 2 and {x.canonical for x in g} == {"valor", "valor_2"}


def test_regression_target(tmp_path):
    n = 200
    pd.DataFrame(
        {
            "x1": RNG.normal(size=n),
            "x2": RNG.normal(size=n),
            "SalePrice": RNG.normal(200, 40, n),
        }
    ).to_csv(tmp_path / "h.csv", index=False)
    rep = inspect_files([tmp_path / "h.csv"])
    assert rep["task"]["task"] == "regression" and rep["task"]["target"] == "saleprice"
    assert rep["split"]["strategy"] == "random"


def test_no_target_asks_user(tmp_path):
    pd.DataFrame({"a": RNG.normal(size=100), "b": RNG.normal(size=100)}).to_csv(
        tmp_path / "s.csv", index=False
    )
    rep = inspect_files([tmp_path / "s.csv"])
    assert rep["task"]["task"] == "unsupervised" and rep["task"]["needs_user_confirmation"]


def test_quality_alerts(tmp_path):
    n = 400
    y = np.repeat([0, 1], [380, 20])  # desbalance + ordenado por clase
    x = RNG.normal(10, 1, n)
    x[5] = 9_000_000.0  # absurdo
    s = RNG.normal(5, 1, n)
    s[10] = -999  # centinela
    s[20] = -999
    pd.DataFrame({"x": x, "s": s, "label": y, "dup": y * 1}).to_csv(tmp_path / "q.csv", index=False)
    rep = inspect_files([tmp_path / "q.csv"])
    c = codes(rep)
    assert {"absurd_values", "sentinel", "imbalance", "sorted_target", "target_leak"} <= c
    leak = next(a for a in rep["alerts"] if a["code"] == "target_leak")
    assert leak["column"] == "dup"


def test_missing_column_alert_and_nulls_only_where_present(tmp_path):
    n = 60
    pd.DataFrame({"a": RNG.normal(size=n), "b": RNG.normal(size=n)}).to_csv(
        tmp_path / "1.csv", index=False
    )
    pd.DataFrame({"a": RNG.normal(size=n)}).to_csv(tmp_path / "2.csv", index=False)
    rep = inspect_files([tmp_path])
    assert any(a["code"] == "missing_column" and a["column"] == "b" for a in rep["alerts"])
    assert not any(a["code"] == "nulls" and a["column"] == "b" for a in rep["alerts"])
    assert col(rep, "b")["role"] == "feature"  # no se descarta por ausencia en otro archivo


def test_recipe_roundtrip_and_edit(tmp_path):
    titanic_like(tmp_path / "t.csv")
    rep = inspect_files([tmp_path / "t.csv"])
    recipe = json.loads(json.dumps(rep["recipe"]))
    a, b = apply_recipe(recipe), apply_recipe(recipe)
    pd.testing.assert_frame_equal(a, b)
    assert "passengerid" not in a.columns and "survived" in a.columns
    recipe["rename"][str(tmp_path / "t.csv")]["Fare"] = "tarifa"
    recipe["drop"].append("age")
    c = apply_recipe(recipe)
    assert "tarifa" in c.columns and "age" not in c.columns
    recipe["version"] = 99
    with pytest.raises(ValueError):
        apply_recipe(recipe)




def test_same_name_different_ranges_still_merges(tmp_path):
    """Cada archivo es otra clase con otro nivel: misma columna aunque los rangos no se solapen."""
    for name, lvl in (("a", 100.0), ("b", 900.0)):
        pd.DataFrame({"power": lvl + RNG.normal(0, 1, 60)}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    g = harmonize([load_table(tmp_path / "a.csv"), load_table(tmp_path / "b.csv")])
    assert len(g) == 1 and len(g[0].members) == 2
    # nombres solo parecidos con rangos disjuntos sí se mantienen separados (conservador)
    pd.DataFrame({"potencia_w": 5000 + RNG.normal(0, 1, 60)}).to_csv(
        tmp_path / "c.csv", index=False
    )
    pd.DataFrame({"potencia_wt": 10.0 + RNG.normal(0, 1, 60)}).to_csv(
        tmp_path / "d.csv", index=False
    )
    g2 = harmonize([load_table(tmp_path / "c.csv"), load_table(tmp_path / "d.csv")])
    assert len(g2) == 2
