"""i18n del backend de datos: en inglés no debe colarse español; en 'es' sale el texto original."""

import re

import numpy as np
import pandas as pd
import pytest

from tinyml_studio.clean import prepare_splits
from tinyml_studio.clean.compare import figure
from tinyml_studio.clean.suggest import suggest_cleaning
from tinyml_studio.dataset import inspect_files
from tinyml_studio.i18n import set_lang
from tinyml_studio.split import split_group

SPANISH_ACCENTS = re.compile(r"[áéíóúñ¿¡]")
SPANISH_WORDS = (
    " de ",
    " los ",
    " la ",
    " con ",
    " sin ",
    " columna",
    " filas",
    " archivo ",
    " alertas",
    " se ",
    " y ",
)


@pytest.fixture(autouse=True)
def _reset_lang():
    yield
    set_lang(None)


def _dataset(tmp_path, n=300):
    rng = np.random.default_rng(1)
    paths = []
    for i, name in enumerate(["Fan_A.csv", "Pump_B.csv", "Heater_C.csv"]):
        temp = rng.normal(20 + 5 * i, 2, n).round(2)
        temp[:3] = -999  # centinela
        temp[5] = 1e7  # absurdo
        df = pd.DataFrame(
            {
                "Temperature": temp,
                "Pressure": rng.normal(1.0, 0.1, n),
                "Vibration": np.where(rng.random(n) < 0.3, np.nan, rng.normal(0, 1, n)),
                "Sample": np.arange(n),
                "Humidity": np.cumsum(rng.normal(0, 0.01, n)),
            }
        )
        if i == 0:
            df["Extra"] = rng.normal(0, 1, n)
        df = pd.concat([df, df.iloc[:60]], ignore_index=True)  # duplicados
        p = tmp_path / name
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


def _texts(rep):
    out = [a["message"] for a in rep["alerts"]]
    out += [c["role_reason"] for c in rep["columns"] if c["role"] != "meta" or True]
    out += [c["kind_reason"] for c in rep["columns"]]
    out += [rep["task"]["reason"], rep["split"]["reason"], rep["summary"]]
    out += [g["reason"] for g in rep["column_groups"]]
    out += [s["why"] for s in rep["recipe"]["cleaning"]]
    return out


def _assert_english(texts):
    assert texts
    for t in texts:
        low = f" {t.lower()} "
        assert not SPANISH_ACCENTS.search(low), t
        for w in SPANISH_WORDS:
            assert w not in low, (w, t)


def test_inspect_clean_split_english(tmp_path):
    paths = _dataset(tmp_path)
    set_lang("en")
    rep = inspect_files(paths)
    _assert_english(_texts(rep))
    codes = {a["code"] for a in rep["alerts"]}
    assert {"sentinel", "absurd_values", "duplicates", "missing_column"} <= codes

    prep = prepare_splits(rep["recipe"])
    assert set(prep.clean) == {"train", "val", "test"}
    fig = figure(*(prep.raw["train"], prep.clean["train"]), ["Temperature"])
    assert "Before / after" in fig.layout.title.text
    with pytest.raises(ValueError) as e:
        split_group(pd.Series(["a", "a", "b"]), (0.8, 0.1, 0.1), 0)
    _assert_english([str(e.value)])
    with pytest.raises(ValueError) as e:
        prepare_splits({**rep["recipe"], "version": 99})
    _assert_english([str(e.value)])
    steps = suggest_cleaning(
        [{"name": "x", "role": "feature", "kind": "numeric_continuous", "stats": {"nulls": 1}}],
        [],
        False,
    )
    _assert_english([s["why"] for s in steps])


def test_spanish_is_default_and_differs(tmp_path):
    paths = _dataset(tmp_path)
    set_lang("en")
    en = inspect_files(paths)
    set_lang("es")
    es = inspect_files(paths)
    assert es["summary_es"] == es["summary"]
    assert en["summary"] != es["summary"]
    assert any("valor(es) absurdo(s)" in a["message"] for a in es["alerts"])
    assert any("absurd value(s)" in a["message"] for a in en["alerts"])
    assert "Archivos:" in es["summary"] and "Files:" in en["summary"]
    set_lang(None)
    assert inspect_files(paths)["summary"] == es["summary"]
