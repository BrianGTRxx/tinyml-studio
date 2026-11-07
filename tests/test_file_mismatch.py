"""Aviso cuando varios archivos parecen medidos de forma distinta (escala o signo)."""

import numpy as np
import pandas as pd

from tinyml_studio.dataset.engine import inspect_files


def _csv(path, scale=1.0, sign=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = 300
    df = pd.DataFrame(
        {
            "p": rng.normal(50, 5, n) * scale,
            "q": sign * rng.normal(-12, 1, n),
            "t": np.arange(n),
        }
    )
    df.to_csv(path, index=False)
    return path


def _mismatch(rep):
    return [a for a in rep["alerts"] if a["code"] == "file_mismatch"]


def test_same_measurements_give_no_alert(tmp_path):
    rep = inspect_files([_csv(tmp_path / "a.csv", seed=1), _csv(tmp_path / "b.csv", seed=2)])
    assert _mismatch(rep) == []


def test_scale_difference_is_flagged(tmp_path):
    rep = inspect_files([_csv(tmp_path / "a.csv"), _csv(tmp_path / "b.csv", scale=500, seed=3)])
    al = _mismatch(rep)
    assert [a["kind"] for a in al] == ["scale"] and al[0]["column"] == "p"
    assert al[0]["severity"] == "high"


def test_opposite_sign_is_flagged(tmp_path):
    rep = inspect_files([_csv(tmp_path / "a.csv"), _csv(tmp_path / "b.csv", sign=-1.0, seed=4)])
    al = _mismatch(rep)
    assert [a["kind"] for a in al] == ["sign"] and al[0]["column"] == "q"


def test_single_file_never_flags(tmp_path):
    assert _mismatch(inspect_files([_csv(tmp_path / "a.csv")])) == []
