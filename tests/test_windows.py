import numpy as np
import pandas as pd
import pytest

from tinyml_studio.window import WindowConfig, make_windows


def _run(X, y, group, time, **kw):
    cfg = WindowConfig(**{"window": 4, "stride": 1, **kw})
    return make_windows(
        np.asarray(X, dtype="float32"),
        np.asarray(y),
        pd.Series(group),
        None if time is None else pd.Series(time),
        cfg,
    )


def test_basic_windows_and_order():
    X = np.arange(10, dtype="float32")[:, None]
    Xw, yw, idx, st = _run(X, [0] * 10, ["g"] * 10, np.arange(10))
    assert Xw.shape == (7, 4, 1) and st["windows"] == 7
    np.testing.assert_array_equal(Xw[0, :, 0], [0, 1, 2, 3])  # de más antigua a más nueva
    np.testing.assert_array_equal(idx[-1], [6, 7, 8, 9])


def test_stride():
    Xw, *_ = _run(np.arange(20)[:, None], [0] * 20, ["g"] * 20, np.arange(20), stride=4)
    assert len(Xw) == 5


def test_sorts_by_time_inside_group():
    t = [3, 1, 0, 2, 4, 5]  # filas desordenadas
    X = np.array(t, dtype="float32")[:, None]
    Xw, *_ = _run(X, [0] * 6, ["g"] * 6, t)
    np.testing.assert_array_equal(Xw[0, :, 0], [0, 1, 2, 3])


def test_never_crosses_groups_or_time_gaps_or_labels():
    # dos grupos de 5 filas: ninguna ventana mezcla filas de ambos
    X = np.arange(10, dtype="float32")[:, None]
    group = ["a"] * 5 + ["b"] * 5
    _, _, idx, st = _run(X, [0] * 10, group, np.arange(10))
    assert all(len({group[i] for i in w}) == 1 for w in idx) and st["windows"] == 4
    # hueco grande de tiempo en medio: se corta
    t = [0, 1, 2, 3, 4, 100, 101, 102, 103, 104]
    _, _, idx2, st2 = _run(X, [0] * 10, ["g"] * 10, t)
    assert all(not (w.min() <= 4 < w.max()) for w in idx2) and st2["segments"] == 2
    # etiqueta que cambia dentro de la ventana: se descarta
    y = [0] * 5 + [1] * 5
    _, yw, _, st3 = _run(X, y, ["g"] * 10, np.arange(10))
    assert st3["dropped_mixed_label"] == 3 and set(yw) == {0, 1}


def test_no_time_uses_row_order_and_fails_when_impossible():
    Xw, *_ = _run(np.arange(6)[:, None], [0] * 6, ["g"] * 6, None)
    assert len(Xw) == 3
    with pytest.raises(ValueError, match="ninguna ventana"):
        _run(np.arange(3)[:, None], [0] * 3, ["g"] * 3, np.arange(3))
    with pytest.raises(ValueError):
        WindowConfig(window=1).validate()
