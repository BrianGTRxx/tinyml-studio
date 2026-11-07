"""Escalado que conserva el signo (potencia reactiva negativa) sin perder la exportación."""

import pandas as pd

from tinyml_studio.clean.steps import Scale


def test_keep_sign_does_not_center():
    df = pd.DataFrame({"q": [-40.0, -45.0, -10.0, 60.0], "p": [1.0, 2.0, 3.0, 4.0]})
    for method in ("standard", "minmax", "robust"):
        st = Scale(columns=["q", "p"], method=method, keep_sign=["q"]).fit(df)
        out = st.transform(df)
        assert st.state["scale"]["q"]["center"] == 0.0
        assert ((out["q"] < 0) == (df["q"] < 0)).all(), method
        assert st.state["scale"]["p"]["center"] != 0.0


def test_default_still_centers():
    df = pd.DataFrame({"q": [-40.0, -45.0, -10.0, 60.0]})
    st = Scale(columns=["q"], method="standard").fit(df)
    assert abs(st.transform(df)["q"].mean()) < 1e-9
