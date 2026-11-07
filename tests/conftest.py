from pathlib import Path

import pytest

LEVELS = {"a": 100.0, "b": 300.0, "c": 700.0}


def three_files(tmp_path, n=400):
    """Tres 'aparatos' (un archivo cada uno) con potencia 100/300/700: las sumas de cualquier
    combinación son distintas, así que una regla de suma correcta es resoluble."""
    import numpy as np
    import pandas as pd

    from tinyml_studio.dataset import inspect_files

    rng = np.random.default_rng(3)
    for k, lvl in LEVELS.items():
        pd.DataFrame(
            {
                "power": lvl + rng.normal(0, lvl * 0.02, n),
                "other": rng.normal(size=n),
                "marker": rng.permutation(n) + {"a": 0, "b": 1000, "c": 2000}[k],  # único
            }
        ).to_csv(tmp_path / f"{k}.csv", index=False)
    return inspect_files([tmp_path])["recipe"]


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory) -> Path:
    """Carpeta con 3 CSV sintéticos (no hay datos reales en este repositorio)."""
    import numpy as np
    import pandas as pd

    d = tmp_path_factory.mktemp("synthetic_data")
    rng = np.random.default_rng(0)
    for name, base in (("device_a", 100.0), ("device_b", 300.0), ("device_c", 700.0)):
        n = 400
        pd.DataFrame(
            {
                "power": base + rng.normal(0, base * 0.03, n),
                "current": base / 120 + rng.normal(0, 0.02, n),
                "time": pd.date_range("2026-01-01", periods=n, freq="s"),
            }
        ).to_csv(d / f"{name}.csv", index=False)
    return d
