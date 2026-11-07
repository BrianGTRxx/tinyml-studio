"""Partición train/val/test por bloques temporales contiguos (sin fuga entre filas vecinas).

Las filas consecutivas de una sesión son casi idénticas; partirlas al azar filtra
información de train a test. Aquí cada grupo (p. ej. un aparato/archivo) se ordena
por tiempo y se corta en tres tramos contiguos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tinyml_studio.i18n import tr


def split_temporal_blocks(
    df: pd.DataFrame,
    *,
    group_column: str | None = None,
    time_column: str | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, np.ndarray]:
    """Devuelve ``{"train","val","test"}`` con posiciones (``iloc``) de ``df``."""
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9 or min(ratios) < 0:
        raise ValueError(
            tr(
                "ratios debe tener 3 valores >= 0 que sumen 1",
                "ratios must have 3 values >= 0 that sum to 1",
            )
        )
    out: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    pos = np.arange(len(df))
    groups = (
        [pos]
        if group_column is None
        else [pos[idx] for idx in df.groupby(group_column, sort=True).indices.values()]
    )
    for g in groups:
        if time_column is not None:
            g = g[np.argsort(df[time_column].to_numpy()[g], kind="stable")]
        n = len(g)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        out["train"].append(g[:n_train])
        out["val"].append(g[n_train : n_train + n_val])
        out["test"].append(g[n_train + n_val :])
    return {k: np.concatenate(v) if v else np.array([], dtype=int) for k, v in out.items()}
