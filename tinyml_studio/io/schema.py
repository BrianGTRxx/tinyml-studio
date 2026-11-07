"""Clasificación de columnas del dataset de ejemplo.

Solo nombres de columnas ya presentes en los CSV; no contiene fórmulas ni calibración.
"""

from __future__ import annotations

import pandas as pd

# Columnas que son telemetría/metadatos, no features (DATASET.md).
METADATA_COLUMNS = frozenset(
    {
        "proyecto",
        "sensor",
        "sensor_id",
        "dispositivo",
        "dispositivo_iot",
        "ubicacion",
        "rssi_wifi",
        "memoria_libre",
        "memoria_total",
        "memoria_usada",
        "uptime",
        "secuencia",
        "time",
        "fase",
        "tipo_medicion",
        "fp_estado",
        "electrodomestico",
        "fabricante",
        "referencia",
        "anio_modelo",
        "modo_operacion",
    }
)


def detect_schema(df: pd.DataFrame) -> str:
    """Devuelve ``"legacy"`` (esquema viejo con ``electrodomestico``), ``"nilm_testing"``
    (esquema con ``proyecto``/``thd``) o ``"unknown"``."""
    cols = set(df.columns)
    if "electrodomestico" in cols:
        return "legacy"
    if {"proyecto", "thd"} <= cols:
        return "nilm_testing"
    return "unknown"


def candidate_features(df: pd.DataFrame) -> list[str]:
    """Columnas numéricas que no son metadatos."""
    return [
        c for c in df.columns if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
