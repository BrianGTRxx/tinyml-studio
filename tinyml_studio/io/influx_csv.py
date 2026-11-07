"""Importador de CSV exportados de InfluxDB (CSV anotado).

Un export de Influx empieza con líneas de anotación (``#group``, ``#datatype``,
``#default``) y después la cabecera real; la primera columna no tiene nombre
(índice de Influx). Para el dataset de ejemplo son 3 líneas (``skiprows=3``);
aquí se detectan contando las líneas iniciales que empiezan con ``#``.
Un CSV normal (sin anotaciones) también se admite.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tinyml_studio.i18n import tr

# Tipos de Influx (#datatype) -> dtype de pandas
_DTYPES = {
    "double": "float64",
    "long": "Int64",
    "unsignedLong": "Int64",
    "boolean": "boolean",
    "string": "string",
}
_TIME_TYPES = {"dateTime:RFC3339", "dateTime:RFC3339Nano"}


@dataclass
class InfluxCSV:
    """Resultado de importar un CSV: datos + metadatos de la importación."""

    df: pd.DataFrame
    path: str
    annotation_rows: int
    separator: str
    influx_types: dict[str, str] = field(default_factory=dict)
    encoding: str = "utf-8"

    @property
    def is_influx(self) -> bool:
        return self.annotation_rows > 0


def _read_head(path: Path, encoding: str, max_lines: int = 50) -> list[str]:
    lines: list[str] = []
    with path.open(encoding=encoding, newline="") as fh:
        for line in fh:
            lines.append(line)
            if len(lines) >= max_lines:
                break
    return lines


def _count_annotations(lines: list[str]) -> int:
    n = 0
    for line in lines:
        if line.startswith("#"):
            n += 1
        else:
            break
    return n


def _detect_separator(header_line: str) -> str:
    try:
        return csv.Sniffer().sniff(header_line, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_influx_csv(path: str | Path, *, parse_time: bool = True) -> InfluxCSV:
    """Lee un CSV de InfluxDB (o un CSV simple) con tipos y fechas correctos.

    - Detecta las líneas ``#...`` iniciales (``skiprows``) y usa ``#datatype``
      para tipar columnas (double→float64, long→Int64 nulable, string→string).
    - Descarta la columna índice sin nombre.
    - Convierte las columnas ``dateTime:*`` a datetime con zona UTC.
    - Detecta el separador a partir de la cabecera.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    # Kaggle y Excel suelen exportar en latin-1/cp1252: utf-8 primero, luego latin-1.
    encoding = "utf-8-sig"
    try:
        head = _read_head(p, encoding)
        with p.open(encoding=encoding) as fh:
            while fh.read(1 << 20):
                pass
    except UnicodeDecodeError:
        encoding = "latin-1"
        head = _read_head(p, encoding)
    n_ann = _count_annotations(head)
    if n_ann >= len(head):
        raise ValueError(
            tr(
                "{f}: El archivo solo tiene anotaciones, sin cabecera",
                "{f}: The file contains only annotations, no header",
                f=p.name,
            )
        )
    header_line = head[n_ann]
    sep = _detect_separator(header_line)

    columns = next(csv.reader([header_line], delimiter=sep))
    influx_types: dict[str, str] = {}
    for line in head[:n_ann]:
        if line.startswith("#datatype"):
            types = next(csv.reader([line], delimiter=sep))
            influx_types = {c: t for c, t in zip(columns, types, strict=False) if c and t}

    dtype = {c: _DTYPES[t] for c, t in influx_types.items() if t in _DTYPES}
    df = pd.read_csv(p, skiprows=n_ann, sep=sep, dtype=dtype, encoding=encoding)

    # Columna índice de Influx: cabecera vacía -> "Unnamed: 0"
    if len(df.columns) and (columns[0] == "" or str(df.columns[0]).startswith("Unnamed")):
        df = df.drop(columns=df.columns[0])

    if parse_time:
        for col, t in influx_types.items():
            if t in _TIME_TYPES and col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            try:
                df["time"] = pd.to_datetime(df["time"], utc=True)
            except (ValueError, TypeError):
                pass

    return InfluxCSV(
        df=df,
        path=str(p),
        annotation_rows=n_ann,
        separator=sep,
        influx_types=influx_types,
        encoding=encoding,
    )
