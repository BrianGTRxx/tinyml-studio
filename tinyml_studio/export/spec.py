"""Qué hace el dispositivo antes del modelo: la limpieza ajustada con train, traducida a operaciones
simples de coma flotante (float32) que se generan en C y que aquí se emulan en numpy.

Cada categórica se expande según su método: onehot (k columnas), dummy (k-1, sin la referencia) u
ordinal (1 columna con el índice; desconocida -> -1).

Entradas del dispositivo: ``num[N_NUM]`` (valores crudos de las columnas numéricas, NaN si falta)
y ``cat[N_CAT]`` (código de categoría: índice >= 0, -1 = falta -> moda de train, -2 = desconocida).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tinyml_studio.clean.steps import Pipeline
from tinyml_studio.i18n import tr
from tinyml_studio.train.features import FeatureEncoder

F32 = np.float32


@dataclass
class DeviceSpec:
    num_names: list[str]
    ops: dict[str, list[tuple]]  # columna -> [("replace", [v..]) | ("range_nan", lo, hi) | ...]
    cat_names: list[str]
    cat_values: dict[str, list[str]]
    cat_fill: dict[str, int]  # código a usar cuando falta (-1 = ninguna: todo ceros)
    feature_names: list[str]
    classes: list[str]
    warnings: list[str] = field(default_factory=list)
    cat_methods: dict[str, str] = field(default_factory=dict)  # onehot | dummy | ordinal

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def build_spec(pipeline: Pipeline, enc: FeatureEncoder) -> DeviceSpec:
    ops: dict[str, list[tuple]] = {c: [] for c in enc.numeric}
    cat_fill = {
        c: (enc.mode_index.get(c, 0) if enc.method(c) == "ordinal" else -1) for c in enc.categorical
    }
    warn: list[str] = []
    per_file: list[str] = []
    for s in pipeline.steps:
        cols = [c for c in s.params.get("columns", [])]
        if s.kind == "replace_values":
            for c in cols:
                if c in ops:
                    ops[c].append(("replace", [float(v) for v in s.params["values"]]))
        elif s.kind == "absurd_to_nan":
            for c, bd in s.state.get("bounds", {}).items():
                if c in ops and "__all__" in bd:
                    ops[c].append(("range_nan", bd["__all__"][0], bd["__all__"][1]))
                    if len(bd) > 1:
                        per_file.append(c)
        elif s.kind == "outliers":
            for c, (lo, hi) in s.state.get("bounds", {}).items():
                if c in ops:
                    kind = "clip" if s.params.get("action", "clip") == "clip" else "range_nan"
                    ops[c].append((kind, lo, hi))
        elif s.kind == "impute":
            for c, v in s.state.get("fill", {}).items():
                if c in ops and v is not None and not isinstance(v, str):
                    ops[c].append(("fill", float(v)))
                elif c in cat_fill and v is not None:
                    cats = enc.categorical[c]
                    cat_fill[c] = cats.index(str(v)) if str(v) in cats else -1
        elif s.kind == "scale":
            for c, p in s.state.get("scale", {}).items():
                if c in ops:
                    ops[c].append(("scale", p["center"], p["scale"]))
        elif s.kind == "filter_rule":
            warn.append(
                tr(
                    "'filter_rule' ({c} {o} {v}) define qué filas son válidas: El firmware debe "
                    "aplicar la misma condición antes de llamar a tm_predict",
                    "'filter_rule' ({c} {o} {v}) defines which rows are valid: The firmware must "
                    "apply the same condition before calling tm_predict",
                    c=s.params.get("column"),
                    o=s.params.get("op"),
                    v=s.params.get("value"),
                )
            )
        elif s.kind not in ("drop_duplicates", "drop_columns"):
            warn.append(
                tr(
                    "paso '{k}' sin equivalente en el dispositivo",
                    "step '{k}' has no equivalent on the device",
                    k=s.kind,
                )
            )
    if per_file:
        warn.append(
            tr(
                "los límites por archivo de 'absurd_to_nan' no existen en el dispositivo (no hay "
                "'archivo'); se usa el límite global de train en: ",
                "the per-file limits of 'absurd_to_nan' do not exist on the device (there is no "
                "'file'); the global train limit is used for: ",
            )
            + ", ".join(per_file)
        )
    for c in enc.numeric:  # relleno final del codificador (mediana de train, en unidades finales)
        ops[c].append(("fill", float(enc.fill[c])))
    return DeviceSpec(
        num_names=list(enc.numeric),
        ops=ops,
        cat_names=list(enc.categorical),
        cat_values={c: list(v) for c, v in enc.categorical.items()},
        cat_fill=cat_fill,
        feature_names=enc.feature_names,
        classes=list(enc.classes),
        warnings=warn,
        cat_methods={c: enc.method(c) for c in enc.categorical},
    )


def raw_inputs(spec: DeviceSpec, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Valores crudos (antes de limpiar) en el formato del dispositivo."""
    num = (
        np.column_stack(
            [
                pd.to_numeric(raw[c], errors="coerce").astype("float64").to_numpy()
                for c in spec.num_names
            ]
        )
        if spec.num_names
        else np.zeros((len(raw), 0))
    )
    cats = []
    for c in spec.cat_names:
        idx = {k: i for i, k in enumerate(spec.cat_values[c])}
        col = raw[c].astype("string")
        cats.append([-1 if pd.isna(v) else idx.get(str(v), -2) for v in col])
    cat = np.array(cats, dtype="int64").T if cats else np.zeros((len(raw), 0), dtype="int64")
    return num, cat


def emulate_features(spec: DeviceSpec, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """Réplica numpy float32 de ``tm_prepare`` (mismas operaciones, mismo orden)."""
    cols = []
    for j, c in enumerate(spec.num_names):
        v = num[:, j].astype(F32)
        for op in spec.ops[c]:
            if op[0] == "replace":
                for x in op[1]:
                    v = np.where(v == F32(x), F32(np.nan), v)
            elif op[0] == "range_nan":
                v = np.where((v < F32(op[1])) | (v > F32(op[2])), F32(np.nan), v)
            elif op[0] == "clip":
                v = np.where(v < F32(op[1]), F32(op[1]), np.where(v > F32(op[2]), F32(op[2]), v))
            elif op[0] == "fill":
                v = np.where(np.isnan(v), F32(op[1]), v)
            elif op[0] == "scale":
                v = ((v - F32(op[1])) / F32(op[2])).astype(F32)
        cols.append(v.astype(F32))
    for j, c in enumerate(spec.cat_names):
        code = cat[:, j].copy()
        code = np.where(code == -1, spec.cat_fill[c], code)
        m = spec.cat_methods.get(c, "onehot")
        if m == "ordinal":  # desconocida (-2) -> -1; la falta ya se rellenó con la moda
            cols.append(np.where(code < 0, -1, code).astype(F32))
            continue
        for k in range(1 if m == "dummy" else 0, len(spec.cat_values[c])):
            cols.append((code == k).astype(F32))
    return np.column_stack(cols).astype(F32)
