"""Pasos de limpieza serializables. Regla de oro: TODO ajuste (medianas, límites, escalas)
se calcula solo con el conjunto de entrenamiento y se guarda en ``state``.

Cada paso se describe como ``{"kind", "params", "state", "why"}`` (JSON puro).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from tinyml_studio.i18n import tr

REGISTRY: dict[str, type[Step]] = {}


def register(cls):
    REGISTRY[cls.kind] = cls
    return cls


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


class Step:
    kind: ClassVar[str] = ""
    # "all": se aplica a train/val/test; "train_only": quita filas solo del entrenamiento
    row_policy: ClassVar[str] = "all"

    def __init__(self, why: str = "", state: dict | None = None, **params):
        self.params = params
        self.why = why
        self.state: dict = state or {}

    def fit(self, df: pd.DataFrame) -> Step:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.params.get("columns", []) if c in df.columns]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "params": _jsonable(self.params),
            "state": _jsonable(self.state),
            "why": self.why,
        }

    @staticmethod
    def from_dict(d: dict) -> Step:
        if d["kind"] not in REGISTRY:
            raise ValueError(
                tr(
                    "paso de limpieza desconocido: {k}",
                    "unknown cleaning step: {k}",
                    k=d["kind"],
                )
            )
        return REGISTRY[d["kind"]](why=d.get("why", ""), state=d.get("state"), **d["params"])


@register
class ReplaceValues(Step):
    """Valores centinela (-999, 9999...) -> nulo."""

    kind = "replace_values"

    def transform(self, df):
        df = df.copy()
        for c in self._cols(df):
            f = _num(df[c])
            df[c] = f.mask(f.isin(self.params["values"]))
        return df


@register
class AbsurdToNaN(Step):
    """Valores fuera de [q01 - k·r, q99 + k·r] (r = q99 - q01) -> nulo; límites por grupo."""

    kind = "absurd_to_nan"

    def fit(self, df):
        k, by = self.params.get("factor", 20.0), self.params.get("by")
        bounds: dict[str, dict[str, list[float]]] = {}

        def b(f: pd.Series):
            f = f.dropna()
            if len(f) < 30:
                return None
            q01, q99 = f.quantile(0.01), f.quantile(0.99)
            r = q99 - q01
            return None if r <= 0 else [float(q01 - k * r), float(q99 + k * r)]

        for c in self._cols(df):
            f = _num(df[c])
            bounds[c] = {}
            if (g := b(f)) is not None:
                bounds[c]["__all__"] = g
            if by and by in df.columns:
                for key, sub in f.groupby(df[by], sort=False):
                    if (g := b(sub)) is not None:
                        bounds[c][str(key)] = g
        self.state = {"bounds": bounds}
        return self

    def transform(self, df):
        df, by = df.copy(), self.params.get("by")
        for c, bd in self.state.get("bounds", {}).items():
            if c not in df.columns or not bd:
                continue
            f = _num(df[c])
            lo = pd.Series(bd.get("__all__", [-np.inf, np.inf])[0], index=df.index)
            hi = pd.Series(bd.get("__all__", [-np.inf, np.inf])[1], index=df.index)
            if by and by in df.columns:
                keys = df[by].astype(str)
                for k, (l_, h_) in ((k, v) for k, v in bd.items() if k != "__all__"):
                    m = keys == k
                    lo[m], hi[m] = l_, h_
            df[c] = f.mask((f < lo) | (f > hi))
        return df


@register
class Impute(Step):
    """Imputa nulos con un estadístico calculado en train (mean/median/mode/constant/ffill)."""

    kind = "impute"

    def fit(self, df):
        strat, stats = self.params.get("strategy", "median"), {}
        for c in self._cols(df):
            s = df[c]
            if strat == "constant":
                stats[c] = self.params.get("fill_value", 0)
            elif strat == "mode" or not pd.api.types.is_numeric_dtype(s):
                m = s.dropna().mode()
                stats[c] = None if m.empty else m.iloc[0]
            elif strat == "mean":
                stats[c] = _num(s).mean()
            else:  # median y ffill (ffill usa la mediana como respaldo)
                stats[c] = _num(s).median()
        self.state = {"fill": {c: (None if pd.isna(v) else v) for c, v in stats.items()}}
        return self

    def transform(self, df):
        df, by = df.copy(), self.params.get("by")
        for c, v in self.state.get("fill", {}).items():
            if c not in df.columns:
                continue
            s = df[c]
            if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                s = _num(s)  # Int64 no admite una mediana fraccionaria
            if self.params.get("strategy") == "ffill":
                s = s.groupby(df[by]).ffill() if by and by in df.columns else s.ffill()
            if v is not None:
                s = s.fillna(v)
            df[c] = s
        return df


@register
class OutlierHandling(Step):
    """Atípicos por IQR o z-score; acción: recortar (clip) o poner nulo (nan)."""

    kind = "outliers"

    def fit(self, df):
        method, k, b = self.params.get("method", "iqr"), self.params.get("k", 3.0), {}
        for c in self._cols(df):
            f = _num(df[c]).dropna()
            if len(f) < 10:
                continue
            if method == "zscore":
                mu, sd = f.mean(), f.std()
                if sd > 0:
                    b[c] = [float(mu - k * sd), float(mu + k * sd)]
            else:
                q1, q3 = f.quantile(0.25), f.quantile(0.75)
                if q3 > q1:
                    b[c] = [float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))]
        self.state = {"bounds": b}
        return self

    def transform(self, df):
        df, action = df.copy(), self.params.get("action", "clip")
        for c, (lo, hi) in self.state.get("bounds", {}).items():
            if c in df.columns:
                f = _num(df[c])
                df[c] = f.clip(lo, hi) if action == "clip" else f.mask((f < lo) | (f > hi))
        return df


@register
class Scale(Step):
    """Escalado (standard / minmax / robust) ajustado solo con train.

    ``keep_sign``: columnas que NO se centran (centro 0): solo se divide por la escala, así un valor
    negativo sigue siendo negativo (p. ej. potencia reactiva).
    En min-max la escala es max(|min|, |max|).
    """

    kind = "scale"

    def fit(self, df):
        method, st = self.params.get("method", "standard"), {}
        keep = set(self.params.get("keep_sign") or [])
        for c in self._cols(df):
            f = _num(df[c]).dropna()
            if f.empty:
                continue
            if method == "minmax":
                a, b = f.min(), f.max()
            elif method == "robust":
                a, b = f.median(), f.quantile(0.75) - f.quantile(0.25)
            else:
                a, b = f.mean(), f.std(ddof=0)
            st[c] = {"center": float(a), "scale": float(b) if b and np.isfinite(b) else 1.0}
            if method == "minmax":
                st[c] = {"center": float(a), "scale": float(b - a) if b > a else 1.0}
                if c in keep:
                    m = max(abs(float(a)), abs(float(b)))
                    st[c] = {"center": 0.0, "scale": m if m > 0 else 1.0}
            elif c in keep:
                st[c]["center"] = 0.0
        self.state = {"scale": st}
        return self

    def transform(self, df):
        df = df.copy()
        for c, p in self.state.get("scale", {}).items():
            if c in df.columns:
                df[c] = (_num(df[c]) - p["center"]) / p["scale"]
        return df


@register
class DropColumns(Step):
    kind = "drop_columns"

    def transform(self, df):
        return df.drop(columns=self._cols(df))


@register
class DropDuplicates(Step):
    """Quita filas duplicadas SOLO del entrenamiento (val/test conservan su forma real)."""

    kind = "drop_duplicates"
    row_policy = "train_only"

    def transform(self, df):
        ignore = set(self.params.get("ignore", []))
        sub = [c for c in df.columns if c not in ignore]
        return df.drop_duplicates(subset=sub)


_OPS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


@register
class FilterRule(Step):
    """Regla de filas (p. ej. ``col > valor``). Se aplica a todos los splits."""

    kind = "filter_rule"

    def transform(self, df):
        op = self.params["op"]
        if op not in _OPS:
            raise ValueError(tr("operador no soportado: {op}", "unsupported operator: {op}", op=op))
        col = self.params["column"]
        keep = _OPS[op](_num(df[col]), self.params["value"]) & _num(df[col]).notna()
        return df[keep]


STATEFUL = {"absurd_to_nan", "impute", "outliers", "scale"}


class Pipeline:
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.fitted = False

    @classmethod
    def from_list(cls, items: list[dict], *, fitted: bool = False) -> Pipeline:
        """``fitted=True`` solo si ``items`` trae el ``state`` ya ajustado con train."""
        p = cls([Step.from_dict(d) for d in items])
        p.fitted = fitted
        return p

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.steps]

    def fit(self, train: pd.DataFrame) -> Pipeline:
        cur = train
        for s in self.steps:
            s.fit(cur)
            cur = s.transform(cur)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, *, is_train: bool = False) -> pd.DataFrame:
        if not self.fitted and any(s.kind in STATEFUL for s in self.steps):
            raise RuntimeError(
                tr(
                    "el pipeline no está ajustado: Llama a fit(train) primero",
                    "the pipeline is not fitted: Call fit(train) first",
                )
            )
        for s in self.steps:
            if s.row_policy == "train_only" and not is_train:
                continue
            df = s.transform(df)
        return df.reset_index(drop=True)
