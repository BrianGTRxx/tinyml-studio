"""Matriz de entrada: numéricas tal cual, categóricas codificadas, etiquetas a índices.

Codificación categórica (clave ``encoding`` de la receta, por defecto ``onehot``):

- ``onehot``: una columna 0/1 por categoría (``col=valor``). Desconocida/falta -> todo ceros.
- ``dummy``: k-1 columnas; la primera categoría alfabética es la referencia y se omite (evita la
  trampa de la variable ficticia en modelos lineales). Desconocida/falta -> todo ceros, que es
  EXACTAMENTE lo mismo que la categoría de referencia (no se distinguen).
- ``ordinal``: UNA columna numérica con el índice alfabético (``col#ordinal``). Desconocida -> -1,
  falta -> índice de la moda de train. Implica un orden que no existe: razonable para árboles,
  arriesgado para modelos lineales, knn, svm o mlp (ver ``encoding_advice``).

El codificador se ajusta SOLO con train y es serializable (JSON): el mismo mapeo debe
reproducirse en el firmware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tinyml_studio.i18n import tr

MAX_CATEGORIES = 20
ENCODING_METHODS = ("onehot", "dummy", "ordinal")

# Familias de modelos (por nombre) para los consejos de codificación.
_ORDER_SENSITIVE = ("linear", "logreg", "ridge", "lasso", "knn", "svm", "svc", "svr", "mlp")
_TREE_LIKE = ("tree", "forest", "boost", "gbm", "xgb")


def normalize_encoding(cfg: dict | None) -> dict:
    """Valida la clave ``encoding`` de la receta y devuelve la forma completa (ausente = onehot)."""
    if cfg is None:
        return {"default": "onehot", "overrides": {}, "max_categories": MAX_CATEGORIES}
    if not isinstance(cfg, dict):
        raise ValueError(tr("'encoding' debe ser un objeto", "'encoding' must be an object"))
    unknown = set(cfg) - {"default", "overrides", "max_categories"}
    if unknown:
        raise ValueError(
            tr(
                "claves desconocidas en 'encoding': {k}",
                "unknown keys in 'encoding': {k}",
                k=sorted(unknown),
            )
        )
    default = cfg.get("default", "onehot")
    overrides = cfg.get("overrides") or {}
    mx = cfg.get("max_categories", MAX_CATEGORIES)

    def bad_method(m):
        return tr(
            "método de codificación inválido: {m!r} (use {ok})",
            "invalid encoding method: {m!r} (use {ok})",
            m=m,
            ok=list(ENCODING_METHODS),
        )

    if default not in ENCODING_METHODS:
        raise ValueError(bad_method(default))
    if not isinstance(overrides, dict):
        raise ValueError(
            tr("'encoding.overrides' debe ser un objeto", "'encoding.overrides' must be an object")
        )
    for m in overrides.values():
        if m not in ENCODING_METHODS:
            raise ValueError(bad_method(m))
    if isinstance(mx, bool) or not isinstance(mx, int) or mx < 1:
        raise ValueError(
            tr(
                "'encoding.max_categories' debe ser un entero >= 1",
                "'encoding.max_categories' must be an integer >= 1",
            )
        )
    return {"default": default, "overrides": dict(overrides), "max_categories": mx}


@dataclass
class FeatureEncoder:
    numeric: list[str] = field(default_factory=list)
    categorical: dict[str, list[str]] = field(default_factory=dict)
    fill: dict[str, float] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)  # solo clasificación
    skipped: dict[str, str] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)  # columna categórica -> método
    mode_index: dict[str, int] = field(default_factory=dict)  # moda de train (ordinal: falta)

    def method(self, c: str) -> str:
        return self.methods.get(c, "onehot")

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        features: list[str],
        target: str | None,
        task: str,
        encoding: dict | None = None,
    ):
        cfg = normalize_encoding(encoding)
        max_cats = cfg["max_categories"]
        enc = cls()
        for c in features:
            s = train[c]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                f = pd.to_numeric(s, errors="coerce").astype("float64")
                if f.notna().sum() == 0:
                    enc.skipped[c] = tr("todo nulo en train", "all null in train")
                    continue
                enc.numeric.append(c)
                enc.fill[c] = float(f.median())
            elif pd.api.types.is_datetime64_any_dtype(s):
                enc.skipped[c] = tr(
                    "fecha/hora no es feature directa", "date/time is not a direct feature"
                )
            else:
                cats = sorted(s.dropna().astype(str).unique())
                if not cats or len(cats) > max_cats:
                    enc.skipped[c] = tr(
                        "{n} categorías (máx. {m})",
                        "{n} categories (max. {m})",
                        n=len(cats),
                        m=max_cats,
                    )
                    continue
                enc.categorical[c] = cats
                enc.methods[c] = cfg["overrides"].get(c, cfg["default"])
                counts = s.dropna().astype(str).value_counts()
                enc.mode_index[c] = int(np.argmax([counts.get(k, 0) for k in cats]))
        if task == "classification":
            enc.classes = sorted(train[target].dropna().astype(str).unique())
        return enc

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric)
        for c, cats in self.categorical.items():
            m = self.method(c)
            if m == "ordinal":
                names.append(f"{c}#ordinal")
            else:
                names += [f"{c}={k}" for k in (cats[1:] if m == "dummy" else cats)]
        return names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.numeric:
            f = pd.to_numeric(df[c], errors="coerce").astype("float64")
            cols.append(f.fillna(self.fill[c]).to_numpy())
        for c, cats in self.categorical.items():
            v = df[c].astype("string")
            m = self.method(c)
            if m == "ordinal":
                idx = {k: i for i, k in enumerate(cats)}
                code = [self.mode_index.get(c, 0) if pd.isna(x) else idx.get(str(x), -1) for x in v]
                cols.append(np.asarray(code, dtype="float64"))
                continue
            for k in cats[1:] if m == "dummy" else cats:
                cols.append((v == k).fillna(False).to_numpy(dtype="float64"))  # desconocida -> 0
        if not cols:
            raise ValueError(tr("no hay features utilizables", "no usable features"))
        return np.column_stack(cols).astype("float32")

    def encode_target(self, y: pd.Series) -> np.ndarray:
        idx = {k: i for i, k in enumerate(self.classes)}
        unknown = set(y.astype(str)) - set(idx)
        if unknown:
            raise ValueError(
                tr(
                    "clases ausentes en train: {c}",
                    "classes missing in train: {c}",
                    c=sorted(unknown),
                )
            )
        return y.astype(str).map(idx).to_numpy(dtype="int64")

    def to_dict(self) -> dict:
        return {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "fill": self.fill,
            "classes": self.classes,
            "skipped": self.skipped,
            "methods": self.methods,
            "mode_index": self.mode_index,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureEncoder:
        cats = d["categorical"]
        methods = {c: d.get("methods", {}).get(c, "onehot") for c in cats}  # antiguo -> onehot
        mode = {c: int(d.get("mode_index", {}).get(c, 0)) for c in cats}
        return cls(d["numeric"], cats, d["fill"], d["classes"], d.get("skipped", {}), methods, mode)


def _family(model: str) -> str:
    n = model.lower()
    if any(t in n for t in _TREE_LIKE):
        return "tree"
    if any(t in n for t in _ORDER_SENSITIVE):
        return "order_sensitive"
    return "other"


def encoding_advice(
    source: dict | FeatureEncoder,
    model_family: str,
    categorical: dict[str, list[str]] | None = None,
) -> list[str]:
    """Avisos bilingües sobre la codificación categórica para un modelo.

    ``source``: receta (con clave ``encoding``), el propio dict ``encoding`` o un
    ``FeatureEncoder`` ajustado. ``categorical`` (columna -> categorías) permite avisar de
    anchuras grandes; se toma del codificador si se pasa uno. ``dummy`` con árboles es inocuo
    (sin aviso).
    """
    if isinstance(source, FeatureEncoder):
        categorical = categorical or source.categorical
        cfg = {"default": "onehot", "overrides": source.methods}
        cols = list(source.categorical)
    else:
        cfg = normalize_encoding(source.get("encoding") if "encoding" in source else source)
        cols = list(categorical or cfg["overrides"])
    methods = {c: cfg["overrides"].get(c, cfg["default"]) for c in cols}
    fam = _family(model_family)
    out: list[str] = []
    ordinal = [c for c, m in methods.items() if m == "ordinal"]
    if ordinal and fam == "order_sensitive":
        out.append(
            tr(
                "{model}: Codificación ordinal en {cols} impone un orden inexistente entre "
                "categorías; usa onehot/dummy para modelos lineales, knn, svm o mlp",
                "{model}: Ordinal encoding on {cols} imposes a non-existent order between "
                "categories; prefer onehot/dummy for linear, knn, svm or mlp models",
                model=model_family,
                cols=ordinal,
            )
        )
    if categorical:
        wide = [c for c, m in methods.items() if m == "onehot" and len(categorical.get(c, ())) > 10]
        if wide:
            out.append(
                tr(
                    "{model}: Onehot en {cols} crea muchas columnas (>10 categorías); considera "
                    "ordinal (árboles), agrupar categorías o bajar max_categories",
                    "{model}: Onehot on {cols} creates many columns (>10 categories); consider "
                    "ordinal (trees), grouping categories or lowering max_categories",
                    model=model_family,
                    cols=wide,
                )
            )
    return out
