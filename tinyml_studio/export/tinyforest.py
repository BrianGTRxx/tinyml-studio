"""Exportación compacta de árboles y bosques de scikit-learn a C99 (``tinyforest``).

Combina técnicas CONOCIDAS (ver ``docs/research/TINYFOREST.md``) y mide cada una por separado:

* ``layout="ref"``: solo nodos internos (las hojas son referencias con un bit de marca), hijos
  izquierdo/derecho por índice absoluto. ``layout="dfs"``: todos los nodos en preorden, hijo
  izquierdo implícito (``i + 1``) y desplazamiento relativo al derecho.
* Tabla de cortes por característica (los umbrales distintos que usa el bosque, ordenados): el
  nodo guarda el índice (``uint8``/``uint16``) en vez de un ``float32``. Con ``quant_input=True`` la
  entrada se cuantiza con búsqueda binaria y los nodos comparan enteros; la comparación
  ``x <= t`` se conserva EXACTAMENTE (demostración en el documento y prueba en los tests).
* ``dag=True``: hash-consing de subárboles idénticos entre todos los árboles (sin pérdida).
* Hojas: ``f32`` (probabilidades float32 deduplicadas, sin pérdida respecto al export actual),
  ``u8`` (probabilidades a 8 bits, con pérdida), ``vote`` (voto duro, con pérdida respecto al
  voto suave de scikit-learn); regresión: ``f32`` o ``u16`` (con pérdida).
* ``max_bins=K``: reduce los cortes por característica a ≤ K (con pérdida).

Hay un evaluador de referencia en numpy cuya aritmética es la misma que la del C generado
(float32 acumulado en el mismo orden), de modo que C y numpy coinciden bit a bit.
No es hardware verificado: ver el documento para los límites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tinyml_studio.i18n import tr

from .cgen import cf

LAYOUTS = ("ref", "dfs")
CLS_LEAVES = ("f32", "u8", "vote")
REG_LEAVES = ("f32", "u16")


# ------------------------------------------------------------------ utilidades
def _utype(maxval: int) -> tuple[str, np.dtype, int]:
    if maxval < 256:
        return "uint8_t", np.dtype(np.uint8), 1
    if maxval < 65536:
        return "uint16_t", np.dtype(np.uint16), 2
    return "uint32_t", np.dtype(np.uint32), 4


def _itype(lo: int, hi: int) -> tuple[str, int]:
    for name, a, b, n in (
        ("uint8_t", 0, 255, 1),
        ("int8_t", -128, 127, 1),
        ("uint16_t", 0, 65535, 2),
        ("int16_t", -32768, 32767, 2),
        ("int32_t", -(2**31), 2**31 - 1, 4),
    ):
        if lo >= a and hi <= b:
            return name, n
    raise ValueError(tr("umbral entero fuera de rango", "integer threshold out of range"))


def _thr32(t: np.ndarray) -> np.ndarray:
    """Mayor float32 <= t (vectorizado; igual que ``cgen.thr_f32``)."""
    f = t.astype(np.float32)
    bad = f.astype(np.float64) > t
    f[bad] = np.nextafter(f[bad], np.float32(-np.inf))
    return f


def _fmt(values: np.ndarray, per_line: int) -> str:
    return ",\n".join(
        "  " + ", ".join(str(v) for v in values[i : i + per_line])
        for i in range(0, len(values), per_line)
    )


class _Arrays:
    """Acumula arrays ``static const`` y cuenta sus bytes (uno a uno, para el desglose)."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.bytes: dict[str, int] = {}

    def uint(self, name: str, a: np.ndarray, ctype: str, itemsize: int) -> None:
        a = np.asarray(a).ravel()
        if len(a) == 0:
            a = np.zeros(1, dtype=np.int64)
        self.bytes[name] = itemsize * len(a)
        self.text.append(
            f"static const {ctype} {name}[{len(a)}] = {{\n{_fmt(a.astype(np.int64), 16)}\n}};"
        )

    def flt(self, name: str, a: np.ndarray) -> None:
        a = np.asarray(a, dtype=np.float32).ravel()
        if len(a) == 0:
            a = np.zeros(1, dtype=np.float32)
        self.bytes[name] = 4 * len(a)
        self.text.append(
            f"static const float {name}[{len(a)}] = {{\n"
            + _fmt(np.array([cf(x) for x in a], dtype=object), 8)
            + "\n};"
        )


# ------------------------------------------------------------------ lectura de scikit-learn
@dataclass
class _Tree:
    feature: np.ndarray  # -1 en hojas
    thr: np.ndarray  # float32 (umbral ya ajustado a float32)
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray  # (n_nodos, k) float64: recuento/proporción por clase, o valor de regresión


def _read_tree(est: Any) -> _Tree:
    T = est.tree_
    v = T.value
    if v.ndim == 3:
        if v.shape[1] != 1:
            raise ValueError(
                tr(
                    "no se admiten modelos multisalida",
                    "multi-output models are not supported",
                )
            )
        v = v[:, 0, :]
    thr = np.zeros(T.node_count, dtype=np.float32)
    internal = T.children_left != -1
    thr[internal] = _thr32(T.threshold[internal])
    return _Tree(
        T.feature.astype(np.int64), thr, T.children_left.astype(np.int64),
        T.children_right.astype(np.int64), np.asarray(v, dtype=np.float64),
    )  # fmt: skip


def _preorder(t: _Tree) -> list[int]:
    out, stack = [], [0]
    while stack:
        i = stack.pop()
        out.append(i)
        if t.left[i] != -1:
            stack.append(int(t.right[i]))
            stack.append(int(t.left[i]))
    return out


def _snap(trees: list[_Tree], max_bins: int) -> None:
    """Reduce a ≤ max_bins los cortes distintos por característica (con pérdida)."""
    feats = np.concatenate([t.feature[t.left != -1] for t in trees])
    thrs = np.concatenate([t.thr[t.left != -1] for t in trees])
    for j in np.unique(feats):
        sel = feats == j
        u, cnt = np.unique(thrs[sel], return_counts=True)
        if len(u) <= max_bins:
            continue
        cum = np.cumsum(cnt) / cnt.sum()  # cuantiles ponderados por uso
        pos = np.minimum(np.searchsorted(cum, (np.arange(max_bins) + 0.5) / max_bins), len(u) - 1)
        kept = np.unique(u[pos])
        for t in trees:
            m = (t.left != -1) & (t.feature == j)
            x = t.thr[m]
            k = np.clip(np.searchsorted(kept, x), 1, len(kept) - 1) if len(kept) > 1 else None
            if k is None:
                t.thr[m] = kept[0]
            else:  # el corte conservado más cercano (en empate, el menor)
                lo, hi = kept[k - 1], kept[k]
                t.thr[m] = np.where(
                    np.abs(x.astype(np.float64) - lo) <= np.abs(hi.astype(np.float64) - x), lo, hi
                )


# ------------------------------------------------------------------ el modelo compacto
@dataclass
class TinyForest:
    task: str  # "classification" | "regression"
    n_features: int
    n_classes: int
    n_trees: int
    layout: str
    leaf: str  # f32 | u8 | vote | u16
    dag: bool
    quant_input: bool
    thr_table: bool
    integer_inputs: bool
    classes: np.ndarray | None
    # nodos
    feat: np.ndarray
    bin: np.ndarray  # índice de corte (si thr_table)
    thr: np.ndarray  # umbral float32 inline (si no thr_table)
    lref: np.ndarray  # layout ref: hijos; dfs: no se usa
    rref: np.ndarray  # layout ref: hijo derecho; dfs: desplazamiento relativo / carga de hoja
    roots: np.ndarray
    leaf_table: np.ndarray | None  # (L, k) según ``leaf``; None si la hoja es la clase
    cuts: np.ndarray  # float32, concatenados por característica
    off: np.ndarray  # n_features + 1
    leaf_scale: tuple[float, float] = (0.0, 0.0)  # (lo, step) para u16
    info: dict[str, Any] = field(default_factory=dict)

    # ---- tamaños
    def layout_arrays(self) -> dict[str, tuple[str, np.ndarray, int]]:
        """nombre -> (tipo C, array, bytes por elemento). Solo lo que el C emitido usa."""
        out: dict[str, tuple[str, np.ndarray, int]] = {}
        # dfs reserva el valor F como centinela de hoja
        ft = _utype(self.n_features if self.layout == "dfs" else max(self.n_features - 1, 0))
        out["tf_feat"] = (ft[0], self.feat, ft[2])
        if self.thr_table:
            nb = int(np.diff(self.off).max()) if self.n_features else 0
            bt = _utype(nb)
            out["tf_bin"] = (bt[0], self.bin, bt[2])
            out["tf_off"] = (_utype(len(self.cuts))[0], self.off, _utype(len(self.cuts))[2])
            out["tf_cut"] = ("float", self.cuts, 4)
        else:
            if self.integer_inputs:
                lo, hi = (
                    int(self.thr.min()) if len(self.thr) else 0,
                    int(self.thr.max() if len(self.thr) else 0),
                )
                ct, sz = _itype(lo, hi)
                out["tf_thr"] = (ct, self.thr.astype(np.int64), sz)
            else:
                out["tf_thr"] = ("float", self.thr, 4)
        rt = self._ref_type()
        if self.layout == "ref":
            out["tf_lc"] = (rt[0], self.lref, rt[2])
        out["tf_rc"] = (rt[0], self.rref, rt[2])
        if self.n_trees > 1:
            out["tf_root"] = (rt[0], self.roots, rt[2])
        if self.leaf_table is not None:
            lt = {"f32": "float", "u8": "uint8_t", "u16": "uint16_t"}[self.leaf]
            out["tf_leaf"] = (lt, self.leaf_table, {"float": 4, "uint8_t": 1, "uint16_t": 2}[lt])
        return out

    def _ref_type(self) -> tuple[str, np.dtype, int]:
        """Tipo de las referencias a nodo/hoja y de las raíces."""
        n_leaf = self.n_classes if self.leaf_table is None else len(self.leaf_table)
        if self.layout == "ref":
            if max(len(self.feat), n_leaf) >= (1 << 15) - 1:
                return "uint32_t", np.dtype(np.uint32), 4
            return "uint16_t", np.dtype(np.uint16), 2
        mx = max(
            int(self.rref.max()) if len(self.rref) else 0,
            int(self.roots.max()) if len(self.roots) else 0,
            n_leaf,
        )
        return _utype(mx)

    @property
    def data_bytes(self) -> int:
        return sum(max(a.size, 1) * sz for (_, a, sz) in self.layout_arrays().values())

    def breakdown(self) -> dict[str, int]:
        return {k: max(a.size, 1) * sz for k, (_, a, sz) in self.layout_arrays().items()}

    @property
    def ram_bytes(self) -> int:
        """Buffers de trabajo del modelo (sin el vector de características)."""
        C = self.n_classes
        if self.task == "regression":
            return 4 if self.n_trees > 1 else 0
        if self.leaf == "vote":
            return 0 if self.n_trees == 1 else C * (1 if self.n_trees < 256 else 2)
        acc = 4 * C if self.leaf == "f32" else C * (2 if self.n_trees * 255 < 65536 else 4)
        q = 0
        if self.quant_input and self.thr_table:
            q = self.n_features * _utype(int(np.diff(self.off).max()) if self.n_features else 0)[2]
        return acc + q

    # ---- evaluador de referencia (numpy), misma aritmética que el C
    def _quantize(self, X: np.ndarray) -> np.ndarray:
        q = np.zeros(X.shape, dtype=np.int64)
        for j in range(self.n_features):
            c = self.cuts[self.off[j] : self.off[j + 1]]
            if len(c):
                q[:, j] = np.searchsorted(c, X[:, j], side="left")  # NaN -> len(c): va a la derecha
        return q

    def leaf_ids(self, X: np.ndarray, use_quant: bool | None = None) -> np.ndarray:
        """(n, n_trees): id de hoja (o clase si ``leaf_table is None``) alcanzado en cada árbol."""
        X = np.asarray(X, dtype=np.float32)
        quant = self.quant_input if use_quant is None else use_quant
        if quant and not self.thr_table:
            raise ValueError("quant_input requiere thr_table")
        Q = self._quantize(X) if (quant and self.thr_table) else None
        n = len(X)
        out = np.zeros((n, self.n_trees), dtype=np.int64)
        for t in range(self.n_trees):
            r = np.full(n, int(self.roots[t]), dtype=np.int64)
            if self.layout == "ref":
                leaf_flag = self._leaf_flag()
                while True:
                    act = np.flatnonzero((r & leaf_flag) == 0)
                    if len(act) == 0:
                        break
                    i = r[act]
                    fj = self.feat[i].astype(np.int64)
                    x = X[act, fj]
                    if Q is not None:
                        go_left = Q[act, fj] <= self.bin[i]
                    elif self.thr_table:
                        go_left = x <= self.cuts[self.off[fj] + self.bin[i]]
                    else:
                        go_left = x <= self.thr[i]
                    r[act] = np.where(go_left, self.lref[i], self.rref[i])
                out[:, t] = r & ~leaf_flag
            else:
                leaf_sent = self.n_features
                while True:
                    act = np.flatnonzero(self.feat[r] != leaf_sent)
                    if len(act) == 0:
                        break
                    i = r[act]
                    fj = self.feat[i].astype(np.int64)
                    x = X[act, fj]
                    if Q is not None:
                        go_left = Q[act, fj] <= self.bin[i]
                    elif self.thr_table:
                        go_left = x <= self.cuts[self.off[fj] + self.bin[i]]
                    else:
                        go_left = x <= self.thr[i]
                    r[act] = np.where(go_left, i + 1, i + self.rref[i])
                out[:, t] = self.rref[r]
        return out

    def _leaf_flag(self) -> int:
        return 0x80000000 if self._ref_type()[2] == 4 else 0x8000

    def predict_index(self, X: np.ndarray, use_quant: bool | None = None) -> np.ndarray:
        """Clase (índice) o valor, con la misma aritmética y orden que el C."""
        ids = self.leaf_ids(X, use_quant)
        n, T = ids.shape
        if self.task == "classification":
            if self.leaf == "vote":
                if T == 1:
                    return ids[:, 0]
                votes = np.zeros((n, self.n_classes), dtype=np.int64)
                for t in range(T):
                    votes[np.arange(n), ids[:, t]] += 1
                return votes.argmax(1)
            if self.leaf == "f32":
                acc = np.zeros((n, self.n_classes), dtype=np.float32)
                for t in range(T):
                    acc = acc + self.leaf_table[ids[:, t]]
                return acc.argmax(1)
            acc = np.zeros((n, self.n_classes), dtype=np.int64)
            for t in range(T):
                acc += self.leaf_table[ids[:, t]]
            return acc.argmax(1)
        if self.leaf == "f32":
            if T == 1:
                return self.leaf_table[ids[:, 0], 0]
            acc = np.zeros(n, dtype=np.float32)
            for t in range(T):
                acc = acc + self.leaf_table[ids[:, t], 0]
            return acc / np.float32(T)
        lo, step = np.float32(self.leaf_scale[0]), np.float32(self.leaf_scale[1])
        if T == 1:
            return lo + self.leaf_table[ids[:, 0], 0].astype(np.float32) * step
        acc = np.zeros(n, dtype=np.int64)
        for t in range(T):
            acc += self.leaf_table[ids[:, t], 0]
        return lo + acc.astype(np.float32) * (step / np.float32(T))

    def predict(self, X: np.ndarray, use_quant: bool | None = None) -> np.ndarray:
        p = self.predict_index(X, use_quant)
        return self.classes[p] if self.task == "classification" and self.classes is not None else p

    # ---- C
    def to_c(self, argmax: str = "tm_argmax") -> tuple[str, int, int, dict[str, Any]]:
        """(código, bytes de datos, bytes de RAM, info); define ``tm_model(const float *f)``."""
        arrs = self.layout_arrays()
        A = _Arrays()
        for name, (ct, a, sz) in arrs.items():
            if ct == "float":
                A.flt(name, a)
            else:
                A.uint(name, a, ct, sz)
        rtype, _, _ = self._ref_type()
        T = self.n_trees
        clsf = self.task == "classification"
        L: list[str] = ["\n\n".join(A.text), ""]
        quant = self.quant_input and self.thr_table
        if self.layout == "ref":
            flag = "0x80000000u" if rtype == "uint32_t" else "0x8000u"
            L.append(f"#define TF_LEAF {flag}")
            cond = "!(r & TF_LEAF)"
            idx = "r"
            step = f"r = ({{cmp}}) ? tf_lc[{idx}] : tf_rc[{idx}];"
            payload = "(unsigned)(r & ~TF_LEAF)"
        else:
            cond = f"tf_feat[r] != {self.n_features}"
            idx = "r"
            step = f"r = ({{cmp}}) ? r + 1u : r + tf_rc[{idx}];"
            payload = "(unsigned)tf_rc[r]"
        if self.thr_table:
            cut = f"tf_cut[tf_off[tf_feat[{idx}]] + tf_bin[{idx}]]"
        else:
            cut = f"(float)tf_thr[{idx}]" if self.integer_inputs else f"tf_thr[{idx}]"
        if quant:
            cmp = f"q[tf_feat[{idx}]] <= tf_bin[{idx}]"
        else:
            cmp = f"f[tf_feat[{idx}]] <= {cut}"
        walk = f"    while ({cond}) {step.format(cmp=cmp)}\n"
        if self.n_trees > 1:
            head = f"  for (unsigned t = 0; t < {T}u; t++) {{\n    unsigned r = tf_root[t];\n"
        else:
            head = f"  {{\n    unsigned r = {int(self.roots[0])}u;\n"
        if quant:
            qt = _utype(int(np.diff(self.off).max()))[0]
            L.append(
                "static void tf_quant(const float *f, " + qt + " *q) {\n"
                f"  for (unsigned j = 0; j < {self.n_features}u; j++) {{\n"
                "    unsigned lo = 0, hi = tf_off[j + 1] - tf_off[j];\n"
                "    const float *c = &tf_cut[tf_off[j]];\n"
                "    while (lo < hi) {\n      unsigned mid = (lo + hi) >> 1;\n"
                "      if (!(f[j] <= c[mid])) lo = mid + 1u; else hi = mid;\n    }\n"
                f"    q[j] = ({qt})lo;\n  }}\n}}\n"
            )
            pre = f"  {qt} q[{self.n_features}];\n  tf_quant(f, q);\n"
        else:
            pre = ""
        if clsf:
            L.append(
                f"static int tm_model(const float *f) {{\n  (void)f;\n{pre}"
                + self._cls_body(head, walk, payload, argmax)
            )
        else:
            L.append(
                f"static float tm_model(const float *f) {{\n  (void)f;\n{pre}"
                + self._reg_body(head, walk, payload)
            )
        return "\n".join(L), sum(A.bytes.values()), self.ram_bytes, self._info(A.bytes)

    def _cls_body(self, head: str, walk: str, payload: str, argmax: str) -> str:
        C, T = self.n_classes, self.n_trees
        if self.leaf == "vote":
            if T == 1:
                return head + walk + f"    return (int){payload};\n  }}\n}}"
            vt = "uint8_t" if T < 256 else "uint16_t"
            return (
                f"  {vt} v[{C}];\n  for (unsigned c = 0; c < {C}u; c++) v[c] = 0;\n"
                + head
                + walk
                + f"    v[{payload}]++;\n  }}\n"
                f"  unsigned best = 0;\n  for (unsigned c = 1; c < {C}u; c++) if (v[c] > v[best]) "
                "best = c;\n  return (int)best;\n}"
            )
        if self.leaf == "f32":
            return (
                f"  float acc[{C}];\n  for (unsigned c = 0; c < {C}u; c++) acc[c] = 0.0f;\n"
                + head
                + walk
                + f"    const float *p = &tf_leaf[{payload} * {C}u];\n"
                f"    for (unsigned c = 0; c < {C}u; c++) acc[c] += p[c];\n  }}\n"
                f"  return {argmax}(acc, {C});\n}}"
            )
        at = "uint16_t" if T * 255 < 65536 else "uint32_t"
        return (
            f"  {at} acc[{C}];\n  for (unsigned c = 0; c < {C}u; c++) acc[c] = 0;\n"
            + head
            + walk
            + f"    const uint8_t *p = &tf_leaf[{payload} * {C}u];\n"
            f"    for (unsigned c = 0; c < {C}u; c++) acc[c] = ({at})(acc[c] + p[c]);\n  }}\n"
            f"  unsigned best = 0;\n  for (unsigned c = 1; c < {C}u; c++) if (acc[c] > acc[best]) "
            "best = c;\n  return (int)best;\n}"
        )

    def _reg_body(self, head: str, walk: str, payload: str) -> str:
        T = self.n_trees
        if self.leaf == "f32":
            if T == 1:
                return head + walk + f"    return tf_leaf[{payload}];\n  }}\n}}"
            return (
                "  float acc = 0.0f;\n" + head + walk
                + f"    acc += tf_leaf[{payload}];\n  }}\n  return acc / {cf(T)};\n}}"
            )  # fmt: skip
        lo, step = np.float32(self.leaf_scale[0]), np.float32(self.leaf_scale[1])
        if T == 1:
            return (
                head + walk + f"    return {cf(lo)} + (float)tf_leaf[{payload}] * {cf(step)};\n"
                "  }\n}"
            )
        return (
            "  uint32_t acc = 0;\n" + head + walk
            + f"    acc += tf_leaf[{payload}];\n  }}\n"
            f"  return {cf(lo)} + (float)acc * {cf(step / np.float32(T))};\n}}"
        )  # fmt: skip

    def _info(self, arr_bytes: dict[str, int]) -> dict[str, Any]:
        return {
            "trees": self.n_trees,
            "nodes": int(len(self.feat)),
            "leaf_rows": 0 if self.leaf_table is None else int(len(self.leaf_table)),
            "layout": self.layout,
            "leaf": self.leaf,
            "dag": self.dag,
            "quant_input": self.quant_input,
            "array_bytes": arr_bytes,
            **self.info,
        }


# ------------------------------------------------------------------ construcción
def _is_cls(est: Any) -> bool:
    return hasattr(est, "classes_")


def build(
    est: Any,
    *,
    layout: str = "ref",
    dag: bool = True,
    leaf: str = "f32",
    max_bins: int | None = None,
    quant_input: bool = False,
    thr_table: bool | None = True,
    integer_inputs: bool = False,
) -> TinyForest:
    """Convierte un DecisionTree/RandomForest/ExtraTrees (clasificador o regresor) de sklearn."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout debe ser uno de {LAYOUTS}")
    forest = hasattr(est, "estimators_")
    members = list(est.estimators_) if forest else [est]
    cls = _is_cls(est)
    if leaf not in (CLS_LEAVES if cls else REG_LEAVES):
        raise ValueError(
            tr(
                "hoja '{leaf}' no válida para esta tarea",
                "leaf '{leaf}' invalid for this task",
                leaf=leaf,
            )
        )
    if integer_inputs:
        if quant_input:
            raise ValueError("integer_inputs no usa quant_input")
        thr_table = False
    if thr_table is None:  # automático: la opción que ocupa menos bytes (quant_input exige tabla)
        kw = dict(layout=layout, dag=dag, leaf=leaf, max_bins=max_bins, quant_input=quant_input)
        a = build(est, thr_table=True, **kw)
        if quant_input:
            return a
        b = build(est, thr_table=False, **kw)
        return a if a.data_bytes <= b.data_bytes else b
    if quant_input and not thr_table:
        raise ValueError("quant_input requiere thr_table=True")
    if dag and layout == "dfs":
        raise ValueError(tr("dag requiere layout='ref'", "dag requires layout='ref'"))
    F = int(est.n_features_in_)
    C = len(est.classes_) if cls else 1
    if getattr(est, "n_outputs_", 1) != 1:
        raise ValueError(
            tr("no se admiten modelos multisalida", "multi-output models are not supported")
        )
    trees = [_read_tree(m) for m in members]
    if max_bins is not None:
        if max_bins < 2:
            raise ValueError("max_bins >= 2")
        _snap(trees, max_bins)
    if integer_inputs:  # con x entero, x <= t equivale a x <= floor(t)
        for t in trees:
            t.thr = np.floor(t.thr).astype(np.float32)
    if cls and not forest:
        leaf = "vote"  # un árbol: el argmax de la hoja es exacto y no necesita tabla
    scale = (0.0, 0.0)

    # ---- carga de cada nodo hoja, según el modo
    def payload_of(t: _Tree) -> np.ndarray:
        v = t.value
        if cls and leaf == "vote":
            return v.argmax(1).astype(np.int64)[:, None]
        if cls and leaf == "f32":
            return (v / v.sum(1, keepdims=True)).astype(np.float32)
        if cls:  # u8
            p = v / v.sum(1, keepdims=True)
            return np.clip(np.floor(p * 255.0 + 0.5), 0, 255).astype(np.uint8)
        if leaf == "f32":
            return v[:, :1].astype(np.float32)
        return v[:, :1]  # u16: se cuantiza abajo

    pay = [payload_of(t) for t in trees]
    if not cls and leaf == "u16":
        vals = np.concatenate([p[t.left == -1, 0] for p, t in zip(pay, trees, strict=True)])
        lo, hi = float(vals.min()), float(vals.max())
        step32 = np.float32((hi - lo) / 65535.0) if hi > lo else np.float32(0.0)
        scale = (float(np.float32(lo)), float(step32))
        pay = [
            (
                np.clip(np.floor((p - lo) / (hi - lo) * 65535.0 + 0.5), 0, 65535).astype(np.uint16)
                if hi > lo
                else np.zeros(p.shape, dtype=np.uint16)
            )
            for p in pay
        ]
    has_table = leaf != "vote"
    ldt = {"f32": np.float32, "u8": np.uint8, "u16": np.uint16}.get(leaf, np.int64)

    # ---- grafo
    leaf_keys: dict[bytes, int] = {}
    leaf_rows: list[np.ndarray] = []

    def leaf_id(row: np.ndarray) -> int:
        if not has_table:
            return int(row[0])
        k = row.tobytes()
        if k not in leaf_keys:
            leaf_keys[k] = len(leaf_rows)
            leaf_rows.append(row)
        return leaf_keys[k]

    nf, nthr, nlc, nrc, roots = [], [], [], [], []
    if layout == "ref":
        # referencias: >= 0 nodo interno; < 0 hoja codificada como ~id (se traduce al final)
        memo: dict[tuple, int] = {}
        for t, p in zip(trees, pay, strict=True):
            ref: dict[int, int] = {}
            for i in reversed(_preorder(t)):
                if t.left[i] == -1:
                    ref[i] = ~leaf_id(p[i])
                    continue
                lr, rr = ref[int(t.left[i])], ref[int(t.right[i])]
                if dag and lr == rr:
                    ref[i] = lr  # ambas ramas llevan al mismo sitio: la prueba sobra
                    continue
                key = (int(t.feature[i]), int(t.thr[i : i + 1].view(np.uint32)[0]), lr, rr)
                if dag and key in memo:
                    ref[i] = memo[key]
                    continue
                nf.append(key[0])
                nthr.append(t.thr[i])
                nlc.append(lr)
                nrc.append(rr)
                ref[i] = len(nf) - 1
                if dag:
                    memo[key] = ref[i]
            roots.append(ref[0])
        n_int = len(nf)
        n_leaf = len(leaf_rows) if has_table else C
        big = max(n_int, n_leaf) >= (1 << 15) - 1
        flag = 0x80000000 if big else 0x8000
        tr_ref = lambda a: np.array([(~r | flag) if r < 0 else r for r in a], dtype=np.int64)  # noqa: E731
        lref, rref, rts = tr_ref(nlc), tr_ref(nrc), tr_ref(roots)
        feat_a = np.array(nf, dtype=np.int64)
        thr_a = np.array(nthr, dtype=np.float32)
    else:
        for t, p in zip(trees, pay, strict=True):
            order = _preorder(t)
            pos = {i: k for k, i in enumerate(order)}
            base = len(nf)
            roots.append(base)
            for i in order:
                if t.left[i] == -1:
                    nf.append(F)  # centinela de hoja
                    nthr.append(np.float32(0))
                    nrc.append(leaf_id(p[i]))
                else:
                    nf.append(int(t.feature[i]))
                    nthr.append(t.thr[i])
                    nrc.append(pos[int(t.right[i])] - pos[i])
        lref = np.zeros(0, dtype=np.int64)
        rref = np.array(nrc, dtype=np.int64)
        rts = np.array(roots, dtype=np.int64)
        feat_a = np.array(nf, dtype=np.int64)
        thr_a = np.array(nthr, dtype=np.float32)

    # ---- tabla de cortes por característica (solo los umbrales que quedan en uso)
    is_int = feat_a < F if layout == "dfs" else np.ones(len(feat_a), dtype=bool)
    cuts_l, off = [], [0]
    bin_a = np.zeros(len(feat_a), dtype=np.int64)
    for j in range(F):
        m = is_int & (feat_a == j)
        u, inv = np.unique(thr_a[m], return_inverse=True)
        bin_a[np.flatnonzero(m)] = inv
        cuts_l.append(u)
        off.append(off[-1] + len(u))
    cuts = np.concatenate(cuts_l).astype(np.float32) if cuts_l else np.zeros(0, np.float32)
    table = None
    if has_table:
        table = np.array(leaf_rows, dtype=ldt).reshape(len(leaf_rows), -1)
        if layout == "ref" and len(table) == 0:
            table = np.zeros((1, C if cls else 1), dtype=ldt)
    if not thr_table:
        cuts, off = np.zeros(0, np.float32), [0] * (F + 1)
    return TinyForest(
        task="classification" if cls else "regression",
        n_features=F,
        n_classes=C,
        n_trees=len(trees),
        layout=layout,
        leaf=leaf,
        dag=dag,
        quant_input=quant_input,
        thr_table=thr_table,
        integer_inputs=integer_inputs,
        classes=np.asarray(est.classes_) if cls else None,
        feat=feat_a,
        bin=bin_a,
        thr=thr_a,
        lref=lref,
        rref=rref,
        roots=rts,
        leaf_table=table,
        cuts=cuts,
        off=np.asarray(off, dtype=np.int64),
        leaf_scale=scale,
        info={"max_bins": max_bins, "thr_table": thr_table, "integer_inputs": integer_inputs},
    )


def emit_compact_trees(est, name, spec, task, n_classes, **opts):
    """Emisor compatible con ``cgen.EMITTERS`` (usado por ``export_run(..., compact=True)``)."""
    kw = {"layout": "ref", "dag": True, "leaf": "f32", "thr_table": None}
    kw.update(opts)
    tf = build(est, **kw)
    code, data_bytes, ram, info = tf.to_c()
    return code, data_bytes, ram, info
