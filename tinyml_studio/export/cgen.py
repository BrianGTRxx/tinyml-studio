"""Generación de C (C99, sin dependencias salvo <math.h>) para modelos clásicos de scikit-learn.

Cada emisor devuelve (código, bytes_de_datos, bytes_de_RAM_de_trabajo). Los datos son
``static const`` (van a flash en el ESP32-S3). No compilar con -ffast-math (se usa isnan).
"""

from __future__ import annotations

import numpy as np

from tinyml_studio.i18n import tr

from .spec import DeviceSpec


def cf(x: float) -> str:
    """Literal float32 exacto de C."""
    f = np.float32(x)
    if np.isnan(f):
        return "NAN"
    if np.isinf(f):
        return "INFINITY" if f > 0 else "(-INFINITY)"
    s = f"{float(f):.9g}"
    if not any(ch in s for ch in ".eE"):
        s += ".0"
    return s + "f"


def thr_f32(t: float) -> np.float32:
    """Mayor float32 <= t, para que ``x <= thr_f32`` equivalga a ``x <= t`` con x float32."""
    f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, np.float32(-np.inf))
    return f


def int_type(lo: int, hi: int) -> tuple[str, int]:
    for name, a, b, n in (
        ("int8_t", -128, 127, 1),
        ("uint8_t", 0, 255, 1),
        ("int16_t", -32768, 32767, 2),
        ("uint16_t", 0, 65535, 2),
        ("int32_t", -(2**31), 2**31 - 1, 4),
    ):
        if lo >= a and hi <= b:
            return name, n
    raise ValueError(tr("entero fuera de rango", "integer out of range"))


class Out:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.data_bytes = 0

    def float_array(self, name: str, a: np.ndarray, per_line: int = 8) -> None:
        a = np.asarray(a, dtype=np.float32).ravel()
        self.data_bytes += 4 * len(a)
        body = ",\n".join(
            "  " + ", ".join(cf(x) for x in a[i : i + per_line]) for i in range(0, len(a), per_line)
        )
        self.lines.append(
            f"static const float {name}[{max(len(a), 1)}] = {{\n{body or '  0.0f'}\n}};"
        )

    def int_array(self, name: str, a: np.ndarray, per_line: int = 16) -> None:
        a = np.asarray(a, dtype=np.int64).ravel()
        t, n = int_type(int(a.min()) if len(a) else 0, int(a.max()) if len(a) else 0)
        self.data_bytes += n * len(a)
        body = ",\n".join(
            "  " + ", ".join(str(int(x)) for x in a[i : i + per_line])
            for i in range(0, len(a), per_line)
        )
        self.lines.append(f"static const {t} {name}[{max(len(a), 1)}] = {{\n{body or '  0'}\n}};")
        return t

    def text(self) -> str:
        return "\n\n".join(self.lines)


ARGMAX = """static int tm_argmax(const float *v, int n) {
  int best = 0;
  for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
  return best;
}"""


# ---------------------------------------------------------------- preprocesamiento
def emit_prepare(spec: DeviceSpec) -> str:
    L = ["void tm_prepare(const float *num, const int *cat, float *f) {", "  float v;"]
    if not spec.cat_names:
        L.append("  (void)cat;")
    for j, c in enumerate(spec.num_names):
        L.append(f"  /* num[{j}] {c} */")
        L.append(f"  v = num[{j}];")
        for op in spec.ops[c]:
            if op[0] == "replace":
                for x in op[1]:
                    L.append(f"  if (v == {cf(x)}) v = NAN;")
            elif op[0] == "range_nan":
                L.append(f"  if (v < {cf(op[1])} || v > {cf(op[2])}) v = NAN;")
            elif op[0] == "clip":
                L.append(
                    f"  if (v < {cf(op[1])}) v = {cf(op[1])}; else if (v > {cf(op[2])}) "
                    f"v = {cf(op[2])};"
                )
            elif op[0] == "fill":
                L.append(f"  if (isnan(v)) v = {cf(op[1])};")
            elif op[0] == "scale":
                L.append(f"  v = (v - ({cf(op[1])})) / {cf(op[2])};")
        L.append(f"  f[{j}] = v;")
    off = len(spec.num_names)
    for j, c in enumerate(spec.cat_names):
        k = len(spec.cat_values[c])
        L.append(f"  /* cat[{j}] {c}: {k} categories (-1 missing, -2 unknown) */")
        L.append("  {")
        L.append(f"    int code = cat[{j}];")
        if spec.cat_fill[c] >= 0:
            L.append(f"    if (code == -1) code = {spec.cat_fill[c]};")
        m = spec.cat_methods.get(c, "onehot")
        if m == "ordinal":
            L.append(f"    f[{off}] = (code < 0) ? -1.0f : (float)code;  /* unknown -> -1 */")
            width = 1
        else:
            first = 1 if m == "dummy" else 0  # dummy: la categoría 0 es la referencia (omitida)
            L.append(
                f"    for (int k = {first}; k < {k}; k++) "
                f"f[{off} + k - {first}] = (code == k) ? 1.0f : 0.0f;"
            )
            width = k - first
        L.append("  }")
        off += width
    L.append("}")
    return "\n".join(L)


# ---------------------------------------------------------------- árboles y bosques
def _tree_arrays(trees, n_features, mode, n_classes):
    """Concatena nodos de varios árboles. mode: 'cls_class' | 'cls_proba' | 'reg'."""
    feat, thr, left, right, roots, proba_rows = [], [], [], [], [], []
    for t in trees:
        T = t.tree_
        base = len(feat)
        roots.append(base)
        value = T.value[:, 0, :] if T.value.ndim == 3 else T.value
        for i in range(T.node_count):
            if T.children_left[i] == -1:  # hoja
                feat.append(-1)
                if mode == "reg":
                    thr.append(np.float32(value[i, 0]))
                    left.append(0)
                elif mode == "cls_class":
                    thr.append(np.float32(0))
                    left.append(int(np.argmax(value[i])))
                else:
                    thr.append(np.float32(0))
                    left.append(len(proba_rows))
                    v = value[i]
                    proba_rows.append(v / v.sum())
                right.append(0)
            else:
                feat.append(int(T.feature[i]))
                thr.append(thr_f32(float(T.threshold[i])))
                left.append(base + int(T.children_left[i]))
                right.append(base + int(T.children_right[i]))
    return feat, thr, left, right, roots, proba_rows


def emit_trees(est, name, spec, task, n_classes):
    forest = name in ("random_forest", "extra_trees")
    trees = list(est.estimators_) if forest else [est]
    mode = "reg" if task == "regression" else ("cls_proba" if forest else "cls_class")
    feat, thr, left, right, roots, proba = _tree_arrays(trees, spec.n_features, mode, n_classes)
    o = Out()
    o.int_array("tm_feat", np.array(feat))
    o.float_array("tm_thr", np.array(thr))
    o.int_array("tm_left", np.array(left))
    o.int_array("tm_right", np.array(right))
    if forest:
        o.int_array("tm_root", np.array(roots))
    if mode == "cls_proba":
        o.float_array("tm_proba", np.array(proba))
    ntrees = len(trees)
    walk = (
        "    int node = {root};\n"
        "    while (tm_feat[node] >= 0)\n"
        "      node = (f[tm_feat[node]] <= tm_thr[node]) ? tm_left[node] : tm_right[node];\n"
    )
    if forest:
        rootv = "tm_root[t]"
        head = f"  for (int t = 0; t < {ntrees}; t++) {{\n"
    else:
        rootv = "0"
        head = "  {\n"
    body = walk.format(root=rootv)
    if mode == "reg":
        if forest:
            code = (
                "static float tm_model(const float *f) {\n  float acc = 0.0f;\n"
                + head
                + body
                + "    acc += tm_thr[node];\n  }\n"
                + f"  return acc / {cf(ntrees)};\n}}"
            )
        else:
            code = (
                "static float tm_model(const float *f) {\n"
                + head
                + body
                + ("    return tm_thr[node];\n  }\n}")
            )
        ram = 4 if forest else 0
    elif mode == "cls_class":
        code = (
            "static int tm_model(const float *f) {\n"
            + head
            + body
            + ("    return (int)tm_left[node];\n  }\n}")
        )
        ram = 0
    else:
        code = (
            f"static int tm_model(const float *f) {{\n  float acc[{n_classes}];\n"
            f"  for (int c = 0; c < {n_classes}; c++) acc[c] = 0.0f;\n"
            + head
            + body
            + f"    const float *p = &tm_proba[(int)tm_left[node] * {n_classes}];\n"
            f"    for (int c = 0; c < {n_classes}; c++) acc[c] += p[c];\n  }}\n"
            f"  return tm_argmax(acc, {n_classes});\n}}"
        )
        ram = 4 * n_classes
    return (
        o.text() + "\n\n" + code,
        o.data_bytes,
        ram,
        {"trees": ntrees, "nodes": len(feat), "leaf_rows": len(proba)},
    )


# ---------------------------------------------------------------- lineales
def emit_linear(est, name, spec, task, n_classes):
    W = np.atleast_2d(est.coef_)
    b = np.atleast_1d(est.intercept_)
    F = spec.n_features
    o = Out()
    o.float_array("tm_W", W)
    o.float_array("tm_b", b)
    if task == "regression":
        code = (
            f"static float tm_model(const float *f) {{\n  float s = tm_b[0];\n"
            f"  for (int i = 0; i < {F}; i++) s += tm_W[i] * f[i];\n  return s;\n}}"
        )
        return o.text() + "\n\n" + code, o.data_bytes, 0, {"parameters": W.size + b.size}
    if W.shape[0] == 1:  # binaria: una sola fila
        code = (
            f"static int tm_model(const float *f) {{\n  float s = tm_b[0];\n"
            f"  for (int i = 0; i < {F}; i++) s += tm_W[i] * f[i];\n  return s > 0.0f ? 1 : 0;\n}}"
        )
        ram = 0
    else:
        code = (
            f"static int tm_model(const float *f) {{\n  float s[{n_classes}];\n"
            f"  for (int c = 0; c < {n_classes}; c++) {{\n    float a = tm_b[c];\n"
            f"    for (int i = 0; i < {F}; i++) a += tm_W[c * {F} + i] * f[i];\n    s[c] = a;\n"
            f"  }}\n  return tm_argmax(s, {n_classes});\n}}"
        )
        ram = 4 * n_classes
    return o.text() + "\n\n" + code, o.data_bytes, ram, {"parameters": W.size + b.size}


# ---------------------------------------------------------------- Naive Bayes gaussiano
def emit_gaussian_nb(est, name, spec, task, n_classes):
    if not np.array_equal(est.classes_, np.arange(n_classes)):
        raise ValueError(
            tr("las clases de Naive Bayes deben ser 0..n-1", "Naive Bayes classes must be 0..n-1")
        )
    F = spec.n_features
    theta = np.asarray(est.theta_, dtype=np.float64)
    var = np.asarray(est.var_, dtype=np.float64)
    # log P(c) - 0.5*sum(log(2*pi*var)) es constante por clase; 0.5/var se precalcula
    const = np.log(est.class_prior_) - 0.5 * np.log(2.0 * np.pi * var).sum(1)
    o = Out()
    o.float_array("tm_mu", theta)
    o.float_array("tm_hiv", 0.5 / var)
    o.float_array("tm_c0", const)
    code = (
        f"static int tm_model(const float *f) {{\n  float s[{n_classes}];\n"
        f"  for (int c = 0; c < {n_classes}; c++) {{\n    float a = tm_c0[c];\n"
        f"    for (int i = 0; i < {F}; i++) {{\n"
        f"      float d = f[i] - tm_mu[c * {F} + i];\n"
        f"      a -= d * d * tm_hiv[c * {F} + i];\n    }}\n    s[c] = a;\n  }}\n"
        f"  return tm_argmax(s, {n_classes});\n}}"
    )
    params = theta.size + var.size + len(const)
    return o.text() + "\n\n" + code, o.data_bytes, 4 * n_classes, {"parameters": int(params)}


# ---------------------------------------------------------------- SVM (libsvm, uno contra uno)
def emit_svm(est, name, spec, task, n_classes):
    if est.kernel not in ("rbf", "linear"):
        raise NotImplementedError(
            tr(
                "kernel '{k}' no soportado (solo rbf y linear)",
                "kernel '{k}' not supported (only rbf and linear)",
                k=est.kernel,
            )
        )
    if not np.array_equal(est.classes_, np.arange(n_classes)):
        raise ValueError(tr("las clases del SVM deben ser 0..n-1", "SVM classes must be 0..n-1"))
    F = spec.n_features
    sv = np.asarray(est.support_vectors_, dtype=np.float32)
    nsv = sv.shape[0]
    starts = np.concatenate([[0], np.cumsum(est.n_support_)])
    o = Out()
    o.float_array("tm_sv", sv)
    o.float_array("tm_dual", est.dual_coef_)
    o.float_array("tm_icpt", np.atleast_1d(est.intercept_))
    o.int_array("tm_start", starts)
    kern = f"expf(-{cf(est._gamma)} * d2)" if est.kernel == "rbf" else "dot"
    kloop = (
        f"    float d2 = 0.0f;\n    for (int i = 0; i < {F}; i++) {{ float d = f[i] - s[i]; "
        "d2 += d * d; }\n"
        if est.kernel == "rbf"
        else f"    float dot = 0.0f;\n    for (int i = 0; i < {F}; i++) dot += f[i] * s[i];\n"
    )
    head = (
        f"static int tm_model(const float *f) {{\n  float k[{nsv}];\n"
        f"  for (int j = 0; j < {nsv}; j++) {{\n    const float *s = &tm_sv[j * {F}];\n"
        f"{kloop}    k[j] = {kern};\n  }}\n"
    )
    if n_classes == 2:
        code = head + (
            f"  float s = tm_icpt[0];\n  for (int j = 0; j < {nsv}; j++) "
            "s += tm_dual[j] * k[j];\n  return s > 0.0f ? 1 : 0;\n}"
        )
    else:
        code = head + (
            f"  int votes[{n_classes}];\n  for (int c = 0; c < {n_classes}; c++) votes[c] = 0;\n"
            f"  int p = 0;\n  for (int i = 0; i < {n_classes}; i++) {{\n"
            f"    for (int j = i + 1; j < {n_classes}; j++, p++) {{\n      float s = tm_icpt[p];\n"
            f"      for (int a = tm_start[i]; a < tm_start[i + 1]; a++) "
            f"s += tm_dual[(j - 1) * {nsv} + a] * k[a];\n"
            f"      for (int a = tm_start[j]; a < tm_start[j + 1]; a++) "
            f"s += tm_dual[i * {nsv} + a] * k[a];\n"
            f"      if (s > 0.0f) votes[i]++; else votes[j]++;\n    }}\n  }}\n"
            f"  int best = 0;\n  for (int c = 1; c < {n_classes}; c++) "
            "if (votes[c] > votes[best]) best = c;\n  return best;\n}"
        )
    ram = 4 * nsv + 4 * n_classes
    return o.text() + "\n\n" + code, o.data_bytes, ram, {"support_vectors": int(nsv)}


# ---------------------------------------------------------------- MLP
def emit_mlp(est, name, spec, task, n_classes):
    act = est.activation
    if act not in ("relu", "identity", "tanh", "logistic"):
        raise NotImplementedError(
            tr("activación '{a}' no soportada", "activation '{a}' not supported", a=act)
        )
    o = Out()
    sizes = [est.coefs_[0].shape[0]] + [w.shape[1] for w in est.coefs_]
    for i, (w, b) in enumerate(zip(est.coefs_, est.intercepts_, strict=True)):
        o.float_array(f"tm_W{i}", w)  # (entradas, salidas), fila principal
        o.float_array(f"tm_b{i}", b)
    maxw = max(sizes[1:])
    actc = {
        "relu": "x > 0.0f ? x : 0.0f",
        "identity": "x",
        "tanh": "tanhf(x)",
        "logistic": "1.0f / (1.0f + expf(-x))",
    }[act]
    L = [
        "static void tm_layer(const float *in, int n_in, const float *W, const float *b,\n"
        "                     int n_out, float *out, int hidden) {\n"
        "  for (int j = 0; j < n_out; j++) {\n    float x = b[j];\n"
        "    for (int i = 0; i < n_in; i++) x += in[i] * W[i * n_out + j];\n"
        f"    out[j] = hidden ? ({actc}) : x;\n  }}\n}}",
        f"static {'float' if task == 'regression' else 'int'} tm_model(const float *f) {{",
        f"  float a[{maxw}], b[{maxw}];",
        "  const float *in = f;",
    ]
    n_layers = len(est.coefs_)
    for i in range(n_layers):
        hidden = 1 if i < n_layers - 1 else 0
        out = "a" if i % 2 == 0 else "b"
        L.append(f"  tm_layer(in, {sizes[i]}, tm_W{i}, tm_b{i}, {sizes[i + 1]}, {out}, {hidden});")
        L.append(f"  in = {out};")
    last = sizes[-1]
    if task == "regression":
        L.append("  return in[0];")
    elif last == 1:
        L.append("  return in[0] > 0.0f ? 1 : 0;")
    else:
        L.append(f"  return tm_argmax(in, {last});")
    L.append("}")
    params = int(sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_))
    return o.text() + "\n\n" + "\n".join(L), o.data_bytes, 8 * maxw, {"parameters": params}


EMITTERS = {
    "decision_tree": emit_trees,
    "random_forest": emit_trees,
    "logreg": emit_linear,
    "linear": emit_linear,
    "lasso": emit_linear,
    "elasticnet": emit_linear,
    "lda": emit_linear,
    "linear_svm": emit_linear,
    "extra_trees": emit_trees,
    "gaussian_nb": emit_gaussian_nb,
    "svm": emit_svm,
    "mlp": emit_mlp,
}
