"""Descripción JSON de la estructura interna de un modelo (estilo Netron).

Forma común: ``{"format", "summary", "inputs", "outputs", "tensors", "nodes", "stats"}``.
Solo se usan claves (sin texto de UI); la interfaz las traduce. Todo es serializable con ``json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TINY_ELEMS = 64  # tensores constantes hasta este tamaño se incluyen completos
HIST_BINS = 20
MAX_LIST = 256  # tope de escalas por canal y de coeficientes mostrados


# ---------------------------------------------------------------- utilidades
def _py(v: Any) -> Any:
    """Convierte tipos numpy/bytes a tipos JSON puros."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def weight_stats(arr: np.ndarray) -> dict[str, Any]:
    """min/max/mean/std + histograma de 20 bins; el arreglo completo solo si es diminuto."""
    a = np.asarray(arr)
    out: dict[str, Any] = {"count": int(a.size)}
    if a.size == 0:
        return out
    f = a.astype(np.float64).ravel()
    out.update(min=float(f.min()), max=float(f.max()), mean=float(f.mean()), std=float(f.std()))
    counts, edges = np.histogram(f, bins=HIST_BINS)
    out["hist"] = {"counts": counts.tolist(), "edges": [float(e) for e in edges]}
    out["zeros_pct"] = float((f == 0).mean() * 100)
    if a.size <= TINY_ELEMS:
        out["values"] = _py(a)
    return out


def _stats(tensors: list[dict], nodes: list[dict], param_count: int | None = None) -> dict:
    ops: dict[str, int] = {}
    for n in nodes:
        ops[n["op"]] = ops.get(n["op"], 0) + 1
    if param_count is None:
        param_count = sum(t.get("elements", 0) for t in tensors if t["is_const"])
    return {
        "param_count": int(param_count),
        "total_const_bytes": int(sum(t["bytes"] for t in tensors if t["is_const"])),
        "op_counts": ops,
    }


def _graph(fmt, summary, inputs, outputs, tensors, nodes, param_count=None) -> dict:
    return _py(
        {
            "format": fmt,
            "summary": summary,
            "inputs": inputs,
            "outputs": outputs,
            "tensors": tensors,
            "nodes": nodes,
            "stats": _stats(tensors, nodes, param_count),
        }
    )


# ---------------------------------------------------------------- TFLite
def _enum_names(cls) -> dict[int, str]:
    return {v: k for k, v in vars(cls).items() if isinstance(v, int) and not k.startswith("_")}


_ITEMSIZE = {
    "FLOAT32": 4,
    "FLOAT16": 2,
    "FLOAT64": 8,
    "INT8": 1,
    "UINT8": 1,
    "INT16": 2,
    "UINT16": 2,
    "INT32": 4,
    "UINT32": 4,
    "INT64": 8,
    "UINT64": 8,
    "BOOL": 1,
    "INT4": 0.5,
}
_NP = {
    "FLOAT32": np.float32,
    "FLOAT16": np.float16,
    "FLOAT64": np.float64,
    "INT8": np.int8,
    "UINT8": np.uint8,
    "INT16": np.int16,
    "UINT16": np.uint16,
    "INT32": np.int32,
    "UINT32": np.uint32,
    "INT64": np.int64,
    "UINT64": np.uint64,
    "BOOL": np.bool_,
}


def _options_dict(opts, names: dict[str, dict[int, str]]) -> dict[str, Any]:
    if opts is None:
        return {}
    out: dict[str, Any] = {}
    for k, v in vars(opts).items():
        if v is None or k.startswith("_"):
            continue
        if k == "fusedActivationFunction":
            out["fused_activation"] = names["act"].get(int(v), int(v))
        elif k == "padding":
            out["padding"] = names["pad"].get(int(v), int(v))
        elif isinstance(v, (bool, int, float, str, bytes, np.generic, np.ndarray, list)):
            out[k] = v
    return out


def tflite_graph(model_bytes: bytes) -> dict:
    """Parsea un flatbuffer .tflite (offline, con el esquema de ``tflite_micro``)."""
    from tflite_micro import runtime

    sfb = runtime.schema_fb
    m = runtime.convert_bytearray_to_object(bytearray(model_bytes))
    op_names = _enum_names(sfb.BuiltinOperator)
    type_names = _enum_names(sfb.TensorType)
    names = {
        "act": _enum_names(sfb.ActivationFunctionType),
        "pad": _enum_names(sfb.Padding),
    }
    sg = m.subgraphs[0]

    tensors: list[dict] = []
    for i, t in enumerate(sg.tensors):
        dtype = type_names.get(int(t.type), str(int(t.type)))
        shape = [int(x) for x in (t.shape if t.shape is not None else [])]
        elems = int(np.prod(shape)) if shape else 1
        buf = m.buffers[t.buffer] if 0 < t.buffer < len(m.buffers) else None
        raw = buf.data if buf is not None else None
        is_const = raw is not None and len(raw) > 0
        q = t.quantization
        quant = None
        if q is not None and q.scale is not None and len(q.scale) > 0:
            sc = [float(x) for x in q.scale]
            zp = [int(x) for x in (q.zeroPoint if q.zeroPoint is not None else [])]
            per_ch = len(sc) > 1
            quant = {
                "scale": sc[:MAX_LIST] if per_ch else sc[0],
                "zero_point": (zp[:MAX_LIST] if per_ch else (zp[0] if zp else 0)),
                "per_channel": per_ch,
                "channels": len(sc),
                "quantized_dimension": int(q.quantizedDimension),
            }
        item: dict[str, Any] = {
            "id": i,
            "name": (t.name.decode() if isinstance(t.name, bytes) else str(t.name or "")),
            "shape": shape,
            "dtype": dtype,
            "quant": quant,
            "is_const": bool(is_const),
            "bytes": int(len(raw)) if is_const else 0,
            "elements": elems,
        }
        if is_const and dtype in _NP:
            arr = np.frombuffer(bytes(bytearray(raw)), dtype=_NP[dtype])
            if arr.size == elems:
                arr = arr.reshape(shape) if shape else arr
            item["data"] = weight_stats(arr)
        tensors.append(item)

    nodes: list[dict] = []
    for j, op in enumerate(sg.operators):
        oc = m.operatorCodes[op.opcodeIndex]
        code = runtime.get_builtin_code_from_operator_code(oc)
        name = op_names.get(code, f"UNKNOWN_{code}")
        if name == "CUSTOM" and oc.customCode:
            c = oc.customCode
            name = c.decode() if isinstance(c, bytes) else str(c)
        nodes.append(
            {
                "id": j,
                "op": name,
                "name": f"{name}_{j}",
                "inputs": [int(x) for x in (op.inputs if op.inputs is not None else []) if x >= 0],
                "outputs": [int(x) for x in (op.outputs if op.outputs is not None else [])],
                "options": _options_dict(op.builtinOptions, names),
            }
        )

    ins = [int(x) for x in sg.inputs]
    outs = [int(x) for x in sg.outputs]
    n_const_params = sum(t["elements"] for t in tensors if t["is_const"])
    dtypes = {tensors[i]["dtype"] for i in ins + outs}
    summary = {
        "n_nodes": len(nodes),
        "n_tensors": len(tensors),
        "n_subgraphs": len(m.subgraphs),
        "tflite_version": int(m.version),
        "model_bytes": len(model_bytes),
        "io_dtypes": sorted(dtypes),
        "quantized": any(
            t["quant"] is not None and t["dtype"] in ("INT8", "UINT8", "INT16") for t in tensors
        ),
    }
    return _graph("tflite", summary, ins, outs, tensors, nodes, n_const_params)


# ---------------------------------------------------------------- Keras
def keras_graph(model) -> dict:
    """Capas de un modelo Keras como nodos, con estadísticas de pesos."""
    tensors: list[dict] = []
    nodes: list[dict] = []

    def add_tensor(name, shape, dtype="FLOAT32", arr=None) -> int:
        tid = len(tensors)
        tensors.append(
            {
                "id": tid,
                "name": name,
                "shape": [int(x) if x is not None else -1 for x in shape],
                "dtype": dtype,
                "quant": None,
                "is_const": arr is not None,
                "bytes": int(arr.nbytes) if arr is not None else 0,
                "elements": int(arr.size)
                if arr is not None
                else int(np.prod([abs(int(x or 1)) for x in shape])),
                **({"data": weight_stats(arr)} if arr is not None else {}),
            }
        )
        return tid

    def shp(x):
        try:
            return list(x)
        except TypeError:
            return []

    cur = add_tensor("input", shp(model.input_shape))
    inputs = [cur]
    params = 0
    for layer in model.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        wins = []
        for w in layer.weights:
            arr = np.asarray(w.numpy())
            params += arr.size
            wins.append(
                add_tensor(w.path if hasattr(w, "path") else w.name, arr.shape, "FLOAT32", arr)
            )
        out = add_tensor(f"{layer.name}:out", shp(layer.output.shape))
        opts: dict[str, Any] = {}
        act = getattr(layer, "activation", None)
        if act is not None:
            opts["activation"] = getattr(act, "__name__", str(act))
        for k in ("units", "filters", "kernel_size", "strides", "padding", "rate", "pool_size"):
            if hasattr(layer, k):
                opts[k] = getattr(layer, k)
        nodes.append(
            {
                "id": len(nodes),
                "op": layer.__class__.__name__,
                "name": layer.name,
                "inputs": [cur, *wins],
                "outputs": [out],
                "options": opts,
            }
        )
        cur = out
    summary = {"n_nodes": len(nodes), "n_tensors": len(tensors), "name": model.name}
    return _graph("keras", summary, inputs, [cur], tensors, nodes, params)


# ---------------------------------------------------------------- sklearn
def _tree_node(
    t, node: int, depth: int, max_depth: int, feats: list[str], classes: list[str] | None
) -> dict:
    n_samples = int(t.n_node_samples[node])
    val = np.asarray(t.value[node]).reshape(-1)
    leaf = t.children_left[node] == -1
    d: dict[str, Any] = {"samples": n_samples, "leaf": bool(leaf)}
    if val.size > 1 or classes is not None:
        s = val.sum()
        dist = (val / s) if s > 0 else val
        d["distribution"] = [float(x) for x in dist]
        k = int(np.argmax(val))
        d["prediction"] = classes[k] if classes and k < len(classes) else k
    else:
        d["value"] = float(val[0])
    d["impurity"] = float(t.impurity[node])
    if leaf:
        d["truncated"] = False
        return d
    f = int(t.feature[node])
    d["feature_index"] = f
    d["feature"] = feats[f] if f < len(feats) else f"x{f}"
    d["threshold"] = float(t.threshold[node])
    if depth >= max_depth:
        d["truncated"] = True
        return d
    d["truncated"] = False
    d["left"] = _tree_node(t, int(t.children_left[node]), depth + 1, max_depth, feats, classes)
    d["right"] = _tree_node(t, int(t.children_right[node]), depth + 1, max_depth, feats, classes)
    return d


def _tree_struct(est, feats, classes, max_depth) -> dict:
    t = est.tree_
    root = _tree_node(t, 0, 0, max_depth, feats, classes)
    return {
        "depth": int(est.get_depth()),
        "n_leaves": int(est.get_n_leaves()),
        "node_count": int(t.node_count),
        "shown_depth": int(max_depth),
        "truncated": bool(int(est.get_depth()) > max_depth),
        "root": root,
    }


def _coef_block(est, feats, classes) -> dict:
    coef = np.atleast_2d(np.asarray(est.coef_, dtype=float))
    inter = np.atleast_1d(np.asarray(getattr(est, "intercept_", np.zeros(coef.shape[0]))))
    n = coef.shape[0]
    if classes and len(classes) == n:
        labels = list(classes)
    elif classes and n == 1 and len(classes) == 2:
        labels = [classes[1]]
    else:
        labels = [str(i) for i in range(n)]
    rows = [
        {
            "class": labels[i],
            "intercept": float(inter[i]) if i < inter.size else 0.0,
            "coefficients": [
                {"feature": feats[j] if j < len(feats) else f"x{j}", "value": float(coef[i, j])}
                for j in range(coef.shape[1])
            ],
        }
        for i in range(n)
    ]
    return {"n_features": int(coef.shape[1]), "classes": rows}


def sklearn_graph(estimator, feature_names, class_names=None, max_tree_depth: int = 4) -> dict:
    """Estructura interna de un estimador scikit-learn (árbol, bosque, lineal, SVM, MLP)."""
    est = estimator
    if hasattr(est, "steps"):  # Pipeline: se describe el último paso
        est = est.steps[-1][1]
    feats = [str(f) for f in feature_names]
    classes = [str(c) for c in class_names] if class_names is not None else None
    kind = est.__class__.__name__
    summary: dict[str, Any] = {"estimator": kind, "n_features": len(feats)}
    tensors: list[dict] = []
    nodes: list[dict] = []
    params = None
    model: dict[str, Any] = {}

    if hasattr(est, "tree_"):
        model = {"type": "tree", **_tree_struct(est, feats, classes, max_tree_depth)}
        summary.update(depth=model["depth"], n_leaves=model["n_leaves"])
    elif hasattr(est, "estimators_") and hasattr(est.estimators_[0], "tree_"):
        trees = est.estimators_
        per = [
            {"depth": int(t.get_depth()), "n_leaves": int(t.get_n_leaves())}
            for t in trees[:MAX_LIST]
        ]
        model = {
            "type": "forest",
            "n_trees": len(trees),
            "per_tree": per,
            "per_tree_truncated": len(trees) > MAX_LIST,
            "first_tree": _tree_struct(trees[0], feats, classes, max_tree_depth),
        }
        summary.update(
            n_trees=len(trees),
            max_depth=max(p["depth"] for p in per),
            total_leaves=int(sum(t.get_n_leaves() for t in trees)),
        )
    elif hasattr(est, "coefs_"):  # MLP
        prev = len(tensors)
        tensors.append(_t(0, "input", [1, len(feats)]))
        cur = prev
        n_layers = len(est.coefs_)
        params = 0
        for i, (w, b) in enumerate(zip(est.coefs_, est.intercepts_, strict=True)):
            ids = []
            for nm, arr in ((f"layer{i}/W", w), (f"layer{i}/b", b)):
                a = np.asarray(arr)
                params += a.size
                ids.append(len(tensors))
                tensors.append(_t(len(tensors), nm, list(a.shape), a))
            act = est.out_activation_ if i == n_layers - 1 else est.activation
            out = len(tensors)
            tensors.append(_t(out, f"layer{i}/out", [1, int(np.asarray(w).shape[1])]))
            nodes.append(
                {
                    "id": i,
                    "op": "Dense",
                    "name": f"dense_{i}",
                    "inputs": [cur, *ids],
                    "outputs": [out],
                    "options": {"units": int(np.asarray(w).shape[1]), "activation": act},
                }
            )
            cur = out
        summary.update(
            hidden_layers=[int(x) for x in est.hidden_layer_sizes]
            if hasattr(est.hidden_layer_sizes, "__iter__")
            else [int(est.hidden_layer_sizes)]
        )
        model = {"type": "mlp"}
        return _finish_sk(summary, tensors, nodes, [0], [cur], model, params)
    elif hasattr(est, "support_vectors_"):
        model = {
            "type": "svm",
            "kernel": est.kernel,
            "n_support": [int(x) for x in np.atleast_1d(est.n_support_)],
            "n_support_total": int(est.support_vectors_.shape[0]),
            "gamma": est._gamma if hasattr(est, "_gamma") else None,
            "degree": int(est.degree),
            "support_vectors_shape": [int(x) for x in est.support_vectors_.shape],
            "dual_coef": weight_stats(np.asarray(est.dual_coef_)),
        }
        summary.update(kernel=est.kernel, n_support=model["n_support_total"])
        params = int(est.support_vectors_.size + np.asarray(est.dual_coef_).size)
        if est.kernel == "linear" and hasattr(est, "coef_"):
            model["linear"] = _coef_block(est, feats, classes)
    elif hasattr(est, "coef_"):
        model = {"type": "linear", **_coef_block(est, feats, classes)}
        params = int(np.asarray(est.coef_).size + np.asarray(est.intercept_).size)
        summary.update(n_classes=len(model["classes"]))
    else:
        model = {"type": "unknown"}
    if classes:
        summary["classes"] = classes
    return _finish_sk(summary, tensors, nodes, [], [], model, params)


def _t(tid: int, name: str, shape: list[int], arr: np.ndarray | None = None) -> dict:
    t: dict[str, Any] = {
        "id": tid,
        "name": name,
        "shape": shape,
        "dtype": "FLOAT64" if arr is not None else "FLOAT32",
        "quant": None,
        "is_const": arr is not None,
        "bytes": int(arr.nbytes) if arr is not None else 0,
        "elements": int(arr.size) if arr is not None else int(np.prod(shape)),
    }
    if arr is not None:
        t["data"] = weight_stats(arr)
    return t


def _finish_sk(summary, tensors, nodes, inputs, outputs, model, params) -> dict:
    g = _graph("sklearn", summary, inputs, outputs, tensors, nodes, params)
    g["model"] = _py(model)
    return g
