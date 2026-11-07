"""Exportación a ONNX de los modelos clásicos (scikit-learn), con el preprocesado DENTRO del grafo.

El ``model.onnx`` recibe valores crudos (``numeric`` float32 con NaN si falta y ``categorical``
int64 con el código de cada categoría, -1 = falta, -2 = desconocida), aplica la misma limpieza y
codificación que el entrenamiento y el firmware (``DeviceSpec``) y después el modelo. Así no hay
desajuste entre entrenamiento e inferencia. Se verifica con onnxruntime contra scikit-learn.

Requiere ``onnx``, ``onnxruntime`` y ``skl2onnx`` (componente opcional "onnx").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio.clean import Pipeline, prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit
from tinyml_studio.train.features import FeatureEncoder

from .spec import DeviceSpec, build_spec, emulate_features, raw_inputs

OPSET = 14
MAX_VERIFY_ROWS = 2000
F32 = np.float32


def _need() -> tuple[Any, Any, Any]:
    try:
        import onnx
        import onnxruntime
        import skl2onnx
    except ImportError as e:
        raise RuntimeError(
            tr(
                "la exportación a ONNX necesita los paquetes onnx, onnxruntime y skl2onnx "
                "(faltan: {m}). Instálalos desde Herramientas > Entorno o con pip.",
                "ONNX export needs the onnx, onnxruntime and skl2onnx packages (missing: {m}). "
                "Install them from Tools > Environment or with pip.",
                m=e.name,
            )
        ) from e
    return onnx, onnxruntime, skl2onnx


def build_preprocess(spec: DeviceSpec, opset: int = OPSET):
    """Grafo ONNX que reproduce ``emulate_features`` (misma lógica y mismo orden)."""
    from onnx import TensorProto, helper, numpy_helper

    nodes: list = []
    inits: list = []
    counter = [0]

    def nm(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    def const(value, dtype) -> str:
        name = nm("c")
        inits.append(numpy_helper.from_array(np.array(value, dtype=dtype), name))
        return name

    def node(op: str, ins: list[str], **attrs) -> str:
        out = nm(op.lower())
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    cols: list[str] = []
    one = const([1], np.int64)
    for j, c in enumerate(spec.num_names):
        v = node("Gather", ["numeric", const(j, np.int64)], axis=1)
        nan = const(np.nan, F32)
        for op in spec.ops[c]:
            if op[0] == "replace":
                for x in op[1]:
                    v = node("Where", [node("Equal", [v, const(F32(x), F32)]), nan, v])
            elif op[0] == "range_nan":
                lo, hi = const(F32(op[1]), F32), const(F32(op[2]), F32)
                bad = node("Or", [node("Less", [v, lo]), node("Greater", [v, hi])])
                v = node("Where", [bad, nan, v])
            elif op[0] == "clip":
                lo, hi = const(F32(op[1]), F32), const(F32(op[2]), F32)
                v = node("Where", [node("Less", [v, lo]), lo, v])
                v = node("Where", [node("Greater", [v, hi]), hi, v])
            elif op[0] == "fill":
                v = node("Where", [node("IsNaN", [v]), const(F32(op[1]), F32), v])
            elif op[0] == "scale":
                v = node("Div", [node("Sub", [v, const(F32(op[1]), F32)]), const(F32(op[2]), F32)])
        cols.append(node("Unsqueeze", [v, one]))
    for j, c in enumerate(spec.cat_names):
        code = node("Gather", ["categorical", const(j, np.int64)], axis=1)
        miss = node("Equal", [code, const(-1, np.int64)])
        code = node("Where", [miss, const(spec.cat_fill[c], np.int64), code])
        method = spec.cat_methods.get(c, "onehot")
        if method == "ordinal":  # desconocida (-2) -> -1; la falta ya se rellenó con la moda
            neg = node("Less", [code, const(0, np.int64)])
            f = node(
                "Cast", [node("Where", [neg, const(-1, np.int64), code])], to=TensorProto.FLOAT
            )
            cols.append(node("Unsqueeze", [f, one]))
            continue
        for k in range(1 if method == "dummy" else 0, len(spec.cat_values[c])):
            hot = node("Cast", [node("Equal", [code, const(k, np.int64)])], to=TensorProto.FLOAT)
            cols.append(node("Unsqueeze", [hot, one]))
    nodes.append(helper.make_node("Concat", cols, ["features"], axis=1))
    ins = []
    if spec.num_names:
        ins.append(
            helper.make_tensor_value_info("numeric", TensorProto.FLOAT, [None, len(spec.num_names)])
        )
    if spec.cat_names:
        ins.append(
            helper.make_tensor_value_info(
                "categorical", TensorProto.INT64, [None, len(spec.cat_names)]
            )
        )
    outs = [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, spec.n_features])]
    graph = helper.make_graph(nodes, "tinyml_preprocess", ins, outs, inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _convert_estimator(est, n_features: int):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.base import is_classifier

    options = {id(est): {"zipmap": False}} if is_classifier(est) else None
    return convert_sklearn(
        est,
        initial_types=[("features", FloatTensorType([None, n_features]))],
        target_opset={"": OPSET, "ai.onnx.ml": 3},
        options=options,
    )


INFER_PY = '''"""Inferencia con onnxruntime (generado por TinyML Studio).
Requiere: pip install onnxruntime numpy"""
import json
import numpy as np
import onnxruntime as ort

meta = json.load(open("onnx_meta.json", encoding="utf-8"))
sess = ort.InferenceSession("model.onnx")


def predict(numeric: dict, categorical: dict | None = None, reject_below=None):
    """numeric: {columna: valor o None}; categorical: {columna: texto o None}.
    Si la probabilidad máxima es menor que reject_below (por defecto la del meta): "desconocido"."""
    cols = meta["inputs"]["numeric"]
    num = np.array([[np.nan if numeric.get(c) is None else numeric[c] for c in cols]],
                   dtype=np.float32)
    feeds = {"numeric": num} if meta["inputs"]["numeric"] else {}
    if meta["inputs"]["categorical"]:
        codes = []
        for c, info in meta["inputs"]["categorical"].items():
            v = (categorical or {}).get(c)
            unk = info["unknown"]
            codes.append(info["missing"] if v is None else info["codes"].get(str(v), unk))
        feeds["categorical"] = np.array([codes], dtype=np.int64)
    outs = sess.run(None, feeds)
    label = outs[0][0]
    if meta["task"] == "classification":
        proba = outs[1][0].tolist() if len(outs) > 1 else None
        thr = reject_below if reject_below is not None else meta.get("reject_below")
        if thr is not None and proba is not None and max(proba) < thr:
            return "desconocido", proba
        return meta["classes"][int(label)], proba
    return float(np.ravel(label)[0]), None


if __name__ == "__main__":
    print(predict({c: 0.0 for c in meta["inputs"]["numeric"]}))
'''


def export_onnx(
    run_dir: str | Path,
    out_dir: str | Path,
    model: str | None = None,
    reject_below: float | None = None,
) -> dict[str, Any]:
    import joblib

    onnx, ort, _skl = _need()
    run, out = Path(run_dir), Path(out_dir)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    recipe = json.loads((run / "recipe_fitted.json").read_text(encoding="utf-8"))
    enc = FeatureEncoder.from_dict(json.loads((run / "encoder.json").read_text(encoding="utf-8")))
    task = results["task"]
    name = model or results["selection"]["best"]
    path = run / f"model_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            tr(
                "no existe {f}: Entrena con ese modelo primero",
                "{f} does not exist: Train with that model first",
                f=path.name,
            )
        )
    est = joblib.load(path)
    spec = build_spec(Pipeline.from_list(recipe["cleaning"], fitted=True), enc)

    emit("generate", 20, tr("Convirtiendo el modelo a ONNX…", "Converting the model to ONNX…"))
    try:
        model_proto = _convert_estimator(est, spec.n_features)
    except Exception as e:  # noqa: BLE001 - skl2onnx no soporta todos los estimadores/versiones
        raise RuntimeError(
            tr(
                "no se pudo convertir '{n}' a ONNX: {e}",
                "could not convert '{n}' to ONNX: {e}",
                n=name,
                e=str(e)[:300],
            )
        ) from e
    # skl2onnx puede bajar el opset por defecto (p. ej. GaussianNB usa 13): hay que igualarlo
    opset = next(
        (o.version for o in model_proto.opset_import if o.domain in ("", "ai.onnx")), OPSET
    )
    pre = build_preprocess(spec, max(opset, 13))
    pre.ir_version = model_proto.ir_version
    merged = onnx.compose.merge_models(pre, model_proto, io_map=[("features", "features")])
    onnx.checker.check_model(merged)

    # ---- verificación con onnxruntime contra scikit-learn (filas de test crudas)
    emit("verify", 60, tr("Verificando con onnxruntime…", "Verifying with onnxruntime…"))
    prep = prepare_splits(recipe, seed=results["reproducibility"]["seed"])
    raw_t = prep.raw["test"]
    if len(raw_t) != len(prep.clean["test"]):
        raise ValueError(
            tr(
                "filter_rule cambia el número de filas del test: Exportación no soportada aún",
                "filter_rule changes the number of test rows: export not supported yet",
            )
        )
    n = min(len(raw_t), MAX_VERIFY_ROWS)
    idx = np.random.default_rng(0).choice(len(raw_t), n, replace=False)
    num, cat = raw_inputs(spec, raw_t.iloc[idx].reset_index(drop=True))
    feeds = {}
    if spec.num_names:
        feeds["numeric"] = num.astype(F32)
    if spec.cat_names:
        feeds["categorical"] = cat.astype(np.int64)
    sess = ort.InferenceSession(merged.SerializeToString())
    outputs = sess.run(None, feeds)
    feats = emulate_features(spec, num, cat)
    expected = est.predict(feats)
    got = np.ravel(outputs[0])
    if task == "classification":
        match = float((got == expected).mean())
        proba_diff = None
        if len(outputs) > 1 and hasattr(est, "predict_proba"):
            proba_diff = float(np.max(np.abs(np.asarray(outputs[1]) - est.predict_proba(feats))))
    else:
        match = float(np.mean(np.abs(got - expected) <= 1e-3 * np.maximum(1.0, np.abs(expected))))
        proba_diff = None
    verification = {
        "rows": int(n),
        "label_match": round(match, 6),
        "max_abs_proba_diff": None if proba_diff is None else round(proba_diff, 8),
        "onnxruntime": ort.__version__,
        "ok": match >= 0.999 and (proba_diff is None or proba_diff < 1e-4),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "model.onnx").write_bytes(merged.SerializeToString())
    meta = {
        "model": name,
        "task": task,
        "classes": spec.classes,
        "inputs": {
            "numeric": spec.num_names,
            "categorical": {
                c: {"codes": {k: i for i, k in enumerate(v)}, "missing": -1, "unknown": -2}
                for c, v in spec.cat_values.items()
            },
        },
        "outputs": [o.name for o in merged.graph.output],
        "features": spec.feature_names,
        "opset": OPSET,
        "warnings": spec.warnings,
        "reject_below": reject_below,
    }
    (out / "onnx_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "infer_onnx.py").write_text(INFER_PY, encoding="utf-8")
    emit("done", 100, tr("ONNX listo", "ONNX ready"))
    return {
        "model": name,
        "task": task,
        "files": ["model.onnx", "onnx_meta.json", "infer_onnx.py"],
        "size_bytes": (out / "model.onnx").stat().st_size,
        "opset": OPSET,
        "inputs": meta["inputs"]["numeric"]
        and {"numeric": len(spec.num_names), "categorical": len(spec.cat_names)},
        "verification": verification,
        "warnings": list(spec.warnings),
    }
