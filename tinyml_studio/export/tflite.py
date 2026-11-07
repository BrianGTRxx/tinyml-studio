"""MLP de Keras -> TFLite float y int8 (cuantización completa) -> arreglo C para TFLite Micro.

Verificación obligatoria (docs/EXPORT_AND_DEPLOY.md): Keras vs tflite float vs tflite int8 sobre el
test, y además el intérprete real de TFLite Micro (paquete ``tflite-micro``) si está instalado.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio.clean import Pipeline, prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit
from tinyml_studio.pyexec import python_argv
from tinyml_studio.train.features import FeatureEncoder
from tinyml_studio.train.metrics import (
    classification_metrics,
    regression_metrics,
)

from .cgen import emit_prepare
from .classic import _names, budget
from .spec import build_spec, emulate_features, raw_inputs

TFLM_HOST_ARENA = 65536  # holgado: el envoltorio de Python añade estructuras de registro
N_CALIBRATION = 300
F1_DROP_WARN = 0.02

TFLM_ADD = {  # nombre en el esquema TFLite -> método de MicroMutableOpResolver / EloquentTinyML
    "FULLY_CONNECTED": "AddFullyConnected",
    "RELU": "AddRelu",
    "SOFTMAX": "AddSoftmax",
    "QUANTIZE": "AddQuantize",
    "DEQUANTIZE": "AddDequantize",
    "LOGISTIC": "AddLogistic",
    "TANH": "AddTanh",
    "RESHAPE": "AddReshape",
}


def _model_ops(tfl: bytes) -> list[str]:
    """Operadores del modelo, en orden. Usa tflite-micro si está; si no, el esquema de TensorFlow
    (tflite-micro no existe para Windows y sin la lista el resolver del sketch quedaría vacío)."""
    try:
        from tflite_micro import runtime

        m = runtime.convert_bytearray_to_object(bytearray(tfl))
        enum = runtime.schema_fb.BuiltinOperator
        code_of = runtime.get_builtin_code_from_operator_code
    except ImportError:
        from tensorflow.lite.python import schema_py_generated as schema

        m = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(tfl), 0))
        enum = schema.BuiltinOperator

        def code_of(oc):
            return max(oc.builtinCode, oc.deprecatedBuiltinCode)

    names = {v: k for k, v in vars(enum).items() if isinstance(v, int) and not k.startswith("_")}
    ops = []
    for op in m.subgraphs[0].operators:
        code = code_of(m.operatorCodes[op.opcodeIndex])
        ops.append(names.get(code, f"UNKNOWN_{code}"))
    return ops


def _arena_probe(tfl_path: Path) -> dict[str, Any]:
    """Arena que pide TFLite Micro (en un subproceso: si falta memoria, TFLM aborta)."""
    code = (
        "import sys\nfrom tflite_micro import runtime\n"
        "b=open(sys.argv[1],'rb').read()\n"
        "runtime.Interpreter.from_bytes(b, arena_size=int(sys.argv[2])).print_allocations()\n"
    )
    r = subprocess.run(
        python_argv(code=code, args=[str(tfl_path), "65536"]),
        capture_output=True,
        text=True,
        timeout=120,
    )
    m = re.search(r"Arena allocation total (\d+) bytes", r.stdout + r.stderr)
    if not m:
        return {"measured": False}
    return {
        "measured": True,
        "arena_bytes": int(m[1]),
        "note": tr(
            "Arena total que registra el asignador de TFLM de PC (kernels de referencia). "
            "En el ESP32-S3 (ESP-NN) puede pedir algo más: Deja margen y confirma con "
            "arena_used_bytes()",
            "Total arena recorded by the PC TFLM allocator (reference kernels). "
            "On the ESP32-S3 (ESP-NN) it may need somewhat more: Leave headroom and confirm "
            "with arena_used_bytes()",
        ),
    }


def _run_tf(interpreter, X: np.ndarray) -> np.ndarray:
    i, o = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    out = []
    for row in X:
        x = row[None].astype(np.float32)
        if i["dtype"] != np.float32:
            s, z = i["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(i["dtype"])
        interpreter.set_tensor(i["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(o["index"])
        if o["dtype"] != np.float32:
            s, z = o["quantization"]
            y = (y.astype(np.float32) - z) * s
        out.append(y[0])
    return np.array(out)


def _run_tflm(tfl: bytes, X: np.ndarray, arena: int) -> np.ndarray:
    from tflite_micro import runtime

    it = runtime.Interpreter.from_bytes(tfl, arena_size=arena)
    d_in = it.get_input_details(0)
    qp = d_in["quantization_parameters"]
    out = []
    for row in X:
        x = row[None].astype(np.float32)
        if d_in["dtype"] != np.float32:
            s, z = float(qp["scales"][0]), int(qp["zero_points"][0])
            x = np.clip(np.round(x / s + z), -128, 127).astype(d_in["dtype"])
        it.set_input(x, 0)
        it.invoke()
        y = it.get_output(0)
        d_out = it.get_output_details(0)
        if d_out["dtype"] != np.float32:
            qo = d_out["quantization_parameters"]
            y = (y.astype(np.float32) - int(qo["zero_points"][0])) * float(qo["scales"][0])
        out.append(np.asarray(y)[0])
    return np.array(out)


def _score(task: str, y: np.ndarray, out: np.ndarray, classes: list[str]) -> dict:
    if task == "classification":
        return classification_metrics(y, out.argmax(1), classes)
    return regression_metrics(y, out[:, 0])


def c_array(data: bytes, name: str) -> str:
    rows = [", ".join(f"0x{b:02x}" for b in data[i : i + 12]) for i in range(0, len(data), 12)]
    body = ",\n  ".join(rows)
    return (
        "/* TFLite model generated by TinyML Studio. Do not edit by hand. */\n"
        "#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n"
        "#ifdef __GNUC__\n#define TM_ALIGN16 __attribute__((aligned(16)))\n#else\n"
        "#define TM_ALIGN16\n#endif\n"
        f"const unsigned char {name}[] TM_ALIGN16 = {{\n  {body}\n}};\n"
        f"const unsigned int {name}_len = {len(data)};\n#endif\n"
    )


def export_tflite(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    int8_io: bool = False,
    flash_kb: float | None = None,
    ram_kb: float | None = None,
    board: str = "esp32-s3",
) -> dict[str, Any]:
    import tensorflow as tf

    run, out = Path(run_dir), Path(out_dir)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    kind = results.get("kind")
    if kind not in ("keras_mlp", "keras_cnn1d"):
        raise ValueError(
            tr(
                "run_dir no es una corrida de 'train-keras' ni 'train-cnn1d' "
                "(falta kind=keras_mlp o keras_cnn1d)",
                "run_dir is not a 'train-keras' or 'train-cnn1d' run "
                "(kind=keras_mlp or keras_cnn1d is missing)",
            )
        )
    wd = None
    recipe = json.loads((run / "recipe_fitted.json").read_text(encoding="utf-8"))
    enc = FeatureEncoder.from_dict(json.loads((run / "encoder.json").read_text(encoding="utf-8")))
    task, seed = results["task"], results["reproducibility"]["seed"]
    model = tf.keras.models.load_model(run / "model.keras")

    if kind == "keras_cnn1d":
        from tinyml_studio.train.cnn1d import prepare_windows
        from tinyml_studio.window import WindowConfig

        wd = prepare_windows(
            recipe,
            WindowConfig.from_dict(results["window"]),
            seed,
            tuple(results.get("features_excluded_by_user", [])),
        )
        prep, enc = wd.prep, wd.enc
        Xtr, Xte, yte = wd.X["train"], wd.X["test"], wd.y["test"]
    else:
        prep = prepare_splits(recipe, seed=seed)
        Xtr = enc.transform(prep.clean["train"])
        Xte = enc.transform(prep.clean["test"])
        yte = (
            enc.encode_target(prep.clean["test"][prep.target])
            if task == "classification"
            else prep.clean["test"][prep.target].to_numpy(dtype="float32")
        )
    rng = np.random.default_rng(seed)
    calib = Xtr[rng.choice(len(Xtr), min(N_CALIBRATION, len(Xtr)), replace=False)]

    emit(
        "convert",
        20,
        tr("Convirtiendo a TFLite (float e int8)…", "Converting to TFLite (float and int8)…"),
    )
    # --- conversión
    f_conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl_float = f_conv.convert()
    q_conv = tf.lite.TFLiteConverter.from_keras_model(model)
    q_conv.optimizations = [tf.lite.Optimize.DEFAULT]
    q_conv.representative_dataset = lambda: ([r[None].astype(np.float32)] for r in calib)
    q_conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if int8_io:
        q_conv.inference_input_type = tf.int8
        q_conv.inference_output_type = tf.int8
    tfl_int8 = q_conv.convert()
    # float16: pesos a la mitad de tamaño, cómputo en float (sirve en Linux/móvil; TFLM de un micro
    # normalmente no lo ejecuta: ver docs). Si la conversión falla no tumba el resto.
    tfl_f16 = None
    try:
        h_conv = tf.lite.TFLiteConverter.from_keras_model(model)
        h_conv.optimizations = [tf.lite.Optimize.DEFAULT]
        h_conv.target_spec.supported_types = [tf.float16]
        tfl_f16 = h_conv.convert()
    except Exception as exc:  # noqa: BLE001
        f16_error = f"{type(exc).__name__}: {exc}"[:200]
    else:
        f16_error = None

    out.mkdir(parents=True, exist_ok=True)
    (out / "model_float.tflite").write_bytes(tfl_float)
    (out / "model_int8.tflite").write_bytes(tfl_int8)
    if tfl_f16 is not None:
        (out / "model_float16.tflite").write_bytes(tfl_f16)

    emit(
        "verify",
        55,
        tr(
            "Verificando Keras ↔ float ↔ int8 ↔ TFLite Micro…",
            "Verifying Keras ↔ float ↔ int8 ↔ TFLite Micro…",
        ),
    )
    # --- verificación Keras vs float vs int8 (vs TFLM)
    p_keras = model.predict(Xte, verbose=0)
    f_it = tf.lite.Interpreter(model_content=tfl_float)
    f_it.allocate_tensors()
    q_it = tf.lite.Interpreter(model_content=tfl_int8)
    q_it.allocate_tensors()
    p_float, p_int8 = _run_tf(f_it, Xte), _run_tf(q_it, Xte)
    ops = _model_ops(tfl_int8)
    arena = _arena_probe(out / "model_int8.tflite") if _has_tflm() else {"measured": False}
    p_tflm = None
    if _has_tflm() and arena.get("measured"):
        p_tflm = _run_tflm(tfl_int8, Xte, TFLM_HOST_ARENA)

    classes = enc.classes
    sc = {
        "keras": _score(task, yte, p_keras, classes),
        "tflite_float": _score(task, yte, p_float, classes),
        "tflite_int8": _score(task, yte, p_int8, classes),
    }
    p_f16 = None
    if tfl_f16 is not None:
        h_it = tf.lite.Interpreter(model_content=tfl_f16)
        h_it.allocate_tensors()
        p_f16 = _run_tf(h_it, Xte)
        sc["tflite_float16"] = _score(task, yte, p_f16, classes)
    if p_tflm is not None:
        sc["tflm_int8"] = _score(task, yte, p_tflm, classes)
    key = "f1_macro" if task == "classification" else "r2"
    warnings: list[str] = []

    def diff(a, b):
        return {"max_abs": float(np.abs(a - b).max()), "mean_abs": float(np.abs(a - b).mean())}

    cmp_ = {
        "float_vs_keras_outputs": diff(p_float, p_keras),
        "int8_vs_float_outputs": diff(p_int8, p_float),
    }
    if p_f16 is not None:
        cmp_["float16_vs_float_outputs"] = diff(p_f16, p_float)
    if task == "classification":
        cmp_["agreement_int8_vs_float"] = float((p_int8.argmax(1) == p_float.argmax(1)).mean())
        cmp_["agreement_int8_vs_keras"] = float((p_int8.argmax(1) == p_keras.argmax(1)).mean())
        if p_tflm is not None:
            cmp_["agreement_tflm_vs_tflite_int8"] = float(
                (p_tflm.argmax(1) == p_int8.argmax(1)).mean()
            )
        recall = {
            c: {
                "keras": sc["keras"]["per_class"][c]["recall"],
                "int8": sc["tflite_int8"]["per_class"][c]["recall"],
                "support": sc["keras"]["per_class"][c]["support"],
            }
            for c in classes
        }
        cmp_["recall_by_class"] = recall
        for c, d in recall.items():
            if d["support"] >= 20 and d["keras"] - d["int8"] > 0.05:
                warnings.append(
                    tr(
                        "int8 pierde {d:.2f} de recall en '{c}'",
                        "int8 loses {d:.2f} of recall on '{c}'",
                        d=d["keras"] - d["int8"],
                        c=c,
                    )
                )
    drop = sc["keras"][key] - sc["tflite_int8"][key]
    if drop > F1_DROP_WARN:
        warnings.append(
            tr(
                "la cuantización int8 baja {key} en {drop:.3f} (> {w})",
                "int8 quantization lowers {key} by {drop:.3f} (> {w})",
                key=key,
                drop=drop,
                w=F1_DROP_WARN,
            )
        )
    if not _has_tflm():
        warnings.append(
            tr(
                "tflite-micro no instalado: No se verificó con el intérprete de TFLite Micro "
                "(pip install -e '.[tflm]')",
                "tflite-micro not installed: Not verified with the TFLite Micro interpreter "
                "(pip install -e '.[tflm]')",
            )
        )

    emit("write", 85, tr("Generando archivos de firmware…", "Generating firmware files…"))
    # --- archivos de firmware
    spec = build_spec(Pipeline.from_list(recipe["cleaning"], fitted=True), enc)
    n_out = len(classes) if task == "classification" else 1
    op_methods = [
        TFLM_ADD.get(o, "Add" + "".join(w.title() for w in o.split("_")))
        for o in dict.fromkeys(ops)
    ]
    arena_sug = int(arena["arena_bytes"] * 1.25 + 1024) if arena.get("measured") else None
    qin = q_it.get_input_details()[0]["quantization"]
    cfg = [
        "#ifndef MODEL_CONFIG_H",
        "#define MODEL_CONFIG_H",
        f"#define TM_N_NUM {len(spec.num_names)}",
        f"#define TM_N_CAT {len(spec.cat_names)}",
        f"#define TM_N_FEATURES {spec.n_features}",
        f"#define TM_N_OUTPUTS {n_out}",
        f"#define TM_TASK_CLASSIFICATION {1 if task == 'classification' else 0}",
        f"#define TM_INT8_IO {1 if int8_io else 0}",
        *([f"#define TM_WINDOW {wd.cfg.window}"] if wd else []),
        f"#define TM_NUM_OPS {len(op_methods)}",
    ]
    if arena_sug:
        cfg.append(
            f"#define TM_ARENA_SIZE {arena_sug}  /* measured {arena['arena_bytes']} B + margin */"
        )
    if int8_io:
        cfg += [
            f"#define TM_INPUT_SCALE {float(qin[0])!r}f",
            f"#define TM_INPUT_ZERO_POINT {int(qin[1])}",
        ]
    cfg.append("#endif")
    (out / "model_config.h").write_text("\n".join(cfg) + "\n", encoding="utf-8")
    (out / "model_data.h").write_text(c_array(tfl_int8, "tm_model_tflite"), encoding="utf-8")
    pre = (
        "\n\n".join(
            [
                '#include <math.h>\n#include "model_config.h"\n#include "tm_prepare.h"',
                _names("TM_NUM_NAMES", spec.num_names),
                _names("TM_CAT_NAMES", spec.cat_names),
                _names("TM_FEATURE_NAMES", spec.feature_names),
                _names(
                    "TM_CLASS_NAMES",
                    classes if task == "classification" else [prep.target],
                ),
                emit_prepare(spec),
            ]
        )
        + "\n"
    )
    (out / "tm_prepare.c").write_text(
        "/* Generated by TinyML Studio: cleaning + encoding. */\n" + pre, encoding="utf-8"
    )
    (out / "tm_prepare.h").write_text(
        '#ifndef TM_PREPARE_H\n#define TM_PREPARE_H\n#include "model_config.h"\n'
        '#ifdef __cplusplus\nextern "C" {\n#endif\n'
        "extern const char *const TM_NUM_NAMES[];\nextern const char *const TM_CAT_NAMES[];\n"
        "extern const char *const TM_FEATURE_NAMES[];\nextern const char *const TM_CLASS_NAMES[];\n"
        "/* num: raw values (NAN if missing); cat: category code\n"
        "   (-1 missing, -2 unknown). */\n"
        "void tm_prepare(const float *num, const int *cat, float *features);\n"
        "#ifdef __cplusplus\n}\n#endif\n#endif\n",
        encoding="utf-8",
    )
    (out / "inference_example.ino").write_text(
        _example_ino(task, op_methods, int8_io, arena_sug, wd is not None), encoding="utf-8"
    )

    # --- vectores de prueba (crudo -> esperado), para repetir en la placa
    raw = prep.raw["test"]
    clean_rows = len(prep.clean["test"])
    aligned = len(raw) == clean_rows and (wd is not None or len(raw) == len(Xte))
    if not aligned:
        warnings.append(
            tr(
                "no se generaron test_vectors.csv: Las filas crudas y limpias del test no "
                "están alineadas (hay una regla de filas o filas sin etiqueta)",
                "test_vectors.csv was not generated: The raw and cleaned test rows are not "
                "aligned (there is a row rule or rows without a label)",
            )
        )
        raw = raw.iloc[0:0]
    num, cat = raw_inputs(spec, raw)
    feats = emulate_features(spec, num, cat) if len(raw) else np.zeros((0, spec.n_features))

    def expected(p):
        if task == "classification":
            return int(p.argmax())
        return float(p[0])

    def fmt(v):
        return "nan" if np.isnan(v) else repr(float(v))

    if wd is not None:
        W, F = wd.cfg.window, spec.n_features
        widx = wd.idx["test"]
        pick = np.unique(np.linspace(0, len(widx) - 1, min(len(widx), 100)).astype(int))
        if not aligned:
            pick = pick[:0]
        wins = feats[widx[pick]] if len(pick) else np.zeros((0, W, F), dtype=np.float32)
        p_dev = _run_tf(q_it, wins) if len(pick) else np.zeros((0, n_out))
        head = []
        for j in range(W):
            head += [f"t{j}_num_{c}" for c in spec.num_names]
            head += [f"t{j}_cat_{c}" for c in spec.cat_names]
        head += [f"f{j}_{k}" for j in range(W) for k in range(F)]
        lines = [",".join(head + ["expected_int8", "expected_keras"])]
        p_k = p_keras[pick] if len(pick) else p_keras[:0]
        for r, w in enumerate(pick):
            cells = []
            for rowi in widx[w]:
                cells += [fmt(v) for v in num[rowi]] + [str(int(v)) for v in cat[rowi]]
            cells += [f"{float(v):.9g}" for v in wins[r].ravel()]
            cells += [str(expected(p_dev[r])), str(expected(p_k[r]))]
            lines.append(",".join(cells))
    else:
        n_vec = min(len(feats), 400)
        p_dev = _run_tf(q_it, feats[:n_vec]) if n_vec else np.zeros((0, n_out))
        lines = [
            ",".join(
                [f"num_{c}" for c in spec.num_names]
                + [f"cat_{c}" for c in spec.cat_names]
                + ["expected_int8", "expected_keras"]
            )
        ]
        for i in range(n_vec):
            lines.append(
                ",".join(
                    [fmt(v) for v in num[i]]
                    + [str(int(v)) for v in cat[i]]
                    + [str(expected(p_dev[i])), str(expected(p_keras[i]))]
                )
            )
    (out / "test_vectors.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    window_c = None
    if wd is not None:
        from .window_c import write_window_c

        window_c = write_window_c(out, spec)
    ram = (arena or {}).get("arena_bytes")
    report = {
        "board": board,
        "io": "int8" if int8_io else "float32",
        "sizes_bytes": {
            "tflite_float": len(tfl_float),
            "tflite_int8": len(tfl_int8),
            **({"tflite_float16": len(tfl_f16)} if tfl_f16 is not None else {}),
        },
        "float16_error": f16_error,
        "ops": list(dict.fromkeys(ops)),
        "tflm_resolver": op_methods,
        "arena": arena,
        "arena_suggested": arena_sug,
        "metrics_test": sc,
        "comparison": cmp_,
        "budget": budget(len(tfl_int8), ram or 0, flash_kb, ram_kb if ram else None),
        "int8_f1_drop": drop,
        "window": wd.cfg.to_dict() if wd else None,
        "window_c_verification": window_c,
        "warnings": warnings + spec.warnings,
    }
    meta = {
        "model": kind,
        "window": wd.cfg.to_dict() if wd else None,
        "task": task,
        "classes": classes,
        "target": prep.target,
        "features": spec.feature_names,
        "inputs_num": spec.num_names,
        "inputs_cat": {c: {k: i for i, k in enumerate(v)} for c, v in spec.cat_values.items()},
        "inputs_cat_encoding": dict(spec.cat_methods),
        "device_preprocessing": {c: [list(o) for o in v] for c, v in spec.ops.items()},
        "io": report["io"],
        "input_quantization": {"scale": float(qin[0]), "zero_point": int(qin[1])},
        "tflm_ops": op_methods,
        "arena_suggested": arena_sug,
        "source_run": results["reproducibility"],
    }
    (out / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _has_tflm() -> bool:
    try:
        import tflite_micro  # noqa: F401

        return True
    except ImportError:
        return False


def _example_ino(
    task: str, ops: list[str], int8_io: bool, arena: int | None, windowed: bool = False
) -> str:
    if windowed:
        return _example_ino_window(task, ops, arena)
    adds = (
        "\n".join(f"  ml.resolver.{o}();" for o in ops)
        or "  // (install tflite-micro to list the ops)"
    )
    cls_post = (
        "  int best = 0;\n  for (int i = 1; i < TM_N_OUTPUTS; i++)\n"
        "    if (ml.outputs[i] > ml.outputs[best]) best = i;\n"
        '  Serial.printf("%s (%.1f%%)\\n", TM_CLASS_NAMES[best], ml.outputs[best] * 100);'
    )
    reg_post = '  Serial.printf("%.4f\\n", ml.outputs[0]);'
    post = cls_post if task == "classification" else reg_post
    note = (
        "// WARNING: this model has int8 I/O; EloquentTinyML expects float.\n"
        "// Re-export without --int8-io or quantize the input by hand with\n"
        "// TM_INPUT_SCALE / TM_INPUT_ZERO_POINT.\n"
        if int8_io
        else ""
    )
    return f"""/*
 * Inference example (TFLite Micro) generated by TinyML Studio. It has NOT been compiled or tested
 * on hardware: it uses the usual EloquentTinyML +
 * tflm_esp32 pattern; check the API against the versions you have installed.
 * Input: RAW values (tm_prepare applies the same cleaning/scaling as training).
 */
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include "model_data.h"
#include "model_config.h"
#include "tm_prepare.h"
{note}
#ifndef TM_ARENA_SIZE
#define TM_ARENA_SIZE {arena or 30000}
#endif
Eloquent::TF::Sequential<TM_NUM_OPS, TM_ARENA_SIZE> ml;

// REPLACE: fill num[] (raw, in TM_NUM_NAMES order) and cat[]
// (category codes, see model_meta.json).
bool readInputs(float *num, int *cat) {{
  for (int i = 0; i < TM_N_NUM; i++) num[i] = NAN;
  for (int i = 0; i < TM_N_CAT; i++) cat[i] = -1;
  return true;
}}

void setup() {{
  Serial.begin(115200);
  delay(1500);
  ml.setNumInputs(TM_N_FEATURES);
  ml.setNumOutputs(TM_N_OUTPUTS);
{adds}
  if (!ml.begin(tm_model_tflite).isOk()) {{
    Serial.println("Error starting model: " + ml.exception.toString());
    while (true) delay(1000);
  }}
}}

void loop() {{
  float num[TM_N_NUM > 0 ? TM_N_NUM : 1];
  int cat[TM_N_CAT > 0 ? TM_N_CAT : 1];
  float x[TM_N_FEATURES];
  if (!readInputs(num, cat)) {{ delay(1000); return; }}
  tm_prepare(num, cat, x);
  if (!ml.predict(x).isOk()) {{
    Serial.println("Inference error: " + ml.exception.toString());
    delay(1000);
    return;
  }}
{post}
  delay(1000);
}}
"""


def _example_ino_window(task: str, ops: list[str], arena: int | None) -> str:
    adds = "\n".join(f"  ml.resolver.{o}();" for o in ops) or "  // (ops from model_meta.json)"
    return f"""/*
 * 1D-CNN example (TFLite Micro) generated by TinyML Studio. It has NOT been compiled or tested
 * on hardware: it uses the usual EloquentTinyML +
 * tflm_esp32 pattern.
 * Each RAW sample goes in through tm_window_push(); once TM_WINDOW consecutive samples are
 * available, inference runs. After a time gap (power-off, pause) call tm_window_reset(): training
 * did not cross large gaps. If there is a row rule (see model_meta.json), apply it first.
 */
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include "model_data.h"
#include "model_config.h"
#include "tm_prepare.h"
#include "tm_window.h"

#ifndef TM_ARENA_SIZE
#define TM_ARENA_SIZE {arena or 30000}
#endif
Eloquent::TF::Sequential<TM_NUM_OPS, TM_ARENA_SIZE> ml;

// REPLACE: fill num[] (raw) and cat[] (codes, see model_meta.json) with the current reading.
bool readInputs(float *num, int *cat) {{
  for (int i = 0; i < TM_N_NUM; i++) num[i] = NAN;
  for (int i = 0; i < TM_N_CAT; i++) cat[i] = -1;
  return true;
}}

void setup() {{
  Serial.begin(115200);
  delay(1500);
  ml.setNumInputs(TM_WINDOW * TM_N_FEATURES);
  ml.setNumOutputs(TM_N_OUTPUTS);
{adds}
  if (!ml.begin(tm_model_tflite).isOk()) {{
    Serial.println("Error starting model: " + ml.exception.toString());
    while (true) delay(1000);
  }}
  tm_window_reset();
}}

void loop() {{
  float num[TM_N_NUM > 0 ? TM_N_NUM : 1];
  int cat[TM_N_CAT > 0 ? TM_N_CAT : 1];
  static float x[TM_WINDOW * TM_N_FEATURES];
  if (!readInputs(num, cat)) {{ delay(1000); return; }}
  tm_window_push(num, cat);
  if (tm_window_ready()) {{
    tm_window_get(x);
    if (!ml.predict(x).isOk()) {{
      Serial.println("Inference error: " + ml.exception.toString());
    }} else {{
      int best = 0;
      for (int i = 1; i < TM_N_OUTPUTS; i++) if (ml.outputs[i] > ml.outputs[best]) best = i;
      Serial.printf("%s (%.1f%%)\\n", TM_CLASS_NAMES[best], ml.outputs[best] * 100);
    }}
  }}
  delay(1000);
}}
"""
