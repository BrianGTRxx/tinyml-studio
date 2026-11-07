"""Exporta un modelo clásico entrenado (``train --out-dir``) a C con verificación Python↔C."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_studio.clean import Pipeline, prepare_splits
from tinyml_studio.i18n import tr
from tinyml_studio.progress import emit
from tinyml_studio.train.features import FeatureEncoder

from .cgen import ARGMAX, EMITTERS, cf, emit_prepare
from .predict import input_ranges
from .spec import build_spec, emulate_features, raw_inputs

MAX_VECTORS_PER_CLASS = 200
MAX_VECTORS = 2000


def _c_str(s: str) -> str:
    """Literal C seguro: ASCII tal cual; UTF-8 y caracteres especiales en octal de 3 dígitos
    (un escape hex como \xf1a se tragaría la letra siguiente)."""
    out = []
    for b in s.encode("utf-8"):
        ch = chr(b)
        if b < 32 or b >= 127 or ch in '"\\?':
            out.append(f"\\{b:03o}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


EMPTY_NAMES = '  ""'


def _names(var: str, names: list[str]) -> str:
    items = ",\n".join(f"  {_c_str(n)}" for n in names)
    return f"const char *const {var}[{max(len(names), 1)}] = {{\n{items or EMPTY_NAMES}\n}};"


def _header(spec, task, model, n_classes) -> tuple[str, str]:
    cfg = [
        "#ifndef MODEL_CONFIG_H",
        "#define MODEL_CONFIG_H",
        f"/* Model: {model} · task: {task} */",
        f"#define TM_N_NUM {len(spec.num_names)}",
        f"#define TM_N_CAT {len(spec.cat_names)}",
        f"#define TM_N_FEATURES {spec.n_features}",
        f"#define TM_N_CLASSES {n_classes}",
        f"#define TM_TASK_CLASSIFICATION {1 if task == 'classification' else 0}",
        "/* Input order: num[] and cat[] as in TM_NUM_NAMES / TM_CAT_NAMES. */",
        "/* cat[i]: category index (see model_meta.json), -1 = missing, -2 = unknown. */",
        "#endif",
    ]
    out_t = "int" if task == "classification" else "float"
    hdr = f"""#ifndef TM_MODEL_H
#define TM_MODEL_H
#include "model_config.h"
#ifdef __cplusplus
extern "C" {{
#endif

extern const char *const TM_NUM_NAMES[];
extern const char *const TM_CAT_NAMES[];
extern const char *const TM_FEATURE_NAMES[];
extern const char *const TM_CLASS_NAMES[];

/* Training-time cleaning + encoding (fitted on train only). */
void tm_prepare(const float *num, const int *cat, float *features);
/* Model on the already prepared feature vector. */
{out_t} tm_predict_features(const float *features);
/* All in one: raw values -> {"class index" if task == "classification" else "value"}.
   With TM_N_CAT == 0, NULL can be passed for cat. */
{out_t} tm_predict(const float *num, const int *cat);
/* How many numeric inputs are OUTSIDE the range seen in training (NAN = missing, not counted).
   Outside that range the model extrapolates and the answer is unreliable: 0 means all fine. */
int tm_out_of_range(const float *num);

#ifdef __cplusplus
}}
#endif
#endif
"""
    return "\n".join(cfg) + "\n", hdr


def _range_c(spec, ranges: dict) -> str:
    """Tablas de mínimo y máximo de entrenamiento y ``tm_out_of_range`` (C99, sin dependencias)."""
    lo, hi = [], []
    for c in spec.num_names:
        r = ranges.get(c)
        lo.append(cf(r["min"]) if r else "(-INFINITY)")
        hi.append(cf(r["max"]) if r else "INFINITY")
    n = max(len(spec.num_names), 1)
    return (
        f"static const float TM_NUM_MIN[{n}] = {{{', '.join(lo) or '0.0f'}}};\n"
        f"static const float TM_NUM_MAX[{n}] = {{{', '.join(hi) or '0.0f'}}};\n\n"
        "int tm_out_of_range(const float *num) {\n"
        "  int n = 0;\n"
        "  for (int i = 0; i < TM_N_NUM; i++)\n"
        "    if (!isnan(num[i]) && (num[i] < TM_NUM_MIN[i] || num[i] > TM_NUM_MAX[i])) n++;\n"
        "  return n;\n"
        "}"
    )


INTEGRATION_MD = """# Integración en el firmware ({model}, {task})

Copia `tm_model.c`, `tm_model.h` y `model_config.h` a tu proyecto (ESP-IDF o Arduino; es C99 y
también compila como C++). No hay dependencias salvo `<math.h>`. **No compiles con `-ffast-math`.**

```c
#include "tm_model.h"

float num[TM_N_NUM];   /* valores CRUDOS en el orden de TM_NUM_NAMES (NAN si falta) */
int   cat[TM_N_CAT > 0 ? TM_N_CAT : 1]; /* códigos de categoría: ver model_meta.json */
/* ... rellena num[] y cat[] con tus lecturas ... */
int clase = tm_predict(num, cat);          /* clasificación: índice en TM_CLASS_NAMES */
```

- La limpieza y el escalado del entrenamiento están dentro de `tm_prepare` (mismos números).
- `tm_out_of_range(num)` devuelve cuántas entradas numéricas salen del rango visto al
  entrenar (0 = todo bien);
  fuera de ese rango el modelo extrapola y el resultado es poco fiable: llámala antes de
  fiarte de `tm_predict`.
- Si `model_meta.json` trae `warnings` (p. ej. una regla de filtro de filas), aplícalas tú antes.
- Antes de confiar: `export_report.json` → `verification` (PC) y **prueba en la placa**
  con `test_vectors.csv`. Esta verificación fue en PC (x86-64), no en el ESP32-S3.
"""


VERIFY_C = r"""/* Host (PC) verification. Build: cc -O2 tm_model.c verify_host.c -lm -o verify
   Run: ./verify test_vectors.csv */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tm_model.h"

#define MAXLINE 65536
int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: verify test_vectors.csv\n"); return 2; }
  FILE *fp = fopen(argv[1], "r");
  if (!fp) { perror("csv"); return 2; }
  static char line[MAXLINE];
  float num[TM_N_NUM > 0 ? TM_N_NUM : 1];
  int cat[TM_N_CAT > 0 ? TM_N_CAT : 1];
  float fexp[TM_N_FEATURES], fgot[TM_N_FEATURES];
  long n = 0, model_ok = 0, raw_ok = 0, pipe_ok = 0;
  double feat_maxdiff = 0.0, model_maxdiff = 0.0;
  if (!fgets(line, MAXLINE, fp)) return 2; /* header */
  while (fgets(line, MAXLINE, fp)) {
    char *tok = strtok(line, ",\n");
    int col = 0;
    double e_model = 0, e_pipe = 0;
    while (tok) {
      if (col < TM_N_NUM) num[col] = strtof(tok, NULL);
      else if (col < TM_N_NUM + TM_N_CAT) cat[col - TM_N_NUM] = atoi(tok);
      else if (col < TM_N_NUM + TM_N_CAT + TM_N_FEATURES)
        fexp[col - TM_N_NUM - TM_N_CAT] = strtof(tok, NULL);
      else if (col == TM_N_NUM + TM_N_CAT + TM_N_FEATURES) e_model = strtod(tok, NULL);
      else e_pipe = strtod(tok, NULL);
      tok = strtok(NULL, ",\n");
      col++;
    }
    tm_prepare(num, cat, fgot);
    for (int i = 0; i < TM_N_FEATURES; i++) {
      double d = fabs((double)fgot[i] - (double)fexp[i]);
      double s = fabs((double)fexp[i]); if (s < 1.0) s = 1.0;
      if (d / s > feat_maxdiff) feat_maxdiff = d / s;
    }
#if TM_TASK_CLASSIFICATION
    int a = tm_predict_features(fexp), b = tm_predict(num, cat);
    if (a == (int)e_model) model_ok++;
    if (b == (int)e_model) raw_ok++;
    if (b == (int)e_pipe) pipe_ok++;
#else
    float a = tm_predict_features(fexp), b = tm_predict(num, cat);
    double dm = fabs((double)a - e_model); if (dm > model_maxdiff) model_maxdiff = dm;
    double sc = fabs(e_model) > 1.0 ? fabs(e_model) : 1.0;
    if (dm / sc < 1e-4) model_ok++;
    if (fabs((double)b - e_model) / sc < 1e-3) raw_ok++;
    if (fabs((double)b - e_pipe) / (fabs(e_pipe) > 1.0 ? fabs(e_pipe) : 1.0) < 1e-2) pipe_ok++;
#endif
    n++;
  }
  fclose(fp);
  printf("VERIFY n=%ld feat_maxrel=%.3g model_ok=%ld raw_ok=%ld pipeline_ok=%ld "
         "model_maxabs=%.3g\n",
         n, feat_maxdiff, model_ok, raw_ok, pipe_ok, model_maxdiff);
  return 0;
}
"""


def _sample_rows(y: np.ndarray, classification: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if classification:
        idx = []
        for k in np.unique(y):
            ids = np.flatnonzero(y == k)
            idx.append(rng.choice(ids, min(len(ids), MAX_VECTORS_PER_CLASS), replace=False))
        idx = np.concatenate(idx)
    else:
        idx = rng.choice(len(y), min(len(y), MAX_VECTORS), replace=False)
    return np.sort(idx[:MAX_VECTORS])


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)


def verify_c(out_dir: Path, cc: str = "cc") -> dict[str, Any]:
    """Compila con -Wall -Wextra (C99 y C++) y corre los vectores. Sin compilador -> unavailable."""
    if shutil.which(cc) is None:
        return {
            "available": False,
            "reason": tr("no hay compilador '{cc}'", "no compiler '{cc}'", cc=cc),
        }
    res: dict[str, Any] = {"available": True}
    exe = out_dir / "verify_host"
    c = _run(
        [
            cc,
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "tm_model.c",
            "verify_host.c",
            "-lm",
            "-o",
            exe.name,
        ],
        out_dir,
    )
    res["compile_ok"] = c.returncode == 0
    res["compile_warnings"] = [ln for ln in c.stderr.splitlines() if "warning" in ln]
    if c.returncode != 0:
        exe.unlink(missing_ok=True)
        res["compile_error"] = c.stderr[-1500:]
        return res
    r = _run([str(exe), "test_vectors.csv"], out_dir)
    exe.unlink(missing_ok=True)  # el binario de PC no se entrega
    m = re.search(
        r"VERIFY n=(\d+) feat_maxrel=(\S+) model_ok=(\d+) raw_ok=(\d+) pipeline_ok=(\d+) "
        r"model_maxabs=(\S+)",
        r.stdout,
    )
    if r.returncode != 0 or not m:
        res["run_error"] = (r.stderr or r.stdout)[-800:]
        return res
    n = int(m[1])
    res.update(
        vectors=n,
        features_max_rel_diff=float(m[2]),
        model_match=int(m[3]) / n,
        raw_to_class_match=int(m[4]) / n,
        vs_full_python_pipeline=int(m[5]) / n,
        model_max_abs_diff=float(m[6]),
    )
    if shutil.which("g++"):
        cpp = _run(["g++", "-fsyntax-only", "-x", "c++", "-Wall", "tm_model.c"], out_dir)
        res["compiles_as_cpp"] = cpp.returncode == 0
    with tempfile.TemporaryDirectory() as td:
        obj = Path(td) / "m.o"
        if (
            _run([cc, "-std=c99", "-Os", "-c", "tm_model.c", "-o", str(obj)], out_dir).returncode
            == 0
        ):
            sz = shutil.which("size")
            if sz:
                o = subprocess.run([sz, str(obj)], capture_output=True, text=True).stdout.split()
                if len(o) >= 9:
                    res["host_x86_64_Os_bytes"] = {
                        "text": int(o[6]),
                        "data": int(o[7]),
                        "note": tr("orientativo: NO es Xtensa", "indicative only: NOT Xtensa"),
                    }
    return res


def budget(data_bytes: int, ram_bytes: int, flash_kb: float | None, ram_kb: float | None) -> dict:
    def one(used: int, kb: float | None) -> dict:
        if kb is None:
            return {
                "used_bytes": used,
                "budget_kb": None,
                "verdict": "sin presupuesto",
                "verdict_code": "no_budget",
                "verdict_label": tr("sin presupuesto", "no budget"),
            }
        frac = used / (kb * 1024)
        if frac > 1:
            code, es, en = "no_fit", "no cabe", "does not fit"
        elif frac > 0.8:
            code, es, en = "tight", "justo (>80 %)", "tight (>80 %)"
        else:
            code, es, en = "fits", "cabe", "fits"
        return {
            "used_bytes": used,
            "budget_kb": kb,
            "fraction": round(frac, 4),
            "verdict": es,  # código estable en español (la UI y los tests lo comparan)
            "verdict_code": code,
            "verdict_label": tr(es, en),
        }

    return {
        "flash_data": one(data_bytes, flash_kb),
        "ram_working": one(ram_bytes, ram_kb),
        "note": tr(
            "flash = solo tablas del modelo; falta el código (varios KB) y el resto de tu "
            "firmware. RAM = buffers de trabajo del modelo, no incluye pila de la tarea ni WiFi.",
            "flash = model tables only; the code (several KB) and the rest of your firmware "
            "are not included. RAM = the model's working buffers, excluding task stack and WiFi.",
        ),
    }


def export_run(
    run_dir: str | Path,
    out_dir: str | Path,
    model: str | None = None,
    *,
    flash_kb: float | None = None,
    ram_kb: float | None = None,
    board: str = "esp32-s3",
    compact: bool = False,
) -> dict[str, Any]:
    import joblib

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
    if name not in EMITTERS:
        raise ValueError(
            tr(
                "el modelo '{name}' no se puede exportar a C (guarda el entrenamiento completo "
                "o su cómputo es demasiado grande para un microcontrolador); elige uno "
                "exportable: {ok}",
                "model '{name}' cannot be exported to C (it stores the whole training set or "
                "its computation is too large for a microcontroller); pick an exportable "
                "one: {ok}",
                name=name,
                ok=", ".join(sorted(EMITTERS)),
            )
        )
    est = joblib.load(path)
    pipeline = Pipeline.from_list(recipe["cleaning"], fitted=True)
    spec = build_spec(pipeline, enc)
    ranges = input_ranges(run, recipe, results, spec)
    n_classes = len(enc.classes) if task == "classification" else 1

    emit("generate", 15, tr("Generando el C del modelo…", "Generating the model C code…"))
    emitter = EMITTERS[name]
    if compact and name in ("decision_tree", "random_forest", "extra_trees"):
        from .tinyforest import emit_compact_trees

        emitter = emit_compact_trees  # modo opcional; por defecto el C no cambia
    body, data_bytes, ram_model, info = emitter(est, name, spec, task, n_classes)
    cfg_h, api_h = _header(spec, task, name, n_classes)
    out_t = "int" if task == "classification" else "float"
    c_src = (
        "\n\n".join(
            [
                f"/* Generated by TinyML Studio. Model: {name} ({task}). Do not edit by hand. */",
                "#include <math.h>\n#include <stddef.h>\n#include <stdint.h>\n"
                '#include "tm_model.h"',
                _names("TM_NUM_NAMES", spec.num_names),
                _names("TM_CAT_NAMES", spec.cat_names),
                _names("TM_FEATURE_NAMES", spec.feature_names),
                _names(
                    "TM_CLASS_NAMES",
                    spec.classes if task == "classification" else [recipe["target"] or "y"],
                ),
                ARGMAX.replace("static int tm_argmax", "static int tm_argmax"),
                emit_prepare(spec),
                _range_c(spec, ranges),
                body,
                f"{out_t} tm_predict_features(const float *features) {{\n"
                "  return tm_model(features);\n}",
                f"{out_t} tm_predict(const float *num, const int *cat) {{\n"
                "  float f[TM_N_FEATURES];\n  tm_prepare(num, cat, f);\n  return tm_model(f);\n}",
            ]
        )
        + "\n"
    )
    # tm_argmax solo si se usa (evita -Wunused-function)
    if "tm_argmax(" not in c_src.split("static int tm_argmax", 1)[1].split("}", 1)[1]:
        c_src = c_src.replace(ARGMAX + "\n\n", "", 1)

    # ---- vectores de prueba (test): crudos, features emuladas, salida de sklearn
    prep = prepare_splits(recipe, seed=results["reproducibility"]["seed"])
    raw_t, clean_t = prep.raw["test"], prep.clean["test"]
    y = (
        enc.encode_target(clean_t[prep.target])
        if task == "classification"
        else clean_t[prep.target].to_numpy(dtype="float64")
    )
    sel = _sample_rows(y, task == "classification", results["reproducibility"]["seed"])
    # los pasos de fila (duplicados) solo tocan train, así que raw_t y clean_t están alineados
    # salvo filter_rule: en ese caso se usa la limpieza completa (is_train=False) ya aplicada.
    num, cat = raw_inputs(spec, raw_t)
    if len(raw_t) != len(clean_t):
        raise ValueError(
            tr(
                "filter_rule cambia el número de filas del test: Exportación no soportada aún",
                "filter_rule changes the number of test rows: export not supported yet",
            )
        )
    feats = emulate_features(spec, num, cat)
    exp_model = est.predict(feats)
    exp_pipe = est.predict(enc.transform(clean_t))
    agree_emul = (
        float((exp_model == exp_pipe).mean())
        if task == "classification"
        else float(
            np.mean(np.abs(exp_model - exp_pipe) <= 1e-2 * np.maximum(1.0, np.abs(exp_pipe)))
        )
    )
    cols = (
        [f"num_{c}" for c in spec.num_names]
        + [f"cat_{c}" for c in spec.cat_names]
        + [f"f_{i}" for i in range(spec.n_features)]
        + ["expected_model", "expected_pipeline"]
    )
    lines = [",".join(cols)]
    for i in sel:
        row = [("nan" if np.isnan(v) else repr(float(v))) for v in num[i]]
        row += [str(int(v)) for v in cat[i]]
        row += [f"{float(v):.9g}" for v in feats[i]]
        row += [repr(float(exp_model[i])), repr(float(exp_pipe[i]))]
        lines.append(",".join(row))

    out.mkdir(parents=True, exist_ok=True)
    (out / "model_config.h").write_text(cfg_h, encoding="utf-8")
    (out / "tm_model.h").write_text(api_h, encoding="utf-8")
    (out / "tm_model.c").write_text(c_src, encoding="utf-8")
    (out / "verify_host.c").write_text(VERIFY_C, encoding="utf-8")
    (out / "test_vectors.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ram_total = ram_model + 4 * spec.n_features
    meta = {
        "model": name,
        "task": task,
        "classes": spec.classes,
        "target": prep.target,
        "inputs": {
            "num": spec.num_names,
            "cat": {
                c: {
                    "codes": {k: i for i, k in enumerate(v)},
                    "missing": -1,
                    "unknown": -2,
                    "missing_uses": spec.cat_fill[c],
                    "encoding": spec.cat_methods.get(c, "onehot"),
                }
                for c, v in spec.cat_values.items()
            },
        },
        "features": spec.feature_names,
        "device_preprocessing": {c: [list(op) for op in ops] for c, ops in spec.ops.items()},
        "warnings": spec.warnings,
        "input_ranges": ranges,
        "model_info": info,
        "source_run": {
            "seed": results["reproducibility"]["seed"],
            "recipe_sha256": results["reproducibility"]["recipe_sha256"],
            "data_sha256": results["reproducibility"]["data_sha256"],
        },
    }
    emit(
        "verify",
        70,
        tr("Compilando y verificando Python ↔ C…", "Compiling and verifying Python ↔ C…"),
    )
    verification = verify_c(out)
    report = {
        "board": board,
        "data_bytes": data_bytes,
        "ram_bytes_estimate": ram_total,
        "budget": budget(data_bytes, ram_total, flash_kb, ram_kb),
        "verification": verification,
        "python_emulation_vs_full_pipeline": agree_emul,
        "python_test_metrics": results["models"][name]["test"].get(
            "f1_macro", results["models"][name]["test"].get("r2")
        ),
        "warnings": spec.warnings,
    }
    if compact:
        report["compact"] = name in ("decision_tree", "random_forest", "extra_trees")
    (out / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
