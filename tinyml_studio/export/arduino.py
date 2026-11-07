"""Genera carpetas de sketch de Arduino listas para abrir a partir de una exportación existente.

Variantes:
- ``raw_c``: exportación clásica (``tm_model.c/.h``) -> sketch sin dependencias; se COMPILA en PC
  contra un shim de Arduino (``Serial`` simulado) y se comprueba que la predicción impresa coincide
  con la esperada de ``test_vectors.csv``.
- ``eloquent``: TFLite con el patrón EloquentTinyML + tflm_esp32 (plantilla del autor).
- ``tflm``: TFLite Micro "crudo" (MicroMutableOpResolver + MicroInterpreter).

Los sketches TFLite NO se compilan aquí (no hay toolchain de Arduino y la API depende de la
versión de las librerías): ``verification.level == "not_compiled"``.
No se escribe ninguna credencial ni dato de red.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tinyml_studio.i18n import tr

from .tflite import TFLM_ADD, c_array

VARIANTS = ("auto", "raw_c", "eloquent", "tflm")
DEFAULT_ARENA = 30000
SERIAL_BAUD = 115200

# ---------------------------------------------------------------- lectura de la exportación


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name.strip()) or "tinyml_sketch"
    return s if re.match(r"[A-Za-z]", s) else f"tm_{s}"


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _detect(export_dir: Path) -> str:
    if (export_dir / "tm_model.c").exists():
        return "raw_c"
    if any((export_dir / f).exists() for f in ("model_int8.tflite", "model_float.tflite")):
        return "tflite"
    raise FileNotFoundError(
        tr(
            "{d} no parece una exportación (falta tm_model.c o model_int8.tflite)",
            "{d} does not look like an export (tm_model.c or model_int8.tflite is missing)",
            d=export_dir,
        )
    )


def _first_vector(export_dir: Path, expected_col: str) -> dict[str, Any]:
    """Primera fila de test_vectors.csv: entradas crudas (num/cat) y salida esperada."""
    p = export_dir / "test_vectors.csv"
    if not p.exists():
        raise FileNotFoundError(tr("falta test_vectors.csv", "test_vectors.csv is missing"))
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        raise ValueError(tr("test_vectors.csv está vacío", "test_vectors.csv is empty"))
    head, row = lines[0].split(","), lines[1].split(",")
    cell = dict(zip(head, row, strict=True))
    num = [cell[h] for h in head if h.startswith("num_")]
    cat = [int(cell[h]) for h in head if h.startswith("cat_")]
    return {"num": num, "cat": cat, "expected": cell[expected_col]}


def _c_float(s: str) -> str:
    v = float(s)
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return "INFINITY" if v > 0 else "(-INFINITY)"
    return f"{v!r}f"


def _input_table(num: list[str], cat: dict[str, Any], units: dict[str, str] | None) -> list[dict]:
    rows: list[dict[str, Any]] = []
    for i, n in enumerate(num):
        r: dict[str, Any] = {"index": i, "name": n, "array": "num"}
        if units and n in units:
            r["unit"] = units[n]
        rows.append(r)
    for i, (n, spec) in enumerate(cat.items()):
        codes = spec.get("codes", spec) if isinstance(spec, dict) else spec
        r = {"index": i, "name": n, "array": "cat", "codes": codes}
        if units and n in units:
            r["unit"] = units[n]
        rows.append(r)
    return rows


def _sample_decl(vec: dict[str, Any]) -> str:
    nums = ", ".join(_c_float(v) for v in vec["num"]) or "0.0f"
    cats = ", ".join(str(v) for v in vec["cat"]) or "0"
    return (
        f"static const float SAMPLE_NUM[{max(len(vec['num']), 1)}] = {{{nums}}};\n"
        f"static const int SAMPLE_CAT[{max(len(vec['cat']), 1)}] = {{{cats}}};"
    )


# ---------------------------------------------------------------- raw_c


def _ino_raw_c(task: str, expected: str) -> tuple[str, str]:
    if task == "classification":
        run = (
            "  int pred = tm_predict(SAMPLE_NUM, SAMPLE_CAT);\n"
            '  Serial.print("pred=");\n  Serial.println(pred);\n'
            '  Serial.print("expected=");\n  Serial.println(EXPECTED);\n'
            '  Serial.print("class=");\n'
            "  Serial.println(pred >= 0 && pred < TM_N_CLASSES ? TM_CLASS_NAMES[pred] : "
            '"?");'
        )
        exp_decl = f"static const int EXPECTED = {int(float(expected))};"
    else:
        run = (
            "  float pred = tm_predict(SAMPLE_NUM, SAMPLE_CAT);\n"
            '  Serial.print("pred=");\n  Serial.println(pred, 6);\n'
            '  Serial.print("expected=");\n  Serial.println(EXPECTED, 6);'
        )
        exp_decl = f"static const float EXPECTED = {_c_float(expected)};"
    return run, exp_decl


def _build_raw_c(export_dir: Path, sk: Path, name: str, meta: dict) -> dict[str, Any]:
    task = meta.get("task", "classification")
    vec = _first_vector(export_dir, "expected_model")
    for f in ("tm_model.c", "tm_model.h", "model_config.h"):
        shutil.copy2(export_dir / f, sk / f)
    run, exp_decl = _ino_raw_c(task, vec["expected"])
    ino = f"""/*
 * {name}: classical ML model exported by TinyML Studio (plain C, no external libraries).
 * Feeds one embedded RAW sample (from test_vectors.csv) to tm_predict() and prints the result
 * over Serial at {SERIAL_BAUD} baud. Replace SAMPLE_NUM / SAMPLE_CAT with your own readings,
 * in the order shown in README.txt (raw units, NAN when missing).
 * tm_predict() already applies the same cleaning/encoding used in training.
 */
#include <math.h>
#include "tm_model.h"

// Raw sample taken from the export's test vectors, and the output the model must give.
{_sample_decl(vec)}
{exp_decl}

void setup() {{
  Serial.begin({SERIAL_BAUD});
  delay(1500);
  Serial.println("TinyML Studio: classical model");
}}

void loop() {{
{run}
  delay(2000);
}}
"""
    (sk / f"{name}.ino").write_text(ino, encoding="utf-8")
    inputs = meta.get("inputs", {})
    return {
        "files": ["tm_model.c", "tm_model.h", "model_config.h"],
        "num": inputs.get("num", []),
        "cat": inputs.get("cat", {}),
        "task": task,
        "classes": meta.get("classes", []),
    }


# ---------------------------------------------------------------- host shim + verificación

_SHIM_ARDUINO_H = r"""// Host shim to compile an Arduino sketch on a PC (NOT the real Arduino core).
#ifndef ARDUINO_H_SHIM
#define ARDUINO_H_SHIM
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
struct MockSerial {
  void begin(unsigned long) {}
  void print(const char *s) { fputs(s, stdout); }
  void print(char c) { fputc(c, stdout); }
  void print(int v) { printf("%d", v); }
  void print(long v) { printf("%ld", v); }
  void print(unsigned int v) { printf("%u", v); }
  void print(double v, int d = 2) { printf("%.*f", d, v); }
  void println() { fputc('\n', stdout); }
  template <typename T> void println(T v) { print(v); fputc('\n', stdout); }
  void println(double v, int d) { print(v, d); fputc('\n', stdout); }
};
static MockSerial Serial;
static inline void delay(unsigned long) {}
static inline unsigned long millis() { return 0; }
void setup();
void loop();
#endif
"""

_SHIM_MAIN = """#include "Arduino.h"
int main() { setup(); loop(); return 0; }
"""


def _host_check(sk: Path, name: str, task: str, expected: str, cc: str, cxx: str) -> dict:
    if not (shutil.which(cc) and shutil.which(cxx)):
        return {
            "level": "not_compiled",
            "detail": tr(
                "no hay compilador C/C++ en esta máquina; el sketch no se compiló",
                "no C/C++ compiler on this machine; the sketch was not compiled",
            ),
        }
    with tempfile.TemporaryDirectory() as td:
        shim = Path(td)
        (shim / "Arduino.h").write_text(_SHIM_ARDUINO_H, encoding="utf-8")
        (shim / "shim_main.cpp").write_text(_SHIM_MAIN, encoding="utf-8")
        exe = shim / "host_sketch"
        obj = shim / "tm_model.o"

        def run(cmd: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        c = run([cc, "-std=c99", "-O2", "-Wall", "-c", str(sk / "tm_model.c"), "-o", str(obj)])
        cmd = [
            cxx,
            "-Wall",
            "-O2",
            f"-I{shim}",
            f"-I{sk}",
            "-include",
            "Arduino.h",  # the Arduino IDE injects this automatically
            "-x",
            "c++",
            str(sk / f"{name}.ino"),
            str(shim / "shim_main.cpp"),
            "-x",
            "none",
            str(obj),
            "-lm",
            "-o",
            str(exe),
        ]
        g = run(cmd) if c.returncode == 0 else c
        if c.returncode != 0 or g.returncode != 0:
            return {
                "level": "not_compiled",
                "detail": tr(
                    "FALLÓ la compilación en PC: {e}",
                    "host compilation FAILED: {e}",
                    e=(g.stderr or c.stderr)[-600:],
                ),
            }
        r = run([str(exe)])
    m = re.search(r"pred=(\S+)", r.stdout)
    e = re.search(r"expected=(\S+)", r.stdout)
    if r.returncode != 0 or not m or not e:
        return {
            "level": "not_compiled",
            "detail": tr(
                "compiló pero la ejecución en PC falló: {o}",
                "compiled but the host run failed: {o}",
                o=(r.stdout + r.stderr)[-300:],
            ),
        }
    if task == "classification":
        ok = int(m[1]) == int(e[1]) == int(float(expected))
    else:
        ok = abs(float(m[1]) - float(expected)) <= 1e-3 * max(1.0, abs(float(expected)))
    if not ok:
        return {
            "level": "not_compiled",
            "detail": tr(
                "compiló pero la predicción ({p}) NO coincide con la esperada ({x})",
                "compiled but the prediction ({p}) does NOT match the expected one ({x})",
                p=m[1],
                x=expected,
            ),
        }
    return {
        "level": "host_compiled",
        "detail": tr(
            "compilado con {cxx} contra un shim de Arduino (Serial simulado) y ejecutado: "
            "predicción {p} = esperada. NO es el núcleo de Arduino ni un ESP32-S3.",
            "compiled with {cxx} against an Arduino shim (mock Serial) and run: Prediction "
            "{p} = expected. This is NOT the Arduino core nor an ESP32-S3.",
            cxx=cxx,
            p=m[1],
        ),
    }


# ---------------------------------------------------------------- TFLite


def _camel(op: str) -> str:
    return "Add" + "".join(w.capitalize() for w in op.split("_"))


def _tflite_bytes(export_dir: Path, which: str) -> tuple[bytes, str]:
    f = export_dir / f"model_{which}.tflite"
    if f.exists():
        return f.read_bytes(), f.name
    if which == "int8" and (export_dir / "model_data.h").exists():
        nums = re.findall(r"0x([0-9a-f]{2})", (export_dir / "model_data.h").read_text())
        return bytes(int(h, 16) for h in nums), "model_data.h"
    raise FileNotFoundError(f"model_{which}.tflite")


def _ops_of(tfl: bytes, meta: dict) -> tuple[list[str], list[str]]:
    """(métodos AddXxx únicos, nombres de op del esquema). Usa el flatbuffer; si no se puede,
    cae al ``tflm_ops`` de model_meta.json."""
    try:
        from .tflite import _model_ops

        names = list(dict.fromkeys(_model_ops(tfl)))
    except Exception:
        return list(dict.fromkeys(meta.get("tflm_ops", []))), []
    return [TFLM_ADD.get(n) or _camel(n) for n in names], names


def _post_snippet(task: str, indent: str = "  ") -> str:
    if task == "classification":
        body = (
            "int best = 0;\n"
            "for (int i = 1; i < N_OUTPUTS; i++) if (y[i] > y[best]) best = i;\n"
            'Serial.print("pred=");\nSerial.println(best);\n'
            'Serial.print("class=");\nSerial.println(CLASS_NAMES[best]);\n'
            'Serial.print("score=");\nSerial.println(y[best], 4);'
        )
    else:
        body = 'Serial.print("pred=");\nSerial.println(y[0], 6);'
    return "\n".join(indent + ln for ln in body.splitlines())


def _expected_decl(task: str, expected: str) -> str:
    if task == "classification":
        return f"static const int EXPECTED = {int(float(expected))};"
    return f"static const float EXPECTED = {_c_float(expected)};"


def _common_decls(meta: dict, task: str, vec: dict, n_out: int, n_ops: int, arena: int) -> str:
    classes = meta.get("classes", [])
    names = ", ".join(json.dumps(c, ensure_ascii=True) for c in classes) or '""'
    return f"""#define N_OUTPUTS {n_out}
#define NUM_OPS   {n_ops}      // distinct ops registered below
#define ARENA_SIZE {arena}   // bytes; start here and raise if tf/interpreter allocation fails
static const char *CLASS_NAMES[{max(len(classes), 1)}] = {{{names}}};
// Raw sample taken from the export's test vectors, and the output the model must give.
{_sample_decl(vec)}
{_expected_decl(task, vec["expected"])}"""


def _ino_eloquent(name: str, meta: dict, task: str, vec: dict, ops: list[str], arena: int) -> str:
    n_out = len(meta.get("classes", [])) if task != "regression" else 1
    adds = "\n".join(f"  tf.resolver.{o}();" for o in ops)
    return f"""/*
 * {name}: TFLite model exported by TinyML Studio, EloquentTinyML + tflm_esp32 variant.
 * NOT compiled by TinyML Studio (no Arduino toolchain, library APIs change between versions):
 * check it against the library versions you installed. See README.txt.
 * Runs one embedded RAW sample through tm_prepare() (same cleaning as training) and predicts.
 */
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include <math.h>
#include "model.h"        // const unsigned char model_data[] = {{...}};
#include "model_config.h"
#include "tm_prepare.h"

{_common_decls(meta, task, vec, n_out, len(ops), arena)}

Eloquent::TF::Sequential<NUM_OPS, ARENA_SIZE> tf;

void setup() {{
  Serial.begin({SERIAL_BAUD});
  delay(1500);
  tf.setNumInputs(TM_N_FEATURES);
  tf.setNumOutputs(N_OUTPUTS);
  // Register ONLY the ops this model uses (read from the .tflite file).
{adds}
  if (!tf.begin(model_data).isOk()) {{
    Serial.println("Model init error: " + tf.exception.toString());
    while (true) delay(1000);
  }}
  Serial.println("Model ready");
}}

void loop() {{
  float x[TM_N_FEATURES];
  float y[N_OUTPUTS];
  tm_prepare(SAMPLE_NUM, SAMPLE_CAT, x);
  if (!tf.predict(x, y).isOk()) {{
    Serial.println("Inference error: " + tf.exception.toString());
    delay(2000);
    return;
  }}
{_post_snippet(task)}
  Serial.print("expected=");
  Serial.println(EXPECTED);
  delay(2000);
}}
"""


def _ino_tflm(name: str, meta: dict, task: str, vec: dict, ops: list[str], arena: int) -> str:
    n_out = len(meta.get("classes", [])) if task != "regression" else 1
    adds = "\n".join(f"  resolver.{o}();" for o in ops)
    return f"""/*
 * {name}: TFLite model exported by TinyML Studio, raw TensorFlow Lite Micro variant.
 * NOT compiled by TinyML Studio (no Arduino toolchain, header paths/APIs change between TFLM
 * versions): check it against the library you installed. See README.txt.
 * Input quantization (scale / zero point) is read from the model itself at run time.
 */
#include <tflm_esp32.h>   // provides the TensorFlow Lite Micro headers on ESP32
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <math.h>
#include "model.h"        // const unsigned char model_data[] = {{...}};
#include "model_config.h"
#include "tm_prepare.h"

{_common_decls(meta, task, vec, n_out, len(ops), arena)}

static tflite::MicroMutableOpResolver<NUM_OPS> resolver;
alignas(16) static uint8_t tensor_arena[ARENA_SIZE];
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;

void setup() {{
  Serial.begin({SERIAL_BAUD});
  delay(1500);
  const tflite::Model *model = tflite::GetModel(model_data);
{adds}
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, ARENA_SIZE);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {{
    Serial.println("AllocateTensors failed: increase ARENA_SIZE");
    while (true) delay(1000);
  }}
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("Model ready");
}}

static float read_output(int i) {{
  if (output->type == kTfLiteInt8)
    return (output->data.int8[i] - output->params.zero_point) * output->params.scale;
  return output->data.f[i];
}}

void loop() {{
  float x[TM_N_FEATURES];
  float y[N_OUTPUTS];
  tm_prepare(SAMPLE_NUM, SAMPLE_CAT, x);
  for (int i = 0; i < TM_N_FEATURES; i++) {{
    if (input->type == kTfLiteInt8) {{
      // q = round(x / scale) + zero_point, saturated to int8 (scale/zero point read from the model)
      long q = lroundf(x[i] / input->params.scale) + input->params.zero_point;
      input->data.int8[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }} else {{
      input->data.f[i] = x[i];
    }}
  }}
  if (interpreter->Invoke() != kTfLiteOk) {{
    Serial.println("Invoke failed");
    delay(2000);
    return;
  }}
  for (int i = 0; i < N_OUTPUTS; i++) y[i] = read_output(i);
{_post_snippet(task)}
  Serial.print("expected=");
  Serial.println(EXPECTED);
  delay(2000);
}}
"""


def _build_tflite(
    export_dir: Path,
    sk: Path,
    name: str,
    meta: dict,
    variant: str,
    model: str,
    arena_size: int | None,
    warnings: list[str],
) -> dict[str, Any]:
    if meta.get("window"):
        raise ValueError(
            tr(
                "los modelos con ventana (1D-CNN) aún no tienen sketch de Arduino automático",
                "windowed (1D-CNN) models do not have an automatic Arduino sketch yet",
            )
        )
    task = meta.get("task", "classification")
    vec = _first_vector(export_dir, "expected_int8")
    tfl, src = _tflite_bytes(export_dir, model)
    ops, schema_ops = _ops_of(tfl, meta)
    if not ops:
        raise ValueError(
            tr(
                "no se pudo leer la lista de ops del modelo (instala tflite-micro o usa un "
                "model_meta.json con tflm_ops)",
                "could not read the model's op list (install tflite-micro or provide a "
                "model_meta.json with tflm_ops)",
            )
        )
    unknown = [n for n in schema_ops if n not in TFLM_ADD]
    if unknown:
        warnings.append(
            tr(
                "ops sin nombre verificado en el exportador ({u}): El método AddXxx se dedujo "
                "del nombre; compruébalo al compilar",
                "ops without a verified name in the exporter ({u}): The AddXxx method was "
                "derived from the name; check it when compiling",
                u=", ".join(unknown),
            )
        )
    arena = arena_size or meta.get("arena_suggested") or DEFAULT_ARENA
    if not (arena_size or meta.get("arena_suggested")):
        warnings.append(
            tr(
                "no hay arena medida: Se usa {a} B por defecto; ajústala",
                "no measured arena: Defaulting to {a} B; tune it",
                a=DEFAULT_ARENA,
            )
        )
    if variant == "eloquent" and meta.get("io") == "int8":
        warnings.append(
            tr(
                "este modelo tiene E/S int8 y EloquentTinyML espera float: Usa la variante "
                "'tflm' o re-exporta sin --int8-io",
                "this model has int8 I/O and EloquentTinyML expects float: Use the 'tflm' "
                "variant or re-export without --int8-io",
            )
        )
    (sk / "model.h").write_text(c_array(tfl, "model_data"), encoding="utf-8")
    for f in ("model_config.h", "tm_prepare.c", "tm_prepare.h"):
        shutil.copy2(export_dir / f, sk / f)
    gen = _ino_eloquent if variant == "eloquent" else _ino_tflm
    (sk / f"{name}.ino").write_text(gen(name, meta, task, vec, ops, arena), encoding="utf-8")
    cat = meta.get("inputs_cat", {})
    return {
        "files": ["model.h", "model_config.h", "tm_prepare.c", "tm_prepare.h"],
        "num": meta.get("inputs_num", []),
        "cat": cat,
        "task": task,
        "ops": ops,
        "arena": arena,
        "source": src,
    }


# ---------------------------------------------------------------- README y pasos


def _steps(variant: str, name: str, info: dict) -> list[str]:
    s = [
        tr(
            "Instala el IDE de Arduino y, en Gestor de placas, la plataforma «esp32» "
            "de Espressif Systems.",
            'Install the Arduino IDE and, in Boards Manager, the "esp32" platform by '
            "Espressif Systems.",
        )
    ]
    if variant == "eloquent":
        s.append(
            tr(
                "En Gestor de bibliotecas instala «EloquentTinyML» y «tflm_esp32».",
                'In Library Manager install "EloquentTinyML" and "tflm_esp32".',
            )
        )
    elif variant == "tflm":
        s.append(
            tr(
                "En Gestor de bibliotecas instala «tflm_esp32» (trae TensorFlow Lite Micro).",
                'In Library Manager install "tflm_esp32" (it ships TensorFlow Lite Micro).',
            )
        )
    s += [
        tr(
            "Abre {n}/{n}.ino (la carpeta debe llamarse igual que el .ino).",
            "Open {n}/{n}.ino (the folder must have the same name as the .ino).",
            n=name,
        ),
        tr(
            "Placa: «ESP32S3 Dev Module». Si el modelo es grande, revisa Partition Scheme; "
            "activa PSRAM solo si tu módulo la tiene.",
            'Board: "ESP32S3 Dev Module". For a large model check Partition Scheme; enable '
            "PSRAM only if your module has it.",
        ),
    ]
    if variant != "raw_c":
        s.append(
            tr(
                "Arena de tensores: ARENA_SIZE = {a} B es un punto de partida; si "
                "falla la reserva, súbela con cuidado (la RAM del ESP32-S3 es limitada).",
                "Tensor arena: ARENA_SIZE = {a} B is a starting point; if allocation fails "
                "raise it carefully (ESP32-S3 RAM is limited).",
                a=info.get("arena", DEFAULT_ARENA),
            )
        )
    s += [
        tr(
            "Sube el sketch y abre el Monitor Serie a {b} baudios.",
            "Upload the sketch and open the Serial Monitor at {b} baud.",
            b=SERIAL_BAUD,
        ),
        tr(
            "Comprueba que «pred=» coincide con «expected=»; después sustituye SAMPLE_NUM / "
            "SAMPLE_CAT por tus lecturas reales (orden en la tabla de entradas).",
            'Check that "pred=" equals "expected="; then replace SAMPLE_NUM / SAMPLE_CAT '
            "with your real readings (order in the inputs table).",
        ),
    ]
    return s


def _readme(name: str, variant: str, steps: list[str], table: list[dict], verif: dict) -> str:
    rows = "\n".join(
        f"  {r['array']}[{r['index']}] = {r['name']}" + (f" [{r['unit']}]" if r.get("unit") else "")
        for r in table
    )
    body = "\n".join(f"{i}. {t}" for i, t in enumerate(steps, 1))
    return (
        f"{name} ({variant})\n{'=' * (len(name) + len(variant) + 3)}\n\n"
        f"{body}\n\n"
        "Inputs / Entradas (RAW values, NAN if missing / valores CRUDOS, NAN si faltan):\n"
        f"{rows}\n\n"
        f"Verification / Verificación: {verif['level']}\n  {verif['detail']}\n\n"
        "Generated by TinyML Studio. No credentials or network settings are included; "
        "add Wi-Fi code yourself and keep secrets out of version control.\n"
    )


# ---------------------------------------------------------------- API pública


def build_arduino_sketch(
    export_dir: str | Path,
    out_dir: str | Path,
    name: str,
    variant: str = "auto",
    *,
    model: str = "int8",
    arena_size: int | None = None,
    units: dict[str, str] | None = None,
    cc: str = "cc",
    cxx: str = "g++",
) -> dict[str, Any]:
    """Crea ``out_dir/<name>/<name>.ino`` (+ archivos) a partir de ``export_dir``.

    ``variant``: 'auto' (raw_c si hay tm_model.c; si no, eloquent, o tflm si la E/S es int8),
    'raw_c', 'eloquent' o 'tflm'. ``model``: 'int8' o 'float' (solo TFLite).
    Devuelve ``{variant, sketch_dir, files, steps, input_table, verification, warnings, ...}``;
    ``files`` son rutas relativas a la carpeta del sketch.
    """
    exp, out = Path(export_dir), Path(out_dir)
    if variant not in VARIANTS:
        raise ValueError(f"variant debe ser uno de {VARIANTS}")
    if model not in ("int8", "float"):
        raise ValueError("model debe ser 'int8' o 'float'")
    name = _safe_name(name)
    kind = _detect(exp)
    meta = _read_json(exp / "model_meta.json")
    if variant == "auto":
        variant = (
            "raw_c" if kind == "raw_c" else ("tflm" if meta.get("io") == "int8" else "eloquent")
        )
    if (variant == "raw_c") != (kind == "raw_c"):
        raise ValueError(
            tr(
                "la variante '{v}' no corresponde a esta exportación ({k})",
                "variant '{v}' does not match this export ({k})",
                v=variant,
                k="C" if kind == "raw_c" else "TFLite",
            )
        )
    sk = out / name
    sk.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = list(meta.get("warnings", [])) if kind == "raw_c" else []

    if variant == "raw_c":
        info = _build_raw_c(exp, sk, name, meta)
        vec = _first_vector(exp, "expected_model")
        verif = _host_check(sk, name, info["task"], vec["expected"], cc, cxx)
    else:
        info = _build_tflite(exp, sk, name, meta, variant, model, arena_size, warnings)
        verif = {
            "level": "not_compiled",
            "detail": tr(
                "no compilado: No hay toolchain de Arduino aquí y la API de {lib} depende de "
                "la versión instalada; compílalo en tu IDE.",
                "not compiled: No Arduino toolchain here and the {lib} API depends on the "
                "installed version; compile it in your IDE.",
                lib="EloquentTinyML/tflm_esp32" if variant == "eloquent" else "TFLite Micro",
            ),
        }

    table = _input_table(info["num"], info["cat"], units)
    steps = _steps(variant, name, info)
    (sk / "README.txt").write_text(_readme(name, variant, steps, table, verif), encoding="utf-8")
    files = [f"{name}.ino", *info["files"], "README.txt"]
    return {
        "variant": variant,
        "name": name,
        "sketch_dir": str(sk),
        "files": files,
        "steps": steps,
        "input_table": table,
        "verification": verif,
        "warnings": warnings,
        **({"ops": info["ops"], "arena_size": info["arena"]} if variant != "raw_c" else {}),
    }
