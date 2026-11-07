"""Herramientas de PC para verificar y medir C generado (compilar, ejecutar, tamaño, latencia).

Todo lo medido aquí es de x86-64 (``cc``). NO es una medición en un microcontrolador.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .cgen import ARGMAX

PRELUDE = "#include <math.h>\n#include <stddef.h>\n#include <stdint.h>\n"
CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-ffp-contract=off"]


def _inc(dirs: tuple[str, ...]) -> list[str]:
    return [f"-I{d}" for d in dirs]


def full_source(code: str) -> str:
    """Código del modelo (como lo devuelve un emisor) más lo mínimo para compilarlo solo."""
    argmax = ARGMAX + "\n\n" if "tm_argmax(" in code else ""
    return PRELUDE + argmax + code + "\n"


_HARNESS = r"""
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "model.c"
static int cmp_d(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
int main(int argc, char **argv) {
  if (argc < 5) return 2;
  long n = atol(argv[2]); int F = atoi(argv[3]); int reps = atoi(argv[4]);
  FILE *fp = fopen(argv[1], "rb"); if (!fp) return 2;
  float *X = (float *)malloc(sizeof(float) * (size_t)n * (size_t)F);
  if (fread(X, sizeof(float), (size_t)n * (size_t)F, fp) != (size_t)n * (size_t)F) return 2;
  fclose(fp);
#if TF_CLS
  int *out = (int *)malloc(sizeof(int) * (size_t)n);
#else
  float *out = (float *)malloc(sizeof(float) * (size_t)n);
#endif
  for (long i = 0; i < n; i++) out[i] = tm_model(&X[i * F]);
  fp = fopen(argv[5], "wb"); fwrite(out, sizeof(*out), (size_t)n, fp); fclose(fp);
  if (reps <= 0) { printf("NS 0\n"); return 0; }
  double *t = (double *)malloc(sizeof(double) * (size_t)reps);
  volatile double sink = 0;
  for (int r = 0; r < reps; r++) {
    struct timespec a, b; double s = 0;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long i = 0; i < n; i++) s += (double)tm_model(&X[i * F]);
    clock_gettime(CLOCK_MONOTONIC, &b);
    sink += s;
    t[r] = ((b.tv_sec - a.tv_sec) * 1e9 + (double)(b.tv_nsec - a.tv_nsec)) / (double)n;
  }
  qsort(t, (size_t)reps, sizeof(double), cmp_d);
  (void)sink;
  printf("NS %.3f\n", t[reps / 2]);
  return 0;
}
"""


def run_host(
    code: str,
    X: np.ndarray,
    classification: bool,
    *,
    reps: int = 15,
    opt: str = "-O2",
    cc: str = "cc",
    include_dirs: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Compila ``code`` con un arnés y lo ejecuta sobre X: salidas y ns por inferencia."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "model.c").write_text(full_source(code), encoding="utf-8")
        (d / "main.c").write_text(
            f"#define TF_CLS {1 if classification else 0}\n" + _HARNESS, encoding="utf-8"
        )
        X.tofile(d / "x.bin")
        exe = d / ("h.exe" if sys.platform == "win32" else "h")  # GCC en Windows añade .exe
        c = subprocess.run(
            [cc, *CFLAGS, *_inc(include_dirs), opt, "main.c", "-lm", "-o", exe.name],
            cwd=d,
            capture_output=True,
            text=True,
        )
        warns = [ln for ln in c.stderr.splitlines() if "warning" in ln]
        if c.returncode != 0:
            return {"ok": False, "error": c.stderr[-1500:]}
        r = subprocess.run(
            [str(exe), "x.bin", str(len(X)), str(X.shape[1]), str(reps), "y.bin"],
            cwd=d,
            capture_output=True,
            text=True,
            timeout=600,
        )
        m = re.search(r"NS (\S+)", r.stdout)
        if r.returncode != 0 or not m:
            return {"ok": False, "error": (r.stderr or r.stdout)[-500:]}
        y = np.fromfile(d / "y.bin", dtype=np.int32 if classification else np.float32)
        return {"ok": True, "pred": y, "ns_per_inference": float(m[1]), "warnings": warns}


def compiled_size(
    code: str, opt: str = "-Os", cc: str = "cc", include_dirs: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Tamaño del objeto de ``cc -c`` de la función del modelo (x86-64): .text y .rodata aparte."""
    entry = "int tf_entry(const float *f) { return (int)tm_model(f); }\n"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "m.c").write_text(full_source(code) + entry, encoding="utf-8")
        c = subprocess.run(
            [cc, *CFLAGS, *_inc(include_dirs), opt, "-c", "m.c", "-o", "m.o"],
            cwd=d,
            capture_output=True,
            text=True,
        )
        if c.returncode != 0 or shutil.which("size") is None:
            return {"ok": False, "error": c.stderr[-800:]}
        out = subprocess.run(["size", "-A", "m.o"], cwd=d, capture_output=True, text=True).stdout
        sec: dict[str, int] = {}
        for ln in out.splitlines():
            p = ln.split()
            if len(p) >= 2 and p[0].startswith("."):
                sec[p[0]] = int(p[1])
        text = sum(v for k, v in sec.items() if k.startswith(".text"))
        ro = sum(v for k, v in sec.items() if k.startswith(".rodata"))
        data = sum(v for k, v in sec.items() if k.startswith(".data"))
        return {"ok": True, "text": text, "rodata": ro, "data": data, "flag": opt}
