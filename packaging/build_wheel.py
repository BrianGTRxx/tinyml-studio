"""Construye la rueda de tinyml-studio con sus datos (placas, manifiesto, locks) dentro.

Uso: python packaging/build_wheel.py [--out packaging/dist/wheels]

Copia temporalmente ``data/`` y ``packaging/env`` a ``tinyml_studio/_data`` (así ``resources.py``
los encuentra instalado fuera del repo), construye con ``uv build --wheel`` (o
``python -m build``) y limpia al terminar.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "packaging" / "dist" / "wheels"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("tinyml_studio-*.whl"):
        old.unlink()
    stage = ROOT / "tinyml_studio" / "_data"
    shutil.rmtree(stage, ignore_errors=True)
    try:
        shutil.copytree(ROOT / "data", stage / "data")
        shutil.copytree(
            ROOT / "packaging" / "env", stage / "env", ignore=shutil.ignore_patterns("*.in", "*.sh")
        )
        uv = shutil.which("uv")
        cmd = (
            [uv, "build", "--wheel", "--out-dir", str(out), str(ROOT)]
            if uv
            else [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(ROOT)]
        )
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            return r.returncode
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(ROOT / "build", ignore_errors=True)
        shutil.rmtree(ROOT / "tinyml_studio.egg-info", ignore_errors=True)
    wheels = sorted(out.glob("tinyml_studio-*.whl"))
    if not wheels:
        print("no se generó la rueda", file=sys.stderr)
        return 1
    print(f"OK {wheels[-1]} ({wheels[-1].stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
