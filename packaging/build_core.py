"""Construye el núcleo congelado: ``python packaging/build_core.py [--variant lite|full]``.

Resultado: ``packaging/dist/tinyml-core/`` (ejecutable ``tinyml-core`` / ``tinyml-core.exe``).
Carpeta de trabajo: ``packaging/build/``. Ambas están en .gitignore.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def folder_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["lite", "full"], default="lite")
    ap.add_argument("--no-clean", action="store_true", help="reutiliza la caché de PyInstaller")
    a = ap.parse_args()
    dist, work = HERE / "dist", HERE / "build"
    if not a.no_clean:
        shutil.rmtree(dist / "tinyml-core", ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    env = dict(os.environ, TINYML_CORE_VARIANT=a.variant)
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm",
        "--distpath", str(dist), "--workpath", str(work), str(HERE / "core.spec"),
    ]  # fmt: skip
    r = subprocess.run(cmd, env=env, cwd=HERE.parent)
    out = dist / "tinyml-core"
    exe = out / ("tinyml-core.exe" if os.name == "nt" else "tinyml-core")
    if r.returncode or not exe.is_file():
        print("la compilación falló", file=sys.stderr)
        return r.returncode or 1
    print(f"OK variante={a.variant}: {exe} ({folder_mb(out):.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
