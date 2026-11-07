"""Prepara ``ui/build-resources`` para electron-builder.

Uso: python packaging/prepare_resources.py --variant raw|offline [--platform ...] [--out DIR]

- ``env/``: manifest.json, locks/, doctor_standalone.py + doctor.py y la rueda de tinyml-studio.
- ``uv/``: binario de uv de la plataforma, bajado de PyPI y VERIFICADO con el sha256 fijado en el
  manifiesto (si no coincide, se aborta).
- ``core/`` (solo variante ``offline``): núcleo congelado de packaging/dist/tinyml-core.
La variante ``raw`` (la normal) no lleva paquetes de Python: la app los instala con uv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "packaging" / "env"


def host_platform() -> str:
    return "windows" if sys.platform == "win32" else "linux"


def fetch_uv(platform: str, out: Path, manifest: dict) -> Path:
    spec = manifest["uv_wheels"][platform]
    version = manifest["uv"]
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/uv/{version}/json", timeout=60))
    url = next((f["url"] for f in meta["urls"] if f["filename"] == spec["filename"]), None)
    if url is None:
        raise SystemExit(f"uv {version}: no existe {spec['filename']} en PyPI")
    data = urllib.request.urlopen(url, timeout=300).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec["sha256"]:
        raise SystemExit(f"sha256 de {spec['filename']} no coincide: {digest} != {spec['sha256']}")
    out.mkdir(parents=True, exist_ok=True)
    exe = "uv.exe" if platform == "windows" else "uv"
    tmp = out / "_uv.whl"
    tmp.write_bytes(data)
    try:
        with zipfile.ZipFile(tmp) as z:
            member = next(n for n in z.namelist() if n.endswith(f".data/scripts/{exe}"))
            (out / exe).write_bytes(z.read(member))
    finally:
        tmp.unlink(missing_ok=True)
    (out / exe).chmod(0o755)
    return out / exe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["raw", "offline"], default="raw")
    ap.add_argument("--platform", choices=["linux", "windows"], default=host_platform())
    ap.add_argument("--out", default=str(ROOT / "ui" / "build-resources"))
    a = ap.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    env_out = out / "env"
    env_out.mkdir(parents=True)
    manifest = json.loads((ENV / "manifest.json").read_text(encoding="utf-8"))
    shutil.copy2(ENV / "manifest.json", env_out / "manifest.json")
    shutil.copytree(ENV / "locks", env_out / "locks")
    shutil.copy2(ENV / "doctor_standalone.py", env_out / "doctor_standalone.py")
    shutil.copy2(ROOT / "tinyml_studio" / "doctor.py", env_out / "doctor.py")
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "packaging" / "build_wheel.py"),
            "--out",
            str(env_out / "wheels"),
        ]
    )
    if r.returncode != 0:
        return r.returncode
    uv = fetch_uv(a.platform, out / "uv", manifest)
    print(f"uv {manifest['uv']} ({a.platform}): {uv.stat().st_size // 1024} KB, sha256 verificado")
    if a.variant == "offline":
        core = ROOT / "packaging" / "dist" / "tinyml-core"
        if not core.is_dir():
            raise SystemExit(
                "falta packaging/dist/tinyml-core: ejecuta packaging/build_core.py primero"
            )
        shutil.copytree(core, out / "core", symlinks=True)
    print(f"recursos listos en {out} (variante {a.variant}, {a.platform})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
