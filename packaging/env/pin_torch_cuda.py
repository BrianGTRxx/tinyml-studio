"""Fija en ``locks/torch-windows.txt`` la rueda de PyTorch con CUDA (Windows).

En Windows, PyPI solo publica PyTorch para CPU (``2.x.y+cpu``): nunca usaría la GPU
NVIDIA. Este script sustituye la entrada ``torch==X`` del lock por la URL de la rueda
``X+cuNNN`` de download.pytorch.org con su hash sha256 (tomado del índice de PyTorch).
El resto del lock (dependencias de PyPI con hashes) no cambia.
Uso (lo llama regen-locks.sh): python pin_torch_cuda.py [--cuda cu130] [--lock RUTA]
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wheel_hash(cuda: str, version: str) -> tuple[str, str]:
    """(URL de la rueda cp312/win_amd64, sha256) según el índice de PyTorch."""
    page = urllib.request.urlopen(f"https://download.pytorch.org/whl/{cuda}/torch/", timeout=60)
    html = page.read().decode("utf-8")
    name = f"torch-{version}%2B{cuda}-cp312-cp312-win_amd64.whl"
    m = re.search(rf'href="[^"]*{re.escape(name)}#sha256=([0-9a-f]{{64}})"', html)
    if not m:
        raise SystemExit(f"el índice {cuda} de PyTorch no tiene {urllib.parse.unquote(name)}")
    return f"https://download.pytorch.org/whl/{cuda}/{name}", m.group(1)


def pin(lock: Path, cuda: str) -> str:
    text = lock.read_text(encoding="utf-8")
    m = re.search(r"^torch(?:==(\S+)| @ \S+) \\\n((?:    --hash=\S+(?: \\)?\n)+)", text, re.M)
    if not m:
        raise SystemExit("no se encontró la entrada de torch en el lock")
    version = m.group(1)
    if version is None:  # ya estaba fijado: leer la versión de la URL
        version = re.search(r"torch-([0-9.]+)%2B", m.group(0)).group(1)
    url, digest = wheel_hash(cuda, version)
    entry = f"torch @ {url} \\\n    --hash=sha256:{digest}\n"
    lock.write_text(text[: m.start()] + entry + text[m.end() :], encoding="utf-8", newline="\n")
    return f"{version}+{cuda} sha256:{digest[:12]}…"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default="cu130")
    ap.add_argument("--lock", default=str(HERE / "locks" / "torch-windows.txt"))
    a = ap.parse_args()
    print("torch-windows fijado a", pin(Path(a.lock), a.cuda), file=sys.stderr)
