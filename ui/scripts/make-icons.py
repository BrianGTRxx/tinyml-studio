"""Genera ui/assets/icon.ico (16..256) a partir de ui/assets/icon.png. Requiere Pillow.

Uso: python ui/scripts/make-icons.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> int:
    src = Image.open(ASSETS / "icon.png").convert("RGBA")
    if min(src.size) < 256:
        raise SystemExit(f"icon.png debe medir al menos 256x256 (mide {src.size})")
    if src.size != (256, 256):
        src = src.resize((256, 256), Image.LANCZOS)
        src.save(ASSETS / "icon.png")
    src.save(ASSETS / "icon.ico", format="ICO", sizes=SIZES)
    print(f"escrito {ASSETS / 'icon.ico'} ({', '.join(f'{w}' for w, _ in SIZES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
