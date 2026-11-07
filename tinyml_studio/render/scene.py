"""Escena de dibujo: primitivas independientes del formato y sus salidas (SVG, PNG/JPG)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

# Paleta categórica apta para daltonismo (Okabe-Ito) + grises de texto y rejilla.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
INK = "#1f2933"
MUTED = "#616e7c"
GRID = "#d9dee4"
BG = "#ffffff"


def est_width(text: str, size: float, bold: bool = False) -> float:
    """Ancho aproximado en px (solo para maquetar: elegir recortes y huecos de leyenda)."""
    return len(text) * size * (0.63 if bold else 0.58)


def shorten(text: str, max_px: float, size: float) -> str:
    if est_width(text, size) <= max_px:
        return text
    n = max(int(max_px / (size * 0.58)) - 1, 1)
    return text[:n] + "…"


@dataclass
class Scene:
    width: float = 880
    height: float = 520
    title: str = ""
    items: list[tuple[Any, ...]] = field(default_factory=list)

    # --- primitivas -------------------------------------------------------------------------
    def rect(self, x, y, w, h, fill=None, stroke=None, sw=1.0) -> None:
        self.items.append(("rect", x, y, w, h, fill, stroke, sw))

    def line(self, x1, y1, x2, y2, color=INK, sw=1.0, dash=False) -> None:
        self.items.append(("line", x1, y1, x2, y2, color, sw, dash))

    def poly(self, pts, color=INK, sw=2.0) -> None:
        self.items.append(("poly", [(float(a), float(b)) for a, b in pts], color, sw))

    def circle(self, cx, cy, r, fill=INK) -> None:
        self.items.append(("circle", cx, cy, r, fill))

    def text(self, x, y, s, size=11, color=INK, anchor="start", bold=False, rotate=0) -> None:
        self.items.append(("text", x, y, str(s), size, color, anchor, bold, rotate))

    # --- salidas ----------------------------------------------------------------------------
    def to_svg(self) -> str:
        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:g}" height="{self.height:g}"'
            f' viewBox="0 0 {self.width:g} {self.height:g}" role="img"'
            f' aria-label="{escape(self.title, {chr(34): "&quot;"})}">',
            f"<title>{escape(self.title)}</title>",
            f'<rect width="100%" height="100%" fill="{BG}"/>',
            '<g font-family="DejaVu Sans, Verdana, Arial, sans-serif">',
        ]
        for it in self.items:
            kind = it[0]
            if kind == "rect":
                _, x, y, w, h, fill, stroke, sw = it
                st = f' stroke="{stroke}" stroke-width="{sw:g}"' if stroke else ""
                out.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}"'
                    f' fill="{fill or "none"}"{st}/>'
                )
            elif kind == "line":
                _, x1, y1, x2, y2, color, sw, dash = it
                d = ' stroke-dasharray="5 4"' if dash else ""
                out.append(
                    f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"'
                    f' stroke="{color}" stroke-width="{sw:g}"{d}/>'
                )
            elif kind == "poly":
                _, pts, color, sw = it
                p = " ".join(f"{a:.2f},{b:.2f}" for a, b in pts)
                out.append(
                    f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw:g}"'
                    ' stroke-linejoin="round" stroke-linecap="round"/>'
                )
            elif kind == "circle":
                _, cx, cy, r, fill = it
                out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:g}" fill="{fill}"/>')
            else:
                _, x, y, s, size, color, anchor, bold, rot = it
                tr_ = f' transform="rotate({rot} {x:.2f} {y:.2f})"' if rot else ""
                wt = ' font-weight="bold"' if bold else ""
                out.append(
                    f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size:g}" fill="{color}"'
                    f' text-anchor="{anchor}"{wt}{tr_}>{escape(s)}</text>'
                )
        out.append("</g></svg>")
        return "\n".join(out)

    def to_image(self, scale: float = 2.0):
        """Dibuja con Pillow (supermuestreo 2x para suavizar) y devuelve una ``PIL.Image`` RGB."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as e:  # pragma: no cover - depende del entorno
            raise RuntimeError(_NEED_PILLOW()) from e
        ss = 2
        k = scale * ss
        im = Image.new("RGB", (round(self.width * k), round(self.height * k)), BG)
        d = ImageDraw.Draw(im)
        fonts: dict[tuple[float, bool], Any] = {}

        def font(size: float, bold: bool):
            key = (round(size * k), bold)
            if key not in fonts:
                fonts[key] = ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), key[0])
            return fonts[key]

        for it in self.items:
            kind = it[0]
            if kind == "rect":
                _, x, y, w, h, fill, stroke, sw = it
                d.rectangle(
                    [x * k, y * k, (x + w) * k - 1, (y + h) * k - 1],
                    fill=fill,
                    outline=stroke,
                    width=max(round(sw * k), 1) if stroke else 0,
                )
            elif kind == "line":
                _, x1, y1, x2, y2, color, sw, dash = it
                w = max(round(sw * k), 1)
                if dash:
                    _dashed(d, (x1 * k, y1 * k), (x2 * k, y2 * k), color, w, 5 * k, 4 * k)
                else:
                    d.line([(x1 * k, y1 * k), (x2 * k, y2 * k)], fill=color, width=w)
            elif kind == "poly":
                _, pts, color, sw = it
                if len(pts) > 1:
                    d.line(
                        [(a * k, b * k) for a, b in pts],
                        fill=color,
                        width=max(round(sw * k), 1),
                        joint="curve",
                    )
            elif kind == "circle":
                _, cx, cy, r, fill = it
                d.ellipse([(cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k], fill=fill)
            else:
                _, x, y, s, size, color, anchor, bold, rot = it
                f = font(size, bold)
                a = {"start": "ls", "middle": "ms", "end": "rs"}[anchor]
                if not rot:
                    d.text((x * k, y * k), s, font=f, fill=color, anchor=a)
                else:
                    box = d.textbbox((0, 0), s, font=f, anchor="ls")
                    tw, th = box[2] - box[0] + 4, box[3] - box[1] + 4
                    tmp = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
                    ImageDraw.Draw(tmp).text(
                        (2 - box[0], 2 - box[1]), s, font=f, fill=color, anchor="ls"
                    )
                    tmp = tmp.rotate(-rot, expand=True)
                    im.paste(
                        tmp, (round(x * k - tmp.width / 2), round(y * k - tmp.height / 2)), tmp
                    )
        if ss != 1:
            im = im.resize((round(self.width * scale), round(self.height * scale)), Image.LANCZOS)
        return im

    def to_png_bytes(self, scale: float = 2.0) -> bytes:
        buf = io.BytesIO()
        self.to_image(scale).save(buf, "PNG", optimize=True)
        return buf.getvalue()

    def save(self, path: str | Path, fmt: str, scale: float = 2.0) -> Path:
        """Guarda como svg / png / jpg (``jpeg`` = ``jpg``)."""
        fmt = fmt.lower().lstrip(".")
        p = Path(path).with_suffix("." + ("jpg" if fmt == "jpeg" else fmt))
        p.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "svg":
            p.write_text(self.to_svg(), encoding="utf-8")
        elif fmt == "png":
            p.write_bytes(self.to_png_bytes(scale))
        elif fmt in ("jpg", "jpeg"):
            self.to_image(scale).save(p, "JPEG", quality=92, optimize=True)
        else:
            raise ValueError(f"formato no soportado: {fmt}")
        return p


def _dashed(d, p1, p2, color, width, on, off) -> None:
    (x1, y1), (x2, y2) = p1, p2
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end = min(pos + on, length)
        d.line(
            [(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
            fill=color,
            width=width,
        )
        pos += on + off


def _NEED_PILLOW() -> str:
    from tinyml_studio.i18n import tr

    return tr(
        "Para guardar PNG/JPG y el PDF hace falta Pillow: Instala el componente «tui» "
        "(o pip install pillow fpdf2).",
        "Saving PNG/JPG and the PDF needs Pillow: Install the 'tui' component "
        "(or pip install pillow fpdf2).",
    )
