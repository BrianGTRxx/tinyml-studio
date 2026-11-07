"""Vista previa en texto (para la terminal): barras con bloques y tablas simples."""

from __future__ import annotations

from tinyml_studio.i18n import tr
from tinyml_studio.render.data import ReportData

BLOCKS = " ▏▎▍▌▋▊▉█"


def bar(value: float, width: int = 28, vmax: float = 1.0) -> str:
    """Barra horizontal con fracciones de bloque (resolución de 1/8 de carácter)."""
    v = max(0.0, min(float(value) / (vmax or 1.0), 1.0)) * width
    full, frac = int(v), int((v - int(v)) * 8)
    return "█" * full + (BLOCKS[frac] if full < width else "")


def text_preview(d: ReportData, model: str | None = None) -> str:
    """Resumen legible sin imágenes: modelos, calidad por partición y F1 por clase."""
    out: list[str] = []
    models = d.results.get("models") or {}
    metric = d.metric
    out.append(
        tr(
            "Modelos ({m}; ★ = elegido, el test no elige)",
            "Models ({m}; ★ = chosen, test never chooses)",
            m=metric,
        )
    )
    width = max((len(n) for n in models), default=8) + 2
    for n, m in models.items():
        v, t = m["val"].get(metric), m["test"].get(metric)
        if isinstance(v, (int, float)) and isinstance(t, (int, float)):
            out.append(
                f"  {('★ ' if n == d.best else '  ') + n:<{width + 2}} val {v:6.3f} {bar(v, 18)}"
            )
            out.append(f"  {'':<{width + 2}} test {t:5.3f} {bar(t, 18)}")
    name = model or d.best
    sp = (d.quality(name).get("splits") or {}).get("test") or {}
    pc = sp.get("per_class") or {}
    if pc:
        out.append("")
        out.append(tr("F1 por clase — {m} (test)", "F1 per class — {m} (test)", m=name))
        w = min(max(len(c) for c in pc) + 1, 28)
        for c, m in sorted(pc.items(), key=lambda kv: kv[1].get("f1", 0)):
            out.append(
                f"  {c[:w]:<{w}} {m.get('f1', 0):5.3f} {bar(m.get('f1', 0), 24)}  n={m.get('support', 0):,}"
            )
    if d.export and d.export.get("budget"):
        b = d.export["budget"]
        out.append("")
        out.append(
            tr("Presupuesto de la placa {b}", "Board budget {b}", b=d.export.get("board", ""))
        )
        for key, label in (("flash_data", "Flash"), ("ram_working", "RAM")):
            r = b.get(key) or {}
            if r:
                out.append(
                    f"  {label:<6} {bar(r.get('fraction', 0), 24)} {r.get('fraction', 0) * 100:5.1f} %  {r.get('verdict_label', '')}"
                )
    return "\n".join(out)
