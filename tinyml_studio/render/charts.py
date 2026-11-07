"""Gráficas como escenas: comparación de modelos, matriz de confusión, ROC/PR, por clase, etc."""

from __future__ import annotations

import math
from collections.abc import Callable

from tinyml_studio.i18n import tr
from tinyml_studio.render.data import ReportData
from tinyml_studio.render.scene import (
    BG,
    GRID,
    INK,
    MUTED,
    PALETTE,
    Scene,
    est_width,
    shorten,
)

SERIES = [*PALETTE[:7], "#882255", "#44AA99", "#AA4499", "#999999", "#117733", "#6699CC"]
VERDICT = {"fits": "#009E73", "tight": "#E69F00"}
OVER = "#D55E00"


# ------------------------------------------------------------------------------ utilidades
def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Marcas "redondas" que cubren [lo, hi] (la última es >= hi)."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.floor(lo / step + 1e-9)
    last = math.ceil(hi / step - 1e-9)
    return [round(k * step, 10) for k in range(first, last + 1)]


def fmt_tick(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:.0f}"


def wrap(text: str, max_px: float, size: float) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if est_width(trial, size) <= max_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _scene(title: str, subtitle: str = "", w: float = 880, h: float = 520) -> Scene:
    s = Scene(w, h, title)
    s.text(24, 30, title, 16, INK, bold=True)
    if subtitle:
        s.text(24, 50, subtitle, 11, MUTED)
    return s


def _axes(s: Scene, box, xlab: str = "", ylab: str = "", yticks=None, xticks=None, fmt=fmt_tick):
    """Marco con rejilla horizontal (y) y, si se piden, marcas en x. Devuelve funciones de escala."""
    left, top, right, bottom = box
    s.line(left, bottom, right, bottom, MUTED, 1.2)
    s.line(left, top, left, bottom, MUTED, 1.2)
    sy = sx = None
    if yticks:
        lo, hi = yticks[0], yticks[-1]

        def sy(v: float) -> float:
            return bottom - (v - lo) / (hi - lo) * (bottom - top)

        for t in yticks:
            s.line(left, sy(t), right, sy(t), GRID, 1)
            s.text(left - 8, sy(t) + 4, fmt(t), 10, MUTED, "end")
    if xticks:
        lo, hi = xticks[0], xticks[-1]

        def sx(v: float) -> float:
            return left + (v - lo) / (hi - lo) * (right - left)

        for t in xticks:
            s.line(sx(t), bottom, sx(t), bottom + 4, MUTED, 1)
            s.text(sx(t), bottom + 17, fmt(t), 10, MUTED, "middle")
    if xlab:
        s.text((left + right) / 2, bottom + 40, xlab, 11, INK, "middle")
    if ylab:
        s.text(left - 52, (top + bottom) / 2, ylab, 11, INK, "middle", rotate=-90)
    return sx, sy


def _legend(
    s: Scene,
    x: float,
    y: float,
    entries: list[tuple[str, str]],
    size: float = 11,
    horizontal: bool = False,
) -> None:
    pos = x
    for i, (label, color) in enumerate(entries):
        yy = y if horizontal else y + i * (size + 8)
        xx = pos if horizontal else x
        s.rect(xx, yy - size + 2, size, size, color)
        s.text(xx + size + 6, yy + 1, label, size, INK)
        pos += size + 6 + est_width(label, size) + 22


def _splits_label(split: str) -> str:
    return {"train": "train", "val": tr("validación", "validation"), "test": "test"}.get(
        split, split
    )


# ------------------------------------------------------------------------------ gráficas
def model_comparison(d: ReportData) -> Scene:
    models = d.results.get("models") or {}
    names = list(models)
    metric = d.metric
    vals = {n: (models[n]["val"].get(metric), models[n]["test"].get(metric)) for n in names}
    flat = [v for pair in vals.values() for v in pair if isinstance(v, (int, float))]
    base = ((d.results.get("baseline") or {}).get("test") or {}).get(metric)
    hi = 1.0 if all(v <= 1.0 for v in flat) else max(flat) * 1.08
    lo = min(0.0, min(flat) if flat else 0.0)
    yt = nice_ticks(lo, hi, 5)
    s = _scene(
        tr("Comparación de modelos", "Model comparison"),
        tr(
            "{m} en validación y test · el test no elige al ganador",
            "{m} on validation and test · the test set never picks the winner",
            m=metric,
        ),
    )
    box = (86, 78, 860, 440)
    _, sy = _axes(s, box, ylab=metric, yticks=yt)
    band = (box[2] - box[0]) / max(len(names), 1)
    bw = min(band * 0.32, 46)
    for i, n in enumerate(names):
        cx = box[0] + band * (i + 0.5)
        for j, (v, color) in enumerate(zip(vals[n], (PALETTE[0], PALETTE[1]), strict=True)):
            if not isinstance(v, (int, float)):
                continue
            x = cx - bw - 2 + j * (bw + 4)
            y0, y1 = sy(0), sy(v)
            s.rect(x, min(y0, y1), bw, abs(y0 - y1), color)
            s.text(x + bw / 2, min(y0, y1) - 5, f"{v:.3f}", 10, INK, "middle")
        label = ("★ " if n == d.best else "") + n
        s.text(cx, box[3] + 18, shorten(label, band - 8, 10), 10, INK, "middle", bold=n == d.best)
    if isinstance(base, (int, float)):
        s.line(box[0], sy(base), box[2], sy(base), OVER, 1.6, dash=True)
        s.text(
            box[2],
            sy(base) - 6,
            tr("línea base (test) {v:.3f}", "baseline (test) {v:.3f}", v=base),
            10,
            OVER,
            "end",
        )
    _legend(
        s,
        86,
        482,
        [(_splits_label("val"), PALETTE[0]), (_splits_label("test"), PALETTE[1])],
        horizontal=True,
    )
    s.text(860, 482, tr("★ = modelo elegido", "★ = chosen model"), 10, MUTED, "end")
    return s


def confusion(d: ReportData, model: str | None = None, split: str = "test") -> Scene:
    name = model or d.best
    sp = (d.quality(name).get("splits") or {}).get(split) or {}
    cm = sp.get("confusion_matrix")
    classes = sp.get("classes") or d.results.get("classes") or []
    if not cm:
        raise ValueError(tr("no hay matriz de confusión", "no confusion matrix"))
    n = len(cm)
    label_px = min(max(est_width(str(c), 10) for c in classes) + 8, 190)
    cell = max(min(54.0, 460.0 / n), 16.0)
    left, top = 24 + label_px + 24, 96
    grid = n * cell
    w = max(left + grid + 150, 700)
    h = top + grid + 70
    s = _scene(
        tr("Matriz de confusión — {m}", "Confusion matrix — {m}", m=name),
        tr(
            "{sp} (n={n:,}) · filas = clase real · columnas = clase predicha (mismo orden)",
            "{sp} (n={n:,}) · rows = true class · columns = predicted class (same order)",
            sp=_splits_label(split),
            n=int(sp.get("n", 0)),
        ),
        w,
        h,
    )
    rows = [sum(r) or 1 for r in cm]
    for i in range(n):
        s.text(
            left - 8,
            top + (i + 0.62) * cell,
            shorten(str(classes[i]), label_px, 10),
            10,
            INK,
            "end",
        )
        s.text(left + (i + 0.5) * cell, top - 8, str(i + 1), 10, MUTED, "middle")
        for j in range(n):
            v = cm[i][j] / rows[i]
            color = _ramp(v)
            s.rect(left + j * cell, top + i * cell, cell, cell, color, BG, 1)
            if cell >= 22 and cm[i][j]:
                s.text(
                    left + (j + 0.5) * cell,
                    top + (i + 0.62) * cell,
                    f"{cm[i][j]:,}",
                    min(11, cell * 0.36),
                    "#ffffff" if v > 0.5 else INK,
                    "middle",
                )
    bx = left + grid + 28
    for k in range(60):
        s.rect(bx, top + k * (grid * 0.6 / 60), 14, grid * 0.6 / 60 + 0.5, _ramp(1 - k / 59))
    s.text(bx + 22, top + 10, "100 %", 10, MUTED)
    s.text(bx + 22, top + grid * 0.6, "0 %", 10, MUTED)
    s.text(bx, top + grid * 0.6 + 22, tr("% de cada clase real", "% of each true class"), 10, MUTED)
    return s


def _ramp(v: float) -> str:
    """Blanco -> azul (0..1), secuencial y legible en escala de grises."""
    v = max(0.0, min(1.0, v))
    a, b = (255, 255, 255), (0, 90, 150)
    r, g, bl = (round(a[i] + (b[i] - a[i]) * v) for i in range(3))
    return f"#{r:02x}{g:02x}{bl:02x}"


def _curves(d: ReportData, kind: str, model: str | None) -> Scene:
    name = model or d.best
    cur = (d.quality(name).get("curves") or {}).get(kind) or {}
    if not cur:
        raise ValueError(tr("no hay curvas {k}", "no {k} curves", k=kind.upper()))
    roc = kind == "roc"
    xk, yk, mk = ("fpr", "tpr", "auc") if roc else ("recall", "precision", "ap")
    title = (
        tr("Curvas ROC (una clase contra el resto)", "ROC curves (one class vs the rest)")
        if roc
        else tr("Curvas precisión-recall", "Precision-recall curves")
    )
    s = _scene(title + f" — {name}", tr("conjunto de test", "test set"), 900, 540)
    box = (86, 78, 640, 460)
    t = nice_ticks(0, 1, 5)
    sx, sy = _axes(
        s, box,
        xlab=tr("tasa de falsos positivos", "false positive rate") if roc else tr("exhaustividad (recall)", "recall"),
        ylab=tr("tasa de verdaderos positivos", "true positive rate") if roc else tr("precisión", "precision"),
        yticks=t, xticks=t,
    )  # fmt: skip
    if roc:
        s.line(sx(0), sy(0), sx(1), sy(1), MUTED, 1.2, dash=True)
    entries = []
    for i, (cls, c) in enumerate(cur.items()):
        xs, ys = c[xk], c[yk]
        step = max(len(xs) // 220, 1)
        pts = [(sx(x), sy(y)) for x, y in zip(xs[::step], ys[::step], strict=False)]
        if len(xs) % step:
            pts.append((sx(xs[-1]), sy(ys[-1])))
        col = SERIES[i % len(SERIES)]
        s.poly(pts, col, 2.2)
        entries.append((f"{shorten(str(cls), 150, 10)}  {c.get(mk, float('nan')):.3f}", col))
    _legend(s, 668, 96, entries[:22], 10)
    s.text(668, 80, f"{mk.upper()} · " + tr("por clase", "per class"), 10, MUTED)
    return s


def roc_curves(d: ReportData, model: str | None = None) -> Scene:
    return _curves(d, "roc", model)


def pr_curves(d: ReportData, model: str | None = None) -> Scene:
    return _curves(d, "pr", model)


def per_class(d: ReportData, model: str | None = None, split: str = "test") -> Scene:
    name = model or d.best
    sp = (d.quality(name).get("splits") or {}).get(split) or {}
    pc = sp.get("per_class") or {}
    if not pc:
        raise ValueError(tr("no hay métricas por clase", "no per-class metrics"))
    rows = sorted(pc.items(), key=lambda kv: kv[1].get("f1", 0))[:16]
    bar_h, gap = 9, 14
    h = 120 + len(rows) * (3 * bar_h + gap) + 40
    s = _scene(
        tr("Métricas por clase — {m}", "Per-class metrics — {m}", m=name),
        tr("{sp} · de peor a mejor F1 (máximo 16 clases)", "{sp} · worst to best F1 (max 16 classes)", sp=_splits_label(split)),
        880,
        h,
    )  # fmt: skip
    left, right, top = 250, 840, 88
    t = nice_ticks(0, 1, 5)
    for v in t:
        x = left + v * (right - left)
        s.line(x, top - 6, x, h - 54, GRID, 1)
        s.text(x, h - 38, fmt_tick(v), 10, MUTED, "middle")
    cols = (PALETTE[0], PALETTE[2], PALETTE[1])
    for i, (cls, m) in enumerate(rows):
        y = top + i * (3 * bar_h + gap)
        s.text(
            left - 10,
            y + 17,
            shorten(f"{cls} (n={m.get('support', 0):,})", 232, 10),
            10,
            INK,
            "end",
        )
        for j, key in enumerate(("precision", "recall", "f1")):
            v = float(m.get(key, 0.0))
            s.rect(left, y + j * bar_h, max(v * (right - left), 1), bar_h - 1, cols[j])
            if key == "f1":
                s.text(left + v * (right - left) + 6, y + j * bar_h + 8, f"{v:.3f}", 9, INK)
    _legend(
        s,
        left,
        h - 14,
        [(tr("precisión", "precision"), cols[0]), ("recall", cols[1]), ("F1", cols[2])],
        10,
        True,
    )
    return s


def class_distribution(d: ReportData) -> Scene:
    sp = d.split or {}
    cls = sp.get("classes") or {}
    if not cls:
        raise ValueError(tr("no hay distribución de clases", "no class distribution"))
    names = sorted({k for dd in cls.values() for k in dd})
    totals = {k: sum(v.values()) or 1 for k, v in cls.items()}
    order = [x for x in ("train", "val", "test") if x in cls]
    cols = dict(zip(("train", "val", "test"), (PALETTE[0], PALETTE[2], PALETTE[1]), strict=True))
    bar_h, gap = 9, 14
    h = 120 + len(names) * (len(order) * bar_h + gap) + 40
    s = _scene(
        tr("Distribución de clases por partición", "Class distribution per split"),
        tr("% de las filas de cada partición (si difieren mucho, la partición no es estratificada)", "% of each split's rows (if they differ a lot, the split is not stratified)"),
        880,
        h,
    )  # fmt: skip
    left, right, top = 250, 840, 88
    peak = max(cls[k].get(n, 0) / totals[k] for k in order for n in names) * 100
    t = nice_ticks(0, max(peak * 1.12, 1), 5)
    for v in t:
        x = left + v / t[-1] * (right - left)
        s.line(x, top - 6, x, h - 54, GRID, 1)
        s.text(x, h - 38, f"{fmt_tick(v)} %", 10, MUTED, "middle")
    for i, n in enumerate(names):
        y = top + i * (len(order) * bar_h + gap)
        s.text(left - 10, y + 15, shorten(str(n), 232, 10), 10, INK, "end")
        for j, k in enumerate(order):
            pct = cls[k].get(n, 0) / totals[k] * 100
            s.rect(left, y + j * bar_h, max(pct / t[-1] * (right - left), 1), bar_h - 1, cols[k])
    _legend(
        s,
        left,
        h - 14,
        [(f"{_splits_label(k)} (n={totals[k]:,})", cols[k]) for k in order],
        10,
        True,
    )
    return s


def budget(d: ReportData) -> Scene:
    ex = d.export or {}
    b = ex.get("budget") or {}
    if not b:
        raise ValueError(tr("no hay presupuesto de la placa", "no board budget"))
    s = _scene(
        tr("Presupuesto de la placa — {b}", "Board budget — {b}", b=ex.get("board", "?")),
        tr("uso del modelo frente al presupuesto utilizable (heurística nuestra)", "model use against the usable budget (our heuristic)"),
        880,
        300,
    )  # fmt: skip
    left, right = 220, 820
    rows = (
        ("flash_data", tr("Flash (tablas del modelo)", "Flash (model tables)"), 1024.0),
        ("ram_working", tr("RAM de trabajo", "Working RAM"), 1024.0),
    )
    for i, (key, label, unit) in enumerate(rows):
        r = b.get(key) or {}
        if not r:
            continue
        y = 96 + i * 70
        frac = float(r.get("fraction", 0.0))
        col = VERDICT.get(str(r.get("verdict_code")), OVER)
        s.text(left - 12, y + 18, label, 11, INK, "end")
        s.rect(left, y, right - left, 26, "#eef1f4")
        s.rect(left, y, max(min(frac, 1.0) * (right - left), 2), 26, col)
        used, cap = float(r.get("used_bytes", 0)), float(r.get("budget_kb", 0))
        s.text(
            left,
            y + 46,
            f"{used / unit:,.1f} KB / {cap:,.0f} KB  ({frac * 100:.1f} %)  —  {r.get('verdict_label', r.get('verdict', ''))}",
            10,
            MUTED,
        )
    for k, line in enumerate(wrap(str(b.get("note", "")), 760, 10)[:3]):
        s.text(24, 248 + k * 14, line, 10, MUTED)
    return s


# ------------------------------------------------------------------------------ catálogo
CHARTS: dict[str, tuple[str, str, Callable[[ReportData], Scene], Callable[[ReportData], bool]]] = {
    "comparison": ("Comparación de modelos", "Model comparison", model_comparison, lambda d: bool(d.results.get("models"))),
    "confusion": ("Matriz de confusión (test)", "Confusion matrix (test)", confusion, lambda d: d.task == "classification" and bool(d.quality().get("splits"))),
    "roc": ("Curvas ROC", "ROC curves", roc_curves, lambda d: bool((d.quality().get("curves") or {}).get("roc"))),
    "pr": ("Curvas precisión-recall", "Precision-recall curves", pr_curves, lambda d: bool((d.quality().get("curves") or {}).get("pr"))),
    "per_class": ("Métricas por clase", "Per-class metrics", per_class, lambda d: d.task == "classification" and bool(d.quality().get("splits"))),
    "distribution": ("Distribución de clases", "Class distribution", class_distribution, lambda d: bool((d.split or {}).get("classes"))),
    "budget": ("Presupuesto de la placa", "Board budget", budget, lambda d: bool((d.export or {}).get("budget"))),
}  # fmt: skip


def available_charts(d: ReportData) -> list[tuple[str, str]]:
    """[(id, etiqueta)] de las gráficas que se pueden dibujar con los datos actuales."""
    out = []
    for cid, (es, en, _fn, ok) in CHARTS.items():
        try:
            if ok(d):
                out.append((cid, tr(es, en)))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def build_charts(d: ReportData, only: list[str] | None = None) -> dict[str, Scene]:
    """Dibuja las gráficas pedidas (o todas las disponibles); una que falle no tumba al resto."""
    wanted = only or [cid for cid, _ in available_charts(d)]
    out: dict[str, Scene] = {}
    for cid in wanted:
        if cid not in CHARTS:
            raise ValueError(tr("gráfica desconocida: {c}", "unknown chart: {c}", c=cid))
        out[cid] = CHARTS[cid][2](d)
    return out
