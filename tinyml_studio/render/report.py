"""Guardar gráficas (SVG / PNG / JPG) y generar el informe PDF."""

from __future__ import annotations

import datetime as _dt
import io
from pathlib import Path
from typing import Any

from tinyml_studio.i18n import tr
from tinyml_studio.render.charts import build_charts
from tinyml_studio.render.data import ReportData
from tinyml_studio.render.scene import FONT_BOLD, FONT_REGULAR, Scene

__all__ = ["ReportData", "build_pdf", "save_charts"]

FORMATS = ("png", "jpg", "svg")


def save_charts(
    d: ReportData,
    out_dir: str | Path,
    formats: list[str] | tuple[str, ...] = ("png",),
    only: list[str] | None = None,
    scale: float = 2.0,
) -> list[Path]:
    """Guarda las gráficas pedidas en ``out_dir`` (``<id>.<formato>``); devuelve las rutas."""
    fmts = [f.lower().lstrip(".") for f in formats]
    bad = [f for f in fmts if f not in (*FORMATS, "jpeg")]
    if bad:
        raise ValueError(tr("formato no soportado: {f}", "unsupported format: {f}", f=bad[0]))
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for cid, scene in build_charts(d, only).items():
        for f in fmts:
            paths.append(scene.save(out / cid, f, scale))
    return paths


# ---------------------------------------------------------------------------------- PDF
def _t(s: Any) -> str:
    return "" if s is None else str(s)


def _fmt(v: Any, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def build_pdf(d: ReportData, path: str | Path, charts: list[str] | None = None) -> Path:
    """Informe PDF (A4) con datos, partición, comparación de modelos, calidad, exportación y gráficas."""
    try:
        from fpdf import FPDF
    except ImportError as e:  # pragma: no cover - depende del entorno
        from tinyml_studio.render.scene import _NEED_PILLOW

        raise RuntimeError(_NEED_PILLOW()) from e

    class Doc(FPDF):
        def footer(self) -> None:
            self.set_y(-12)
            self.set_font("DejaVu", "", 8)
            self.set_text_color(110, 120, 130)
            self.cell(
                0, 6, f"TinyML Studio · {tr('página', 'page')} {self.page_no()}/{{nb}}", align="C"
            )

    pdf = Doc(format="A4")
    pdf.set_margins(16, 16, 16)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font("DejaVu", "", str(FONT_REGULAR))
    pdf.add_font("DejaVu", "B", str(FONT_BOLD))
    pdf.alias_nb_pages()
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def h1(s: str) -> None:
        pdf.set_font("DejaVu", "B", 20)
        pdf.set_text_color(31, 41, 51)
        pdf.multi_cell(W, 9, s, new_x="LMARGIN", new_y="NEXT")

    def h2(s: str) -> None:
        pdf.ln(3)
        pdf.set_font("DejaVu", "B", 13)
        pdf.set_text_color(0, 90, 150)
        pdf.multi_cell(W, 7, s, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(217, 222, 228)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
        pdf.ln(2)

    def p(s: str, size: int = 10, muted: bool = False) -> None:
        pdf.set_font("DejaVu", "", size)
        pdf.set_text_color(*((97, 110, 124) if muted else (31, 41, 51)))
        pdf.multi_cell(W, 5.2, s, new_x="LMARGIN", new_y="NEXT")

    def table(header: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
        pdf.set_font("DejaVu", "", 9)
        pdf.set_text_color(31, 41, 51)
        with pdf.table(
            col_widths=widths, text_align="LEFT", line_height=5.5, first_row_as_headings=True,
            borders_layout="SINGLE_TOP_LINE",
        ) as t:  # fmt: skip
            r = t.row()
            for c in header:
                r.cell(c, style=_head_style())
            for row in rows:
                r = t.row()
                for c in row:
                    r.cell(_t(c))
        pdf.ln(2)

    def _head_style():
        from fpdf.fonts import FontFace

        return FontFace(emphasis="BOLD", color=(31, 41, 51), fill_color=(238, 241, 244))

    res = d.results
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    h1(tr("Informe de entrenamiento TinyML", "TinyML training report"))
    p(f"{d.dataset_name or tr('dataset', 'dataset')} · {now}", 10, True)

    # 1. Resumen
    h2(tr("1. Resumen", "1. Summary"))
    best = d.best
    bm = (res.get("models") or {}).get(best or "", {})
    p(
        tr(
            "Tarea: {t} · variable objetivo: {y} · {n} modelos entrenados · elegido: {b} "
            "(criterio: {m} en validación; el test no interviene en la elección).",
            "Task: {t} · target: {y} · {n} models trained · chosen: {b} "
            "(criterion: {m} on validation; the test set takes no part in the choice).",
            t=_task(res.get("task")),
            y=_t(res.get("target")),
            n=len(res.get("models") or {}),
            b=_t(best),
            m=d.metric,
        )  # fmt: skip
    )
    if bm:
        p(
            tr(
                "{b}: {m} = {v} en validación y {t} en test; ajuste {s} s; tamaño {c}.",
                "{b}: {m} = {v} on validation and {t} on test; fit {s} s; size {c}.",
                b=_t(best),
                m=d.metric,
                v=_fmt(bm["val"].get(d.metric)),
                t=_fmt(bm["test"].get(d.metric)),
                s=_fmt(bm.get("fit_seconds"), 2),
                c=_size(bm.get("complexity")),
            )  # fmt: skip
        )
    base = ((res.get("baseline") or {}).get("test") or {}).get(d.metric)
    if isinstance(base, (int, float)):
        p(
            tr(
                "Línea base (clase más frecuente) en test: {v}.",
                "Baseline (most frequent class) on test: {v}.",
                v=_fmt(base),
            )
        )

    # 2. Datos y partición
    h2(tr("2. Datos y partición", "2. Data and split"))
    if d.report:
        p(_t(d.report.get("summary") or d.report.get("summary_es")).split("\n\n")[0][:900], 9, True)
    if d.split:
        sz = d.split.get("sizes") or {}
        strat = (d.split.get("strategy") or {}).get("strategy", "?")
        p(
            tr("Estrategia: {s} · train {a:,} · validación {b:,} · test {c:,}.", "Strategy: {s} · train {a:,} · validation {b:,} · test {c:,}.",
               s=strat, a=sz.get("train", 0), b=sz.get("val", 0), c=sz.get("test", 0))
        )  # fmt: skip
    for w in res.get("warnings") or []:
        p("• " + _t(w), 9, True)

    # 3. Modelos
    h2(tr("3. Comparación de modelos", "3. Model comparison"))
    rows = []
    for n, m in (res.get("models") or {}).items():
        rows.append(
            [
                ("★ " if n == best else "") + n,
                _fmt(m["val"].get(d.metric)),
                _fmt(m["test"].get(d.metric)),
                _fmt(m.get("fit_seconds"), 2),
                _size(m.get("complexity")),
            ]
        )
    table(
        [
            tr("Modelo", "Model"),
            f"{d.metric} val",
            f"{d.metric} test",
            tr("Ajuste (s)", "Fit (s)"),
            tr("Tamaño", "Size"),
        ],
        rows,
        [52, 24, 24, 24, 54],
    )

    # 4. Calidad del elegido
    q = d.quality()
    splits = q.get("splits") or {}
    if splits:
        h2(
            tr(
                "4. Calidad del modelo elegido ({b})",
                "4. Quality of the chosen model ({b})",
                b=_t(best),
            )
        )
        keys = [
            k
            for k in ("accuracy", "balanced_accuracy", "f1_macro", "mAP", "roc_auc_macro", "mcc")
            if any(k in s for s in splits.values())
        ]
        table(
            [tr("Métrica", "Metric"), "train", tr("validación", "validation"), "test"],
            [
                [k, *[_fmt((splits.get(s) or {}).get(k)) for s in ("train", "val", "test")]]
                for k in keys
            ],
            [64, 36, 36, 36],
        )
        for note in (splits.get("test") or {}).get("notes") or []:
            p("• " + _t(note), 9, True)

    # 5. Exportación
    ex = d.export
    if ex:
        h2(tr("5. Exportación a C", "5. Export to C"))
        b = ex.get("budget") or {}
        v = ex.get("verification") or {}
        p(tr("Placa: {b}.", "Board: {b}.", b=_t(ex.get("board"))))
        for key, label in (("flash_data", "Flash"), ("ram_working", "RAM")):
            r = b.get(key) or {}
            if r:
                p(
                    f"{label}: {r.get('used_bytes', 0):,} B / {r.get('budget_kb', 0):,} KB ({r.get('fraction', 0) * 100:.1f} %) — {_t(r.get('verdict_label'))}"
                )
        if v:
            p(tr("Verificación Python↔C en PC con {n:,} vectores: Modelo {m:.2%}, pipeline completo {f:.2%}.", "Python↔C check on PC with {n:,} vectors: Model {m:.2%}, full pipeline {f:.2%}.",
                 n=int(v.get("vectors", 0)), m=float(v.get("model_match", 0)), f=float(v.get("vs_full_python_pipeline", 0))))  # fmt: skip
        p(_t(b.get("note")), 9, True)
        p(
            tr(
                "Esto se verificó en PC. Falta probar en la placa real con test_vectors.csv.",
                "This was verified on a PC. It still has to be tested on the real board with test_vectors.csv.",
            ),
            9,
            True,
        )

    # 6. Gráficas
    scenes = build_charts(d, charts)
    if scenes:
        h2(tr("6. Gráficas", "6. Charts"))
        for sc in scenes.values():
            _image(pdf, sc, W)
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    return out


def _image(pdf, scene: Scene, W: float) -> None:
    h = W * scene.height / scene.width
    if pdf.get_y() + h + 4 > pdf.h - pdf.b_margin:
        pdf.add_page()
    pdf.image(io.BytesIO(scene.to_png_bytes(2.0)), x=pdf.l_margin, w=W)
    pdf.ln(3)


def _task(t: Any) -> str:
    return {
        "classification": tr("clasificación", "classification"),
        "regression": tr("regresión", "regression"),
    }.get(str(t), _t(t))


def _size(c: Any) -> str:
    if isinstance(c, dict):
        return " ".join(f"{k}={v}" for k, v in c.items())
    return _t(c)
