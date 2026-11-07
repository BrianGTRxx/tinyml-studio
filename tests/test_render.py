"""Gráficas (SVG/PNG/JPG) e informe PDF: dibujo, formatos, idiomas y casos límite."""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PIL")
pytest.importorskip("fpdf")

from tinyml_studio.i18n import set_lang  # noqa: E402
from tinyml_studio.render import (  # noqa: E402
    ReportData,
    Scene,
    available_charts,
    build_charts,
    build_pdf,
    save_charts,
    text_preview,
)
from tinyml_studio.render.charts import nice_ticks  # noqa: E402
from tinyml_studio.render.scene import est_width, shorten  # noqa: E402
from tinyml_studio.tui.session import Session  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_lang():
    yield
    set_lang(None)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("render")
    rng = np.random.default_rng(1)
    n = 450
    base = np.repeat([0.0, 4.0, 8.0], n // 3)
    p = tmp / "d.csv"
    pd.DataFrame(
        {
            "f1": base + rng.normal(0, 1.2, n),
            "f2": rng.normal(size=n) + base / 2,
            "label": np.repeat(["a", "b", "c"], n // 3),
        }
    ).to_csv(p, index=False)
    s = Session(workdir=tmp / "w")
    s.inspect([str(p)])
    s.prepare()
    s.run_train(["decision_tree", "random_forest"])
    return ReportData(
        results=s.train, split=s.split, recipe=s.recipe, report=s.report, dataset_name="d.csv"
    )


def test_nice_ticks_cover_the_range():
    for lo, hi in [(0, 1), (0, 64.7), (-0.3, 1.0), (0, 0.0004), (5, 5)]:
        t = nice_ticks(lo, hi)
        assert t[0] <= lo + 1e-9 and t[-1] >= hi - 1e-9 and len(t) >= 2
        assert t == sorted(t)


def test_scene_svg_is_well_formed_and_escapes_text():
    sc = Scene(200, 100, 'T "q" <b>')
    sc.rect(10, 10, 50, 20, "#ff0000")
    sc.line(0, 0, 100, 100, dash=True)
    sc.poly([(0, 0), (10, 20), (30, 5)])
    sc.circle(50, 50, 4)
    sc.text(100, 50, "a < b & c", 11, anchor="middle")
    sc.text(20, 60, "vertical", rotate=-90)
    root = ET.fromstring(sc.to_svg())  # XML válido pese a <, & y comillas
    assert root.tag.endswith("svg") and "a &lt; b &amp; c" in sc.to_svg()


def test_scene_png_and_jpg_have_real_content_and_accents(tmp_path):
    sc = Scene(300, 120, "x")
    sc.text(10, 40, "Precisión ñandú ¿qué? ≈ ↔ ★", 16)
    sc.rect(10, 60, 100, 30, "#0072B2")
    png = sc.save(tmp_path / "a", "png", 1.0)
    jpg = sc.save(tmp_path / "b", "jpeg", 1.0)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and jpg.suffix == ".jpg"
    assert jpg.read_bytes()[:2] == b"\xff\xd8"
    from PIL import Image

    im = Image.open(png)
    assert im.size == (300, 120)
    colors = {c for _, c in im.getcolors(maxcolors=100000)}
    assert (0, 114, 178) in colors  # el rectángulo azul
    # la fuente incluida tiene glifos de verdad: el texto con acentos deja más tinta que un hueco
    dark = sum(n for n, c in im.getcolors(maxcolors=100000) if sum(c) < 200 and c != (0, 114, 178))
    assert dark > 150
    with pytest.raises(ValueError):
        sc.save(tmp_path / "c", "bmp")


def test_text_helpers():
    assert shorten("abcdefghij", 20, 10).endswith("…") and shorten("ab", 100, 10) == "ab"
    assert est_width("abcd", 10) > est_width("ab", 10)


def test_available_charts_and_build(data):
    ids = [c for c, _ in available_charts(data)]
    assert {"comparison", "confusion", "roc", "pr", "per_class", "distribution"} <= set(ids)
    assert "budget" not in ids  # sin exportación no hay presupuesto
    scenes = build_charts(data)
    assert list(scenes) == ids and all(s.items for s in scenes.values())
    with pytest.raises(ValueError, match="desconocida|unknown"):
        build_charts(data, ["nope"])
    with pytest.raises(ValueError, match="presupuesto|budget"):
        build_charts(data, ["budget"])


def test_confusion_chart_uses_the_real_matrix(data):
    sc = build_charts(data, ["confusion"])["confusion"]
    cm = data.quality()["splits"]["test"]["confusion_matrix"]
    texts = [it[3] for it in sc.items if it[0] == "text"]
    assert f"{max(max(r) for r in cm):,}" in texts  # el mayor recuento aparece escrito
    assert all(str(c) in texts for c in "123")  # columnas numeradas


def test_budget_chart_and_pdf_with_export(data):
    export = {
        "board": "esp32-s3",
        "budget": {
            "flash_data": {
                "used_bytes": 2048,
                "budget_kb": 7168,
                "fraction": 0.0003,
                "verdict_code": "fits",
                "verdict_label": "cabe",
            },
            "ram_working": {
                "used_bytes": 64,
                "budget_kb": 204,
                "fraction": 0.0003,
                "verdict_code": "over",
                "verdict_label": "no cabe",
            },
            "note": "nota de prueba " * 12,
        },
        "verification": {"vectors": 100, "model_match": 1.0, "vs_full_python_pipeline": 1.0},
    }
    d = ReportData(**{**data.__dict__, "export": export})
    sc = build_charts(d, ["budget"])["budget"]
    assert any("no cabe" in str(it[3]) for it in sc.items if it[0] == "text")


def test_save_charts_all_formats_and_pdf(data, tmp_path):
    paths = save_charts(data, tmp_path / "out", ["png", "svg", "jpg"], ["comparison", "per_class"])
    assert sorted(p.name for p in paths) == [
        "comparison.jpg", "comparison.png", "comparison.svg",
        "per_class.jpg", "per_class.png", "per_class.svg",
    ]  # fmt: skip
    pdf = build_pdf(data, tmp_path / "sub" / "r.pdf")
    raw = pdf.read_bytes()
    assert raw[:5] == b"%PDF-" and raw.rstrip().endswith(b"%%EOF") and len(raw) > 20_000
    with pytest.raises(ValueError, match="formato|format"):
        save_charts(data, tmp_path, ["gif"])


def test_pdf_text_is_extractable_and_bilingual(data, tmp_path):
    pytest.importorskip("pypdf")
    from pypdf import PdfReader

    es = " ".join(
        p.extract_text() for p in PdfReader(str(build_pdf(data, tmp_path / "es.pdf"))).pages
    )
    assert "Informe de entrenamiento" in es and "clasificación" in es
    set_lang("en")
    en = " ".join(
        p.extract_text() for p in PdfReader(str(build_pdf(data, tmp_path / "en.pdf"))).pages
    )
    assert "training report" in en and "classification" in en and "Informe de" not in en


def test_text_preview_marks_chosen_model(data):
    t = text_preview(data)
    assert "★" in t and "█" in t and "random_forest" in t


def test_regression_results_only_offer_the_comparison():
    results = {
        "task": "regression",
        "selection": {"metric": "r2", "best": "m"},
        "baseline": {"test": {"r2": 0.0}},
        "models": {
            "m": {"val": {"r2": 0.8}, "test": {"r2": 0.75}, "fit_seconds": 0.1, "complexity": {}}
        },
    }
    d = ReportData(results=results)
    assert [c for c, _ in available_charts(d)] == ["comparison"]
    assert build_charts(d)["comparison"].items


def test_cli_report_from_a_training_folder(data, tmp_path, capsys):
    import json

    from tinyml_studio.cli import main

    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(data.results), encoding="utf-8")
    out = tmp_path / "rep"
    assert (
        main(
            ["--lang", "en", "report", str(run), "--out", str(out), "--formats", "png,svg", "--pdf"]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert (out / "comparison.png").exists() and (out / "confusion.svg").exists()
    assert (out / "informe.pdf").read_bytes()[:5] == b"%PDF-" and "Available charts" in text
    assert main(["report", str(tmp_path / "no"), "--out", str(out)]) == 2
    assert main(["report", str(run), "--out", str(out), "--formats", "gif"]) == 2
