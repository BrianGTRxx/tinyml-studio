"""TUI: lógica de sesión y recorrido completo con el piloto sin pantalla de Textual."""

import asyncio
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable, Static, TabbedContent  # noqa: E402

from tinyml_studio.tui.app import TinyMLApp  # noqa: E402
from tinyml_studio.tui.session import Session  # noqa: E402

RNG = np.random.default_rng(3)


@pytest.fixture
def csv(tmp_path):
    n = 450
    base = np.repeat([0.0, 5.0, 10.0], n // 3)
    p = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "f1": base + RNG.normal(0, 1, n),
            "f2": RNG.normal(size=n) + base / 2,
            "label": np.repeat(["a", "b", "c"], n // 3),
        }
    ).to_csv(p, index=False)
    return p


def test_session_roles_split_train(csv, tmp_path):
    s = Session(workdir=tmp_path / "w")
    s.inspect([str(csv)])
    assert s.counts()["inputs"] == 2 and s.recipe["target"] == "label"
    s.set_role("f2", "drop")
    assert s.counts()["inputs"] == 1 and "f2" in s.recipe["drop"]
    s.set_role("f2", "target")  # un único objetivo coherente
    assert s.recipe["target"] == "f2" and s.recipe["roles"]["label"] != "target"
    s.set_role("f2", "feature")
    s.set_role("label", "target")
    sp = s.prepare()
    assert sp["sizes"]["train"] > sp["sizes"]["test"] > 0
    res = s.run_train(["decision_tree"])
    assert res["selection"]["best"] == "decision_tree"
    assert s.train_rows()[0][0].startswith("decision_tree")
    out = s.save_recipe(str(tmp_path / "r" / "recipe.json"))
    assert out.exists()
    with pytest.raises(FileNotFoundError):
        s.inspect([str(tmp_path / "nope.csv")])


def test_tui_full_flow(csv, tmp_path):
    async def go():
        app = TinyMLApp([str(csv)], workdir=tmp_path / "w")
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(100):
                await pilot.pause(0.2)
                if app.session.report:
                    break
            assert app.session.report, "el dataset no se analizó"
            assert app.query_one("#alerts", DataTable).row_count >= 0
            assert app.query_one("#cols", DataTable).row_count == 3
            # variables: marcar f2 como no usada con la tecla x
            app.action_goto(1)
            await pilot.pause()
            t = app.query_one("#cols", DataTable)
            t.move_cursor(row=1)
            t.focus()
            await pilot.press("x")
            await pilot.pause()
            assert app.session.recipe["roles"]["f2"] == "drop"
            assert "1" in str(app.query_one("#counts", Static).render())
            # partición
            app.action_goto(3)
            await pilot.pause(0.5)
            app.run_split()
            for _ in range(100):
                await pilot.pause(0.2)
                if app.session.split:
                    break
            assert app.session.split
            assert app.query_one("#classes", DataTable).row_count == 3
            # entrenar
            app.action_goto(4)
            await pilot.pause()
            app.query_one("#models").deselect_all()
            app.query_one("#models").select("decision_tree")
            await pilot.pause(0.5)
            app.run_train()
            for _ in range(300):
                await pilot.pause(0.2)
                if app.session.train:
                    break
            assert app.session.train
            await pilot.pause(0.5)
            assert app.query_one("#results", DataTable).row_count == 1
            assert app.query_one("#tabs", TabbedContent).active == "t-train"

    asyncio.run(go())


@pytest.mark.skipif(shutil.which("cc") is None, reason="sin compilador C")
def test_session_export(csv, tmp_path):
    s = Session(workdir=tmp_path / "w")
    s.inspect([str(csv)])
    s.run_train(["decision_tree"])
    rep = s.run_export("esp32-s3")
    assert rep["verification"]["model_match"] == 1.0
    assert rep["budget"]["flash_data"]["verdict_code"] in (
        "fits",
        "tight",
    )  # usa los límites de la placa
    assert (Path(s.export_dir) / "tm_model.h").exists()


def test_tui_environment_tab(tmp_path):
    async def go():
        app = TinyMLApp([], workdir=tmp_path / "w")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.5)
            app.action_goto(6)
            await pilot.pause(0.5)
            app.run_env()
            for _ in range(100):
                await pilot.pause(0.2)
                if app.query_one("#env-log").lines:
                    break
            text = "\n".join(str(ln.text) for ln in app.query_one("#env-log").lines)
            assert "numpy" in text and "textual" in text
            assert app.query_one("#tabs", TabbedContent).active == "t-env"

    asyncio.run(go())


def test_tui_english_labels_from_worker_threads(csv, tmp_path):
    """Con ``--lang en`` lo que vuelve de los hilos (tablas, avisos) debe salir en inglés."""
    from tinyml_studio.i18n import set_lang
    from tinyml_studio.tui.app import _message

    set_lang("en")

    async def go():
        app = TinyMLApp([str(csv)], workdir=tmp_path / "w")
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(100):
                await pilot.pause(0.2)
                if app.session.report:
                    break
            await pilot.pause(0.5)
            cols = app.query_one("#cols", DataTable)
            cells = {str(cols.get_cell_at((r, 2))) for r in range(cols.row_count)}
            assert "Input (X)" in cells and not any("Entrada" in c for c in cells)
            assert "Inputs (X)" in str(app.query_one("#counts", Static).render())
            steps = app.query_one("#steps", DataTable)
            assert steps.row_count > 0
            assert all(
                not str(steps.get_cell_at((r, 0))).islower() for r in range(steps.row_count)
            )  # etiquetas humanas, no 'impute' / 'scale'

    try:
        asyncio.run(go())
        assert _message(FileNotFoundError("does not exist: x")) == "Does not exist: x"
        assert _message(KeyError("k")).startswith("KeyError")
    finally:
        set_lang(None)


def test_session_train_rows_are_plain_text(csv, tmp_path):
    s = Session(workdir=tmp_path / "w")
    s.inspect([str(csv)])
    s.run_train(["decision_tree"])
    size = s.train_rows()[0][4]
    assert "{" not in size and "nodes=" in size


def test_tui_install_components_asks_for_confirmation(tmp_path, monkeypatch):
    from tinyml_studio import envmgr

    ran: list[list[str]] = []
    monkeypatch.setattr(envmgr, "run", lambda ids, on_line, **kw: (ran.append(ids), on_line("ok")))

    async def go():
        app = TinyMLApp([], workdir=tmp_path / "w")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.5)
            app.action_goto(6)
            await pilot.pause(0.3)
            sel = app.query_one("#comp-list")
            assert sel.option_count >= 3  # tui, keras, torch (+ tflm en Linux)
            app.install_components()  # nada elegido: no instala
            assert not ran
            sel.select("keras")
            app.install_components()  # 1.ª pulsación: pide confirmación, no instala
            await pilot.pause(0.2)
            assert not ran and str(app.query_one("#install-comp").label).lower().startswith(
                "confirm"
            )
            app.install_components()  # 2.ª pulsación: instala
            for _ in range(50):
                await pilot.pause(0.2)
                if ran:
                    break
            assert ran == [["keras"]]

    asyncio.run(go())


def test_tui_environment_check_button_is_visible(tmp_path):
    """El campo de ruta no debe empujar el botón «Revisar» fuera de la pantalla."""

    async def go():
        app = TinyMLApp([], workdir=tmp_path / "w")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.5)
            app.action_goto(6)
            await pilot.pause(0.5)
            btn = app.query_one("#run-env")
            assert btn.region.width > 0 and btn.region.right <= app.size.width

    asyncio.run(go())


def test_tui_report_tab_saves_charts_and_pdf(csv, tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("fpdf")

    async def go():
        app = TinyMLApp([str(csv)], workdir=tmp_path / "w")
        async with app.run_test(size=(130, 40)) as pilot:
            for _ in range(100):
                await pilot.pause(0.2)
                if app.session.report:
                    break
            app.action_goto(7)  # sin entrenar: guía y no falla
            await pilot.pause(0.5)
            assert app.query_one("#rep-charts").option_count == 0
            app.session.prepare()
            app.session.run_train(["decision_tree"])
            app.action_goto(0)
            await pilot.pause(0.2)
            app.action_goto(7)
            await pilot.pause(0.6)
            assert app.query_one("#rep-charts").option_count >= 5
            assert "decision_tree" in "\n".join(
                str(ln.text) for ln in app.query_one("#rep-log").lines
            )
            out = tmp_path / "informe"
            app.query_one("#repdir").value = str(out)
            app.save_charts()
            for _ in range(100):
                await pilot.pause(0.2)
                if (out / "comparison.png").exists():
                    break
            assert (out / "comparison.png").read_bytes()[:4] == b"\x89PNG"
            app.make_pdf()
            for _ in range(150):
                await pilot.pause(0.2)
                if (out / "informe.pdf").exists():
                    break
            assert (out / "informe.pdf").read_bytes()[:5] == b"%PDF-"
            # sin carpeta de destino: avisa y no escribe nada
            app.query_one("#repdir").value = ""
            app.save_charts()
            await pilot.pause(0.2)

    asyncio.run(go())


def test_session_reason_of_explains_and_notes_user_change(tmp_path):
    p = tmp_path / "t.csv"
    n = 120
    pd.DataFrame(
        {
            "rssi": RNG.integers(-80, -40, n),
            "x": RNG.normal(size=n),
            "label": np.repeat(["a", "b"], n // 2),
        }
    ).to_csv(p, index=False)
    s = Session(workdir=tmp_path / "w")
    s.inspect([str(p)])
    assert "rssi" in s.reason_of("rssi")
    assert s.reason_of("x")
    s.set_role("rssi", "feature")
    assert "ti" in s.reason_of("rssi") or "you" in s.reason_of("rssi")


def test_tui_variables_tab_shows_reason(csv, tmp_path):
    async def go():
        app = TinyMLApp([str(csv)], workdir=tmp_path / "w")
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(100):
                await pilot.pause(0.2)
                if app.session.report:
                    break
            app.action_goto(1)
            await pilot.pause()
            assert "f1" in str(app.query_one("#reason", Static).render())

    asyncio.run(go())
