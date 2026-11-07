"""Aplicación Textual: el asistente de TinyML Studio en la terminal."""

from __future__ import annotations

import functools
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    LoadingIndicator,
    RichLog,
    Select,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)

from tinyml_studio.i18n import get_lang, tr, use_lang
from tinyml_studio.tui.session import Session

STEPS = ("dataset", "variables", "cleaning", "split", "train", "export", "env", "report")


def _names() -> list[str]:
    return [
        tr("Dataset", "Dataset"),
        tr("Variables", "Variables"),
        tr("Limpieza", "Cleaning"),
        tr("Partición", "Split"),
        tr("Entrenar", "Train"),
        tr("Exportar", "Export"),
        tr("Entorno", "Environment"),
        tr("Informe", "Report"),
    ]


def _kind(k: str) -> str:
    """Tipo de columna en lenguaje humano (la GUI usa las mismas etiquetas)."""
    return {
        "numeric_continuous": tr("Número (continuo)", "Number (continuous)"),
        "numeric_discrete": tr("Número (discreto)", "Number (discrete)"),
        "binary": tr("Sí/No", "Yes/No"),
        "categorical": tr("Categoría", "Category"),
        "text": tr("Texto libre", "Free text"),
        "datetime": tr("Fecha/hora", "Date/time"),
        "id": tr("Identificador", "Identifier"),
        "constant": tr("Constante", "Constant"),
        "empty": tr("Vacía", "Empty"),
    }.get(k, k)


def _role(r: str) -> str:
    return {
        "feature": tr("Entrada (X)", "Input (X)"),
        "target": tr("Objetivo (y)", "Target (y)"),
        "group": tr("Grupo", "Group"),
        "time": tr("Tiempo", "Time"),
        "meta": tr("Metadato (no se usa)", "Metadata (unused)"),
        "drop": tr("Descartar", "Discard"),
    }.get(r, r)


def _severity(s: str) -> str:
    return {
        "high": tr("Alta", "High"),
        "medium": tr("Media", "Medium"),
        "info": tr("Info", "Info"),
    }.get(s, s)


def _step(k: str) -> str:
    return {
        "absurd_to_nan": tr("Valores absurdos → faltantes", "Absurd values → missing"),
        "impute": tr("Rellenar faltantes", "Fill missing values"),
        "scale": tr("Escalar", "Scale"),
        "clip": tr("Recortar atípicos", "Clip outliers"),
        "replace_sentinel": tr("Reemplazar valores centinela", "Replace sentinel values"),
        "drop_duplicates": tr("Quitar duplicados", "Remove duplicates"),
        "drop_rows": tr("Quitar filas", "Drop rows"),
        "drop_columns": tr("Quitar columnas", "Drop columns"),
        "range_nan": tr("Fuera de rango → faltantes", "Out-of-range → missing"),
    }.get(k, k)


def _strategy(k: str) -> str:
    return {
        "temporal_blocks": tr("bloques temporales", "temporal blocks"),
        "group": tr("por grupo", "by group"),
        "stratified": tr("estratificada", "stratified"),
        "random": tr("aleatoria", "random"),
    }.get(k, k)


def _langed(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Los hilos de trabajo no heredan el idioma (ContextVar): lo fija el que llama.

    ``call_from_thread`` también copia el contexto del hilo, así que los mensajes que vuelven a la
    interfaz salen en el idioma elegido con ``--lang`` y no en el predeterminado.
    """

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with use_lang(self._lang):
            return fn(self, *args, **kwargs)

    return wrapper


def _message(e: Exception) -> str:
    """Errores del usuario (ruta mala, falta un paso) sin el nombre de la clase de Python."""
    if isinstance(e, (ValueError, FileNotFoundError)) and e.args and isinstance(e.args[0], str):
        msg = e.args[0]
        return msg[:1].upper() + msg[1:]
    return f"{type(e).__name__}: {e}"


def _board_label(b: dict[str, Any]) -> str:
    """Nombre + memoria de la placa; las de Linux (Raspberry Pi) traen RAM, no SRAM."""
    if b.get("sram_kb"):
        mem = f"{b['sram_kb']:g} KB SRAM"
    elif b.get("ram_kb"):
        mem = f"{b['ram_kb']:g} KB RAM"
    else:
        mem = tr("memoria sin dato", "memory unknown")
    return f"{b['name']} · {mem}"


class RolesTable(DataTable):
    """Tabla de columnas: t = objetivo (y), f = entrada (X), x = sin usar."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # las descripciones del pie siguen el idioma activo (BINDINGS de clase se fijan al importar)
        self._bindings.bind("t", "role('target')", tr("t: Objetivo (y)", "t: Target (y)"))
        self._bindings.bind("f", "role('feature')", tr("f: Entrada (X)", "f: Input (X)"))
        self._bindings.bind("x", "role('drop')", tr("x: No usar", "x: Ignore"))

    def action_role(self, role: str) -> None:
        self.app.set_role_at_cursor(role)  # type: ignore[attr-defined]


class TinyMLApp(App):
    CSS = """
    Screen { background: $surface; }
    TabbedContent { height: 1fr; }
    .hint { color: $text-muted; margin: 0 0 1 0; }
    .row { height: auto; margin: 0 0 1 0; }
    #path, #outdir, #pypath, #repdir { width: 1fr; }
    .row2 { height: 10; margin: 0 0 1 0; }
    #rep-charts, #rep-formats { width: 1fr; height: 10; }
    DataTable { height: 1fr; min-height: 6; }
    #summary-box { height: 10; border: round $primary 40%; padding: 0 1; }
    #counts, #split-info { margin: 1 0; }
    #reason { margin: 0 0 1 0; color: $warning; }
    RichLog { height: 1fr; min-height: 6; border: round $primary 40%; }
    LoadingIndicator { height: 3; }
    SelectionList { height: 10; }
    """
    TITLE = "TinyML Studio"
    BINDINGS = [
        *[Binding(str(i + 1), f"goto({i})", f"{i + 1}", show=False) for i in range(1, len(STEPS))],
    ]

    def __init__(self, start_paths: list[str] | None = None, workdir: Path | None = None):
        super().__init__()
        self.session = Session()
        if workdir:
            self.session.workdir = Path(workdir)
            self.session.workdir.mkdir(parents=True, exist_ok=True)
        self.start_paths = start_paths or []
        self._lang = get_lang()
        self._armed: tuple[str, ...] | None = None  # componentes pendientes de confirmar
        self.sub_title = tr("terminal", "terminal")
        # pie de pantalla en el idioma activo; las teclas 1-8 se muestran como una sola entrada
        self.bind("q", "quit", description=tr("Salir", "Quit"))
        self.bind("1", "goto(0)", description=tr("Ir a la pestaña", "Go to tab"), key_display="1-8")
        self.bind("ctrl+s", "save_recipe", description=tr("Guardar receta", "Save recipe"))

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        names = _names()
        boards = self.session.boards()
        self._board_options = [(_board_label(b), b["id"]) for b in boards]
        self._board_default = (
            "esp32-s3" if any(b["id"] == "esp32-s3" for b in boards) else boards[0]["id"]
        )
        yield Header()
        with TabbedContent(id="tabs"):
            with TabPane(f"1 {names[0]}", id="t-dataset"):
                yield Static(
                    tr(
                        "Escribe la ruta de un CSV o de una carpeta "
                        "(varias rutas separadas por coma).",
                        "Type the path of a CSV or a folder (several paths separated by commas).",
                    ),
                    classes="hint",
                )
                with Horizontal(classes="row"):
                    yield Input(placeholder="carpeta con tus CSV", id="path")
                    yield Button(tr("Analizar", "Analyze"), id="analyze", variant="primary")
                yield LoadingIndicator(id="busy-dataset")
                with VerticalScroll(id="summary-box"):
                    yield Static(
                        tr(
                            "Aún no hay dataset: Escribe una ruta y pulsa Analizar (o Enter).",
                            "No dataset yet: Type a path and press Analyze (or Enter).",
                        ),
                        id="summary",
                    )
                yield DataTable(id="alerts", cursor_type="row")
            with TabPane(f"2 {names[1]}", id="t-variables"):
                yield Static(
                    tr(
                        "Con una fila seleccionada: T = objetivo (y) · f = entrada (X) · "
                        "x = no usar.",
                        "On a selected row: T = target (y) · f = input (X) · x = ignore.",
                    ),
                    classes="hint",
                )
                yield Static(
                    tr(
                        "Primero analiza un dataset (pestaña 1).",
                        "Analyze a dataset first (tab 1).",
                    ),
                    id="counts",
                )
                yield Static("", id="reason")
                yield RolesTable(id="cols", cursor_type="row")
            with TabPane(f"3 {names[2]}", id="t-cleaning"):
                yield Static(
                    tr(
                        "Pasos de limpieza sugeridos (se ajustan solo con train, "
                        "sin fuga de datos).",
                        "Suggested cleaning steps (fitted on train only, no data leakage).",
                    ),
                    classes="hint",
                )
                yield DataTable(id="steps", cursor_type="row")
            with TabPane(f"4 {names[3]}", id="t-split"):
                yield Button(
                    tr("Limpiar y partir", "Clean and split"), id="run-split", variant="primary"
                )
                yield LoadingIndicator(id="busy-split")
                yield Static(
                    tr(
                        "Pulsa el botón para limpiar con train y partir 80/10/10 "
                        "(requiere un dataset analizado).",
                        "Press the button to clean with train and split 80/10/10 "
                        "(needs an analyzed dataset).",
                    ),
                    id="split-info",
                )
                yield DataTable(id="classes", cursor_type="row")
            with TabPane(f"5 {names[4]}", id="t-train"):
                yield Static(
                    tr(
                        "Elige los modelos y entrena. El test nunca decide el ganador.",
                        "Pick the models and train. The test set never chooses the winner.",
                    ),
                    classes="hint",
                )
                yield SelectionList[str](id="models")
                yield Button(tr("Entrenar", "Train"), id="run-train", variant="primary")
                yield LoadingIndicator(id="busy-train")
                yield DataTable(id="results", cursor_type="row")
                yield RichLog(id="train-log", wrap=True, markup=False)
            with TabPane(f"6 {names[5]}", id="t-export"):
                yield Static(
                    tr(
                        "Placa de destino: Se exporta a C y se verifica que cabe.",
                        "Target board: Exports to C and checks that it fits.",
                    ),
                    classes="hint",
                )
                yield Select[str](
                    self._board_options, id="board", allow_blank=False, value=self._board_default
                )
                yield Button(tr("Exportar a C", "Export to C"), id="run-export", variant="primary")
                yield LoadingIndicator(id="busy-export")
                yield RichLog(id="export-log", wrap=True, markup=False)
                with Horizontal(classes="row"):
                    yield Input(
                        placeholder=tr("Carpeta de destino", "Destination folder"), id="outdir"
                    )
                    yield Button(tr("Guardar archivos", "Save files"), id="save-files")
            with TabPane(f"7 {names[6]}", id="t-env"):
                yield Static(
                    tr(
                        "Revisa qué Python usa el núcleo y qué le falta. "
                        "Deja la ruta vacía para el actual "
                        "o escribe la de tu venv/conda (p. ej. ~/miniconda3/envs/tml/bin/python).",
                        "Check which Python the core uses and what it is missing. "
                        "Leave the path empty "
                        "for the current one or type your venv/conda path.",
                    ),
                    classes="hint",
                )
                with Horizontal(classes="row"):
                    yield Input(
                        placeholder=tr("Ruta del Python (opcional)", "Python path (optional)"),
                        id="pypath",
                    )
                    yield Button(tr("Revisar", "Check"), id="run-env", variant="primary")
                yield Static(
                    tr(
                        "Instalar componentes en el entorno actual (versiones fijadas y hashes, "
                        "desde pypi.org). Pide confirmación.",
                        "Install components into the current environment (pinned versions and "
                        "hashes, from pypi.org). Asks for confirmation.",
                    ),
                    classes="hint",
                )
                yield SelectionList[str](id="comp-list")
                yield Button(tr("Instalar componentes", "Install components"), id="install-comp")
                yield LoadingIndicator(id="busy-env")
                yield RichLog(id="env-log", wrap=True, markup=False)
            with TabPane(f"8 {names[7]}", id="t-report"):
                yield Static(
                    tr(
                        "Guarda las gráficas (PNG, JPG, SVG) y el informe PDF en una carpeta. "
                        "En la terminal no se ven imágenes: Abre los archivos con tu visor.",
                        "Save the charts (PNG, JPG, SVG) and the PDF report to a folder. "
                        "A terminal cannot show images: Open the files with your viewer.",
                    ),
                    classes="hint",
                )
                with Horizontal(classes="row2"):
                    yield SelectionList[str](id="rep-charts")
                    yield SelectionList[str](id="rep-formats")
                with Horizontal(classes="row"):
                    yield Input(
                        placeholder=tr(
                            "Carpeta de destino (p. ej. ./informe)",
                            "Destination folder (e.g. ./report)",
                        ),
                        id="repdir",
                    )
                    yield Button(tr("Guardar gráficas", "Save charts"), id="save-charts")
                    yield Button(tr("Informe PDF", "PDF report"), id="make-pdf", variant="primary")
                yield LoadingIndicator(id="busy-report")
                yield RichLog(id="rep-log", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self._fill_components()
        self._fill_formats()
        for wid in (
            "busy-dataset",
            "busy-split",
            "busy-train",
            "busy-export",
            "busy-env",
            "busy-report",
        ):
            self.query_one(f"#{wid}").display = False
        a = self.query_one("#alerts", DataTable)
        a.add_columns(tr("Gravedad", "Severity"), tr("Mensaje", "Message"))
        c = self.query_one("#cols", DataTable)
        c.add_columns(tr("Columna", "Column"), tr("Tipo", "Type"), tr("Rol", "Role"))
        s = self.query_one("#steps", DataTable)
        s.add_columns(tr("Paso", "Step"), tr("Columnas", "Columns"), tr("Por qué", "Why"))
        k = self.query_one("#classes", DataTable)
        k.add_columns(tr("Clase", "Class"), "train", "val", "test")
        r = self.query_one("#results", DataTable)
        r.add_columns(
            tr("Modelo", "Model"), "val", "test", tr("Ajuste (s)", "Fit (s)"), tr("Tamaño", "Size")
        )
        if self.start_paths:
            self.query_one("#path", Input).value = ",".join(self.start_paths)
            self.analyze()

    # ------------------------------------------------------------------ navegación
    def action_goto(self, i: int) -> None:
        self.query_one("#tabs", TabbedContent).active = f"t-{STEPS[i]}"

    def _busy(self, name: str, on: bool) -> None:
        self.query_one(f"#busy-{name}").display = on

    def _fail(self, e: Exception) -> None:
        self.notify(_message(e), severity="error", timeout=12)

    # ------------------------------------------------------------------ 1. dataset
    @on(Button.Pressed, "#analyze")
    @on(Input.Submitted, "#path")
    def analyze(self) -> None:
        paths = [p for p in self.query_one("#path", Input).value.split(",") if p.strip()]
        self._busy("dataset", True)
        self._analyze(paths)

    @work(thread=True, exclusive=True, group="dataset")
    @_langed
    def _analyze(self, paths: list[str]) -> None:
        try:
            report = self.session.inspect(paths)
        except Exception as e:  # noqa: BLE001 - se muestra al usuario
            self.call_from_thread(self._busy, "dataset", False)
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self._show_dataset, report)

    def _show_dataset(self, report: dict[str, Any]) -> None:
        self._busy("dataset", False)
        self.query_one("#summary", Static).update(report.get("summary") or report["summary_es"])
        t = self.query_one("#alerts", DataTable)
        t.clear()
        for al in report["alerts"][:200]:
            t.add_row(_severity(al["severity"]), al["message"])
        if len(report["alerts"]) > 200:
            t.add_row(
                "…",
                tr(
                    "… y {n} alertas más (el resumen de arriba las cuenta todas)",
                    "… and {n} more alerts (the summary above counts them all)",
                    n=len(report["alerts"]) - 200,
                ),
            )
        self._fill_variables()
        self._fill_steps()
        models = self.query_one("#models", SelectionList)
        models.clear_options()
        avail = self.session.available_models()
        slow = {"svm", "svr", "gradient_boosting"}
        models.add_options([(m, m, m not in slow) for m in avail])
        self.notify(tr("Dataset analizado", "Dataset analyzed"))

    # ------------------------------------------------------------------ 2. variables
    def _fill_variables(self) -> None:
        t = self.query_one("#cols", DataTable)
        t.clear()
        for c in self.session.report["columns"]:  # type: ignore[index]
            t.add_row(
                c["name"],
                _kind(c["kind"]),
                _role(self.session.role_of(c["name"])),
                key=c["name"],
            )
        self._update_counts()
        self._show_reason()

    def _show_reason(self) -> None:
        t = self.query_one("#cols", DataTable)
        box = self.query_one("#reason", Static)
        if self.session.report is None or t.row_count == 0:
            box.update("")
            return
        col = str(t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value)
        box.update(
            Text(tr("Motivo de «{c}»: ", "Why «{c}»: ", c=col) + self.session.reason_of(col))
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "cols":
            self._show_reason()

    def _update_counts(self) -> None:
        c = self.session.counts()
        self.query_one("#counts", Static).update(
            tr(
                "Entradas (X): {i} · Objetivo (y): {t} · Sin usar: {u}",
                "Inputs (X): {i} · Target (y): {t} · Unused: {u}",
                i=c["inputs"],
                t=c["target"] or "—",
                u=c["unused"],
            )
        )

    def set_role_at_cursor(self, role: str) -> None:
        if self.session.recipe is None:
            return
        t = self.query_one("#cols", DataTable)
        if t.row_count == 0:
            return
        col = str(t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value)
        try:
            self.session.set_role(col, role)
        except Exception as e:  # noqa: BLE001
            self._fail(e)
            return
        cursor = t.cursor_row
        self._fill_variables()
        t.move_cursor(row=cursor)
        self._fill_steps()

    # ------------------------------------------------------------------ 3. limpieza
    def _fill_steps(self) -> None:
        t = self.query_one("#steps", DataTable)
        t.clear()
        for s in (self.session.recipe or {}).get("cleaning", []):
            cols = s.get("params", {}).get("columns", [])
            t.add_row(_step(s["kind"]), str(len(cols)), s.get("why", ""))

    # ------------------------------------------------------------------ 4. partición
    @on(Button.Pressed, "#run-split")
    def run_split(self) -> None:
        if self.session.recipe is None:
            self.notify(
                tr("Primero analiza un dataset", "Analyze a dataset first"), severity="warning"
            )
            return
        self._busy("split", True)
        self._split()

    @work(thread=True, exclusive=True, group="split")
    @_langed
    def _split(self) -> None:
        try:
            res = self.session.prepare()
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._busy, "split", False)
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self._show_split, res)

    def _show_split(self, res: dict[str, Any]) -> None:
        self._busy("split", False)
        sz = res["sizes"]
        strat = _strategy(res["strategy"].get("strategy", "?"))
        self.query_one("#split-info", Static).update(
            tr(
                "Estrategia: {s} · train {a} · val {b} · test {c}",
                "Strategy: {s} · train {a} · val {b} · test {c}",
                s=strat,
                a=sz.get("train"),
                b=sz.get("val"),
                c=sz.get("test"),
            )
        )
        t = self.query_one("#classes", DataTable)
        t.clear()
        names = sorted({k for d in res["classes"].values() for k in d})
        for n in names:
            t.add_row(
                n, *[str(res["classes"].get(s, {}).get(n, 0)) for s in ("train", "val", "test")]
            )

    # ------------------------------------------------------------------ 5. entrenamiento
    @on(Button.Pressed, "#run-train")
    def run_train(self) -> None:
        if self.session.recipe is None:
            self.notify(
                tr("Primero analiza un dataset", "Analyze a dataset first"), severity="warning"
            )
            return
        chosen = list(self.query_one("#models", SelectionList).selected)
        if not chosen:
            self.notify(
                tr("Elige al menos un modelo", "Pick at least one model"), severity="warning"
            )
            return
        self._busy("train", True)
        self.query_one("#train-log", RichLog).clear()
        self._train(chosen)

    @work(thread=True, exclusive=True, group="train")
    @_langed
    def _train(self, models: list[str]) -> None:
        try:
            res = self.session.run_train(models)
            from tinyml_studio.train.report import summary

            text = summary(res)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._busy, "train", False)
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self._show_train, text)

    def _show_train(self, text: str) -> None:
        self._busy("train", False)
        t = self.query_one("#results", DataTable)
        key = (self.session.train or {}).get("selection", {}).get("metric", "")
        t.clear(columns=True)  # las cabeceras dicen qué métrica se muestra
        t.add_columns(
            tr("Modelo", "Model"),
            f"{key} val",
            f"{key} test",
            tr("Ajuste (s)", "Fit (s)"),
            tr("Tamaño", "Size"),
        )
        for row in self.session.train_rows():
            t.add_row(*row)
        log = self.query_one("#train-log", RichLog)
        log.write(tr("★ = mejor modelo según validación.", "★ = best model by validation."))
        log.write(text)
        self.notify(tr("Entrenamiento terminado", "Training finished"))

    # ------------------------------------------------------------------ 6. exportar
    @on(Button.Pressed, "#run-export")
    def run_export(self) -> None:
        if self.session.run_dir is None:
            self.notify(tr("Primero entrena un modelo", "Train a model first"), severity="warning")
            return
        board = str(self.query_one("#board", Select).value)
        self._busy("export", True)
        self._export(board)

    @work(thread=True, exclusive=True, group="export")
    @_langed
    def _export(self, board: str) -> None:
        try:
            rep = self.session.run_export(board)
            from tinyml_studio.export.report import summary

            text = summary(rep, str(self.session.export_dir))
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._busy, "export", False)
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self._show_export, text)

    def _show_export(self, text: str) -> None:
        self._busy("export", False)
        log = self.query_one("#export-log", RichLog)
        log.clear()
        log.write(text)

    # ------------------------------------------------------------------ 7. entorno
    @on(Button.Pressed, "#run-env")
    @on(Input.Submitted, "#pypath")
    def run_env(self) -> None:
        self._busy("env", True)
        self._env(self.query_one("#pypath", Input).value)

    @work(thread=True, exclusive=True, group="env")
    @_langed
    def _env(self, python: str) -> None:
        try:
            text = self.session.env_report(python, ["base", "tui", "keras", "torch", "tflm"])
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._busy, "env", False)
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self._show_env, text)

    def _show_env(self, text: str) -> None:
        self._busy("env", False)
        log = self.query_one("#env-log", RichLog)
        log.clear()
        log.write(text)

    def _fill_components(self) -> None:
        from tinyml_studio import envmgr

        comp = self.query_one("#comp-list", SelectionList)
        comp.clear_options()
        key = "es" if self._lang == "es" else "en"
        try:
            options = [
                (c["label"][key], c["id"], False)
                for c in envmgr.available_components()
                if c["id"] != "base"
            ]
        except (OSError, ValueError, KeyError):  # sin manifiesto: la TUI abre igual, sin lista
            options = []
        comp.add_options(options)

    @on(Button.Pressed, "#install-comp")
    def install_components(self) -> None:
        btn = self.query_one("#install-comp", Button)
        chosen = tuple(self.query_one("#comp-list", SelectionList).selected)
        if getattr(sys, "frozen", False):
            self.notify(
                tr(
                    "El núcleo incluido no se modifica: Usa tu entorno o el instalador de la app.",
                    "The bundled core is not modified: Use your environment or the app installer.",
                ),
                severity="warning",
            )
            return
        if not chosen:
            self.notify(
                tr("Elige al menos un componente", "Pick at least one component"),
                severity="warning",
            )
            return
        if self._armed != chosen:  # primera pulsación: pide confirmación
            self._armed = chosen
            btn.label = tr("Confirmar instalación", "Confirm installation")
            log = self.query_one("#env-log", RichLog)
            log.clear()
            log.write(
                tr(
                    "Se instalarán {c} (más el núcleo si falta) en {p}\n"
                    "Desde pypi.org, con versiones fijadas y hashes. "
                    "Pulsa «Confirmar instalación» para continuar o elige otros.",
                    "Will install {c} (plus the core if missing) into {p}\n"
                    "From pypi.org, with pinned versions and hashes. "
                    "Press “Confirm installation” to continue or pick others.",
                    c=", ".join(chosen),
                    p=sys.executable,
                )
            )
            return
        self._armed = None
        btn.label = tr("Instalar componentes", "Install components")
        self._busy("env", True)
        self.query_one("#env-log", RichLog).clear()
        self._install(list(chosen))

    @work(thread=True, exclusive=True, group="env")
    @_langed
    def _install(self, ids: list[str]) -> None:
        from tinyml_studio import envmgr

        log = self.query_one("#env-log", RichLog)
        try:
            envmgr.run(ids, lambda line: self.call_from_thread(log.write, line))
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._busy, "env", False)
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self._busy, "env", False)
        self.call_from_thread(self.notify, tr("Instalación terminada", "Installation finished"))

    # ------------------------------------------------------------------ 8. informe
    def _fill_formats(self) -> None:
        fm = self.query_one("#rep-formats", SelectionList)
        fm.clear_options()
        fm.add_options([("PNG", "png", True), ("JPG", "jpg", False), ("SVG", "svg", False)])

    @on(TabbedContent.TabActivated)
    def _tab_changed(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "t-report":
            self._refresh_report()

    def _refresh_report(self) -> None:
        log = self.query_one("#rep-log", RichLog)
        sel = self.query_one("#rep-charts", SelectionList)
        keep = set(sel.selected)
        sel.clear_options()
        log.clear()
        if self.session.train is None:
            log.write(tr("Primero entrena un modelo (pestaña 5).", "Train a model first (tab 5)."))
            return
        try:
            charts = self.session.charts_available()
            sel.add_options([(label, cid, not keep or cid in keep) for cid, label in charts])
            log.write(self.session.preview())
        except Exception as e:  # noqa: BLE001
            self._fail(e)

    def _report_target(self) -> str | None:
        dest = self.query_one("#repdir", Input).value.strip()
        if not dest:
            self.notify(
                tr("Escribe una carpeta de destino", "Type a destination folder"),
                severity="warning",
            )
            return None
        if self.session.train is None:
            self.notify(tr("Primero entrena un modelo", "Train a model first"), severity="warning")
            return None
        return dest

    @on(Button.Pressed, "#save-charts")
    def save_charts(self) -> None:
        dest = self._report_target()
        if dest is None:
            return
        fmts = list(self.query_one("#rep-formats", SelectionList).selected)
        only = list(self.query_one("#rep-charts", SelectionList).selected)
        if not fmts or not only:
            self.notify(
                tr("Elige gráficas y formatos", "Pick charts and formats"), severity="warning"
            )
            return
        self._busy("report", True)
        self._save_charts(dest, fmts, only)

    @work(thread=True, exclusive=True, group="report")
    @_langed
    def _save_charts(self, dest: str, fmts: list[str], only: list[str]) -> None:
        try:
            paths = self.session.save_charts(dest, fmts, only)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._busy, "report", False)
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self._show_saved, [str(p) for p in paths])

    @on(Button.Pressed, "#make-pdf")
    def make_pdf(self) -> None:
        dest = self._report_target()
        if dest is None:
            return
        self._busy("report", True)
        self._pdf(dest, list(self.query_one("#rep-charts", SelectionList).selected))

    @work(thread=True, exclusive=True, group="report")
    @_langed
    def _pdf(self, dest: str, charts: list[str]) -> None:
        try:
            path = self.session.make_pdf(dest, charts)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._busy, "report", False)
            self.call_from_thread(self._fail, e)
            return
        self.call_from_thread(self._show_saved, [str(path)])

    def _show_saved(self, paths: list[str]) -> None:
        self._busy("report", False)
        log = self.query_one("#rep-log", RichLog)
        folder = os.path.commonpath([os.path.dirname(p) for p in paths]) if paths else ""
        log.write(tr("Guardado en {d}:", "Saved to {d}:", d=folder))
        log.write(", ".join(os.path.basename(p) for p in paths))
        self.notify(tr("{n} archivo(s) guardado(s)", "{n} file(s) saved", n=len(paths)))

    @on(Button.Pressed, "#save-files")
    def save_files(self) -> None:
        dest = self.query_one("#outdir", Input).value.strip()
        if not dest:
            self.notify(
                tr("Escribe una carpeta de destino", "Type a destination folder"),
                severity="warning",
            )
            return
        if self.session.export_dir is None:
            self.notify(tr("Primero exporta", "Export first"), severity="warning")
            return
        try:
            out = Path(dest).expanduser()
            shutil.copytree(self.session.export_dir, out, dirs_exist_ok=True)
            self.session.save_recipe(str(out / "recipe.json"))
        except Exception as e:  # noqa: BLE001
            self._fail(e)
            return
        self.notify(tr("Guardado en {p}", "Saved to {p}", p=str(out)))

    def action_save_recipe(self) -> None:
        if self.session.recipe is None:
            self.notify(
                tr("Primero analiza un dataset", "Analyze a dataset first"), severity="warning"
            )
            return
        dest = self.query_one("#outdir", Input).value.strip() or str(self.session.workdir)
        try:
            p = self.session.save_recipe(str(Path(dest).expanduser() / "recipe.json"))
        except Exception as e:  # noqa: BLE001
            self._fail(e)
            return
        self.notify(tr("Receta guardada: {p}", "Recipe saved: {p}", p=str(p)))


def run(paths: list[str] | None = None) -> int:
    TinyMLApp(paths).run()
    return 0
