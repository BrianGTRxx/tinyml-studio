"""Estado y operaciones de la TUI, sin dependencias de Textual (se prueban solas).

Todo lo que hace la interfaz de terminal pasa por aquí y llama al mismo núcleo que la GUI y la
CLI: la receta JSON es el estado reproducible y se puede abrir en cualquiera de las tres.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinyml_studio.i18n import tr

ROLES = ("feature", "target", "drop", "meta", "group", "time")


def _complexity(c: Any) -> str:
    """``{'nodes': 105, 'depth': 10}`` -> ``nodes=105 depth=10`` (igual que la GUI)."""
    if isinstance(c, dict):
        return " ".join(f"{k}={v}" for k, v in c.items())
    return "" if c is None else str(c)


@dataclass
class Session:
    paths: list[str] = field(default_factory=list)
    report: dict[str, Any] | None = None
    recipe: dict[str, Any] | None = None
    split: dict[str, Any] | None = None
    train: dict[str, Any] | None = None
    run_dir: Path | None = None
    export: dict[str, Any] | None = None
    export_dir: Path | None = None
    workdir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="tinyml-tui-")))

    # ------------------------------------------------------------------ 1. dataset
    def inspect(self, paths: list[str]) -> dict[str, Any]:
        from tinyml_studio.dataset import inspect_files

        clean = [p.strip().strip('"') for p in paths if p.strip()]
        if not clean:
            raise ValueError(tr("indica al menos una ruta", "enter at least one path"))
        for p in clean:
            if not Path(p).exists():
                raise FileNotFoundError(tr("no existe: {p}", "does not exist: {p}", p=p))
        self.report = inspect_files(clean)
        self.recipe = json.loads(json.dumps(self.report["recipe"]))
        self.paths = clean
        self.split = self.train = self.export = self.run_dir = self.export_dir = None
        return self.report

    def load_recipe(self, path: str) -> dict[str, Any]:
        recipe = json.loads(Path(path).read_text(encoding="utf-8"))
        self.inspect(list(recipe["files"]))
        self.recipe = recipe
        return recipe

    def save_recipe(self, path: str) -> Path:
        if not self.recipe:
            raise ValueError(tr("primero abre un dataset", "open a dataset first"))
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.recipe, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    # ------------------------------------------------------------------ 2. variables
    def role_of(self, column: str) -> str:
        assert self.recipe is not None and self.report is not None
        if column in self.recipe["roles"]:
            return self.recipe["roles"][column]
        col = next((c for c in self.report["columns"] if c["name"] == column), None)
        return col["role"] if col else "feature"

    def reason_of(self, column: str) -> str:
        """Por qué el rol actual: motivo de la sugerencia, o aviso si lo cambió el usuario."""
        assert self.report is not None
        col = next((c for c in self.report["columns"] if c["name"] == column), None)
        if col is None:
            return ""
        if self.role_of(column) != col["role"]:
            return tr(
                "Cambiado por ti (la sugerencia era: {r}).",
                "Changed by you (the suggestion was: {r}).",
                r=col["role"],
            )
        return str(col.get("role_reason") or "") or tr(
            "Sin nada raro: Se usa como entrada.", "Nothing unusual: Used as an input."
        )

    def set_role(self, column: str, role: str) -> None:
        """Igual que la GUI: un único objetivo coherente en la receta."""
        if self.recipe is None:
            raise ValueError(tr("primero abre un dataset", "open a dataset first"))
        if role not in ROLES:
            raise ValueError(role)
        r = self.recipe
        previous = self.role_of(column)
        r["roles"][column] = role
        if role == "target":
            old = r.get("target")
            if old and old != column and r["roles"].get(old) == "target":
                r["roles"][old] = "feature"
            r["target"] = column
            if r.get("label_from_file"):
                r["label_from_file"]["enabled"] = False
        elif previous == "target" and r.get("target") == column:
            r["target"] = None
        r["drop"] = [k for k, v in r["roles"].items() if v == "drop" and k != "source"]
        self.split = self.train = self.export = None

    def counts(self) -> dict[str, Any]:
        assert self.recipe is not None and self.report is not None
        roles = [self.role_of(c["name"]) for c in self.report["columns"]]
        lff = (self.recipe.get("label_from_file") or {}).get("enabled")
        return {
            "inputs": roles.count("feature"),
            "unused": roles.count("drop") + roles.count("meta"),
            "target": tr("nombre del archivo", "file name") if lff else self.recipe.get("target"),
        }

    # ------------------------------------------------------------------ 3-4. limpieza y partición
    def prepare(self, seed: int = 42) -> dict[str, Any]:
        from tinyml_studio.clean import prepare_splits

        if self.recipe is None:
            raise ValueError(tr("primero abre un dataset", "open a dataset first"))
        prep = prepare_splits(self.recipe, seed=seed)
        dist: dict[str, dict[str, int]] = {}
        if prep.target and prep.target in prep.clean["train"].columns:
            for name, df in prep.clean.items():
                dist[name] = {str(k): int(v) for k, v in df[prep.target].value_counts().items()}
        self.split = {
            "sizes": {k: int(len(v)) for k, v in prep.clean.items()},
            "classes": dist,
            "target": prep.target,
            "steps": prep.pipeline.to_list(),
            "strategy": self.recipe.get("split", {}),
        }
        return self.split

    # ------------------------------------------------------------------ 5. entrenamiento
    def available_models(self) -> list[str]:
        from tinyml_studio.train.models import available

        task = (self.recipe or {}).get("task", "classification")
        return list(available(task)) if task in ("classification", "regression") else []

    def run_train(self, models: list[str], seed: int = 42) -> dict[str, Any]:
        from tinyml_studio.train import train_models

        if self.recipe is None:
            raise ValueError(tr("primero abre un dataset", "open a dataset first"))
        out = self.workdir / "run"
        self.train = train_models(self.recipe, models or None, seed=seed, out_dir=out)
        self.run_dir = out
        self.export = None
        return self.train

    def train_rows(self) -> list[tuple[str, ...]]:
        assert self.train is not None
        key = self.train["selection"]["metric"]
        best = self.train["selection"]["best"]
        rows = []
        for name, m in self.train["models"].items():
            rows.append(
                (
                    f"{name} ★" if name == best else name,
                    f"{m['val'].get(key, float('nan')):.3f}",
                    f"{m['test'].get(key, float('nan')):.3f}",
                    f"{m['fit_seconds']:.2f}",
                    _complexity(m.get("complexity")),
                )
            )
        return rows

    # ------------------------------------------------------------------ 6. exportar
    def boards(self) -> list[dict[str, Any]]:
        from tinyml_studio.boards import load_boards

        return load_boards()["boards"]

    def run_export(self, board_id: str, model: str | None = None) -> dict[str, Any]:
        from tinyml_studio.export import export_run

        if self.run_dir is None:
            raise ValueError(tr("primero entrena un modelo", "train a model first"))
        out = self.workdir / "export"
        board = next((b for b in self.boards() if b["id"] == board_id), None)
        usable = (board or {}).get("usable") or {}  # presupuesto utilizable (heurística nuestra)
        self.export = export_run(
            self.run_dir,
            out,
            model,
            board=board_id,
            flash_kb=usable.get("flash_kb"),
            ram_kb=usable.get("ram_kb"),
        )
        self.export_dir = out
        return self.export

    # ------------------------------------------------------------------ 7. entorno
    def env_report(self, python: str | None = None, components: list[str] | None = None) -> str:
        """Informe del doctor (texto) para el Python actual o uno propio (venv/conda)."""
        from tinyml_studio.doctor import doctor, format_report

        py = (python or "").strip().strip('"') or None
        return format_report(doctor(py, components or ["base", "tui"]))

    # ------------------------------------------------------------------ 8. informe (gráficas y PDF)
    def report_data(self):
        """Datos del informe con lo que haya hecho la sesión (entrenamiento obligatorio)."""
        from tinyml_studio.render import ReportData

        if self.train is None:
            raise ValueError(tr("primero entrena un modelo", "train a model first"))
        name = Path(self.paths[0]).name if self.paths else ""
        return ReportData(
            results=self.train,
            split=self.split,
            export=self.export,
            recipe=self.recipe,
            report=self.report,
            dataset_name=name,
        )

    def charts_available(self) -> list[tuple[str, str]]:
        from tinyml_studio.render import available_charts

        return available_charts(self.report_data()) if self.train else []

    def preview(self) -> str:
        from tinyml_studio.render import text_preview

        return text_preview(self.report_data())

    def save_charts(
        self, out_dir: str, formats: list[str], only: list[str] | None = None
    ) -> list[Path]:
        from tinyml_studio.render import save_charts

        return save_charts(self.report_data(), out_dir, formats or ["png"], only or None)

    def make_pdf(self, target: str, charts: list[str] | None = None) -> Path:
        """``target`` puede ser una carpeta (se crea ``informe.pdf``) o un archivo ``.pdf``."""
        from tinyml_studio.render import build_pdf

        p = Path(target).expanduser()
        if p.suffix.lower() != ".pdf":
            p = p / "informe.pdf"
        return build_pdf(self.report_data(), p, charts or None)
