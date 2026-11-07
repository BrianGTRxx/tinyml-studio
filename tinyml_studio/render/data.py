"""Datos que alimentan gráficas e informe (los produce la sesión de la TUI o la CLI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportData:
    """Resultados ya calculados; nada aquí se recalcula al dibujar."""

    results: dict[str, Any]  # salida de train_models (results.json)
    split: dict[str, Any] | None = None  # sizes / classes por partición (tui.session.prepare)
    export: dict[str, Any] | None = None  # salida de export_run
    recipe: dict[str, Any] | None = None
    report: dict[str, Any] | None = None  # salida de inspect_files (alertas, columnas)
    dataset_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return str(self.results.get("task", "classification"))

    @property
    def best(self) -> str | None:
        return (self.results.get("selection") or {}).get("best")

    @property
    def metric(self) -> str:
        return str((self.results.get("selection") or {}).get("metric", "f1_macro"))

    def quality(self, model: str | None = None) -> dict[str, Any]:
        name = model or self.best
        return ((self.results.get("models") or {}).get(name or "", {}) or {}).get("quality") or {}
