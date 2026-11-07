"""Gráficas e informe PDF generados en el núcleo (sin servidor ni navegador).

La TUI y la CLI usan este módulo para guardar los MISMOS archivos que ofrece la interfaz gráfica:
gráficas en SVG / PNG / JPG y un informe PDF. Las gráficas se describen una vez como una escena de
primitivas (``scene.Scene``) y se dibujan con tres salidas: SVG (texto, sin dependencias), PNG/JPG
(Pillow) y PDF (fpdf2, con las gráficas incrustadas). Pillow y fpdf2 se importan solo al usarlos.
"""

from tinyml_studio.render.charts import available_charts, build_charts
from tinyml_studio.render.report import ReportData, build_pdf, save_charts
from tinyml_studio.render.scene import Scene
from tinyml_studio.render.textviz import text_preview

__all__ = [
    "ReportData",
    "Scene",
    "available_charts",
    "build_charts",
    "build_pdf",
    "save_charts",
    "text_preview",
]
