# -*- mode: python ; coding: utf-8 -*-
# Núcleo Python de TinyML Studio como UNA carpeta autónoma (onedir) -> dist/tinyml-core/.
# Uso: python packaging/build_core.py [--variant lite|full]   (variable TINYML_CORE_VARIANT)
import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

HERE = Path(SPECPATH).resolve()  # noqa: F821  (SPECPATH lo define PyInstaller)
ROOT = HERE.parent
VARIANT = os.environ.get("TINYML_CORE_VARIANT", "lite")
if VARIANT not in ("lite", "full"):
    raise SystemExit(f"TINYML_CORE_VARIANT debe ser lite o full, no {VARIANT!r}")


def no_tests(name: str) -> bool:
    return ".tests" not in name and not name.endswith(".conftest") and ".testing._private" not in name


# Marcador de variante (lo lee tinyml_studio.environment.variant()).
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)
(BUILD / "tinyml_variant.txt").write_text(VARIANT + "\n", encoding="utf-8")

datas = [
    (str(ROOT / "data"), "data"),  # boards.seed.yaml + data/boards/*.yaml (tinyml_studio.resources)
    (str(BUILD / "tinyml_variant.txt"), "."),
    # La pestaña Entorno de la TUI lee la lista de componentes (tinyml_studio.envmgr.env_dir -> env/).
    (str(ROOT / "packaging" / "env" / "manifest.json"), "env"),
]
binaries = []

# Módulos que se importan dentro de funciones / por cadena: el análisis estático no los ve.
hiddenimports = collect_submodules("tinyml_studio")
for pkg in ("uvicorn", "sklearn", "scipy", "pandas", "joblib", "psutil", "yaml", "pydantic"):
    hiddenimports += collect_submodules(pkg, filter=no_tests)
hiddenimports += ["multiprocessing", "email.mime.text", "_yaml"]
hiddenimports += collect_submodules("fontTools") + ["defusedxml"]  # fpdf2 (PDF con fuentes TTF)
datas += [(str(ROOT / "tinyml_studio" / "render" / "fonts"), "tinyml_studio/render/fonts")]
# Interfaz de terminal (tinyml-core tui): Textual lleva .tcss/datos y Rich carga módulos por nombre.
for pkg in ("textual", "rich", "fpdf", "PIL"):
    _d, _b, _h = collect_all(pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

excludes = [
    "torch", "torchvision", "torchaudio", "jax", "jaxlib", "matplotlib", "IPython", "ipykernel",
    "notebook", "jupyter", "jupyter_client", "jupyter_core", "nbformat", "tkinter", "pytest",
    "plotly", "onnx", "onnxruntime", "sphinx", "docutils", "PIL.ImageTk", "numba", "llvmlite",
    "setuptools", "pkg_resources", "sklearn.tests", "scipy.tests", "pandas.tests",
]  # fmt: skip

if VARIANT == "full":
    # tensorflow lo recoge su hook oficial (excluye include/ y .lib; collect_all metería 430 MB
    # de cabeceras). tflite_micro no tiene hook: collect_all.
    d, b, h = collect_all("tflite_micro")
    datas += d
    binaries += b
    hiddenimports += h
    hiddenimports += ["tensorflow"]

    # Keras 3: los backends torch/jax/openvino no se empaquetan (collect_all los importaría y
    # torch tira el proceso de PyInstaller); solo el backend de TensorFlow.
    def keras_ok(name: str) -> bool:
        return no_tests(name) and not any(
            f".backend.{b}" in name for b in ("torch", "jax", "openvino", "numpy")
        )

    # hook-tensorflow (pyinstaller-hooks-contrib) llama a collect_submodules('keras') SIN filtro y
    # eso importa keras.src.backend.torch -> segfault del subproceso. Se envuelve para filtrar.
    import PyInstaller.utils.hooks as _hooks

    _orig_collect = _hooks.collect_submodules

    def _collect_filtered(package, *args, **kwargs):
        if package == "keras" and "filter" not in kwargs and not args:
            kwargs["filter"] = keras_ok
        return _orig_collect(package, *args, **kwargs)

    _hooks.collect_submodules = _collect_filtered
    hiddenimports += collect_submodules("keras", filter=keras_ok)
    datas += collect_data_files("keras")
    binaries += collect_dynamic_libs("keras")
    # sin CUDA: el hook infiere ~1.2 GB de paquetes nvidia-* (cudnn, nccl...) que no se empaquetan
    excludes += ["tensorboard", "nvidia"]
else:
    excludes += ["tensorflow", "keras", "tflite_micro", "tensorboard", "h5py", "ml_dtypes"]

a = Analysis(  # noqa: F821
    [str(HERE / "core_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts + [("X utf8", None, "OPTION")],  # modo UTF-8: en Windows el ↔ de los resúmenes fallaba con cp1252
    [],
    exclude_binaries=True,  # onedir: nada se desempaqueta en cada arranque
    name="tinyml-core",
    debug=False,
    strip=False,
    upx=False,
    console=True,  # el protocolo usa stdin/stdout
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="tinyml-core",
)
