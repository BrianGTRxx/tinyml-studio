"""Punto de entrada del ejecutable congelado (PyInstaller): ``tinyml-core``.

El ejecutable congelado no es un intérprete de Python, así que los subprocesos que el núcleo
lanzaba con ``sys.executable -m ...`` / ``-c ...`` se relanzan como subcomandos de sí mismo
(ver ``tinyml_studio.pyexec``):

- ``serve`` (por defecto, también sin argumentos): el sidecar, mismo protocolo que
  ``python -m tinyml_studio.api``.
- ``run-module <módulo> [args...]``: como ``python -m <módulo> args...``.
- ``run-code <código> [args...]``: como ``python -c <código> args...``.

También funciona sin congelar: ``python -m tinyml_studio.core_main ...``.
"""

from __future__ import annotations

import multiprocessing
import runpy
import sys


def _serve() -> int:
    from tinyml_studio.api.__main__ import main as serve_main

    return serve_main()


def _run_module(module: str, args: list[str]) -> int:
    sys.argv = [module, *args]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        return _exit_code(e.code)
    return 0


def _run_code(code: str, args: list[str]) -> int:
    sys.argv = ["-c", *args]
    scope = {"__name__": "__main__", "__builtins__": __builtins__}
    try:
        exec(compile(code, "<string>", "exec"), scope)  # noqa: S102
    except SystemExit as e:
        return _exit_code(e.code)
    return 0


def _exit_code(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "serve"
    if cmd == "serve":
        return _serve()
    if cmd == "run-module" and len(args) >= 2:
        return _run_module(args[1], args[2:])
    if cmd == "run-code" and len(args) >= 2:
        return _run_code(args[1], args[2:])
    if (
        cmd == "cli"
    ):  # la CLI completa dentro del ejecutable (p. ej. `tinyml-core cli doctor --json`)
        from tinyml_studio.cli import main as cli_main

        return cli_main(args[1:])
    if cmd == "tui":
        from tinyml_studio.tui.app import run

        return run(args[1:])
    print(
        "uso: tinyml-core [serve | cli <args> | tui [ruta...] | run-module <módulo> [args...] "
        "| run-code <código> [args...]]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
