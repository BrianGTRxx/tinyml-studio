"""Arranque del sidecar: ``python -m tinyml_studio.api``.

Protocolo con Electron:
- El token llega por la variable de entorno ``TINYML_SESSION_TOKEN`` (no por argv,
  que es visible en la lista de procesos); se elimina del entorno al leerlo.
- Se enlaza ``127.0.0.1`` puerto 0 (aleatorio, elegido por el SO) y el socket ya
  enlazado se entrega a uvicorn (sin carreras de puerto).
- Primera línea de stdout: ``{"event":"ready","port":N,"pid":P}``.
- Se apaga solo si stdin se cierra (el padre murió o lo cerró): sin procesos huérfanos.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time

import uvicorn

from tinyml_studio.i18n import tr

from .app import create_app


def _watch_parent(server: uvicorn.Server) -> None:
    try:
        while sys.stdin.buffer.read(4096):
            pass
    except (OSError, ValueError):
        pass
    server.should_exit = True


def main() -> int:
    token = os.environ.pop("TINYML_SESSION_TOKEN", "")
    roots = [r for r in os.environ.get("TINYML_ALLOWED_ROOTS", "").split(os.pathsep) if r]
    if not token:
        print(tr("falta TINYML_SESSION_TOKEN", "TINYML_SESSION_TOKEN is missing"), file=sys.stderr)
        return 2
    app = create_app(
        token,
        roots,
        os.environ.get("TINYML_WORKDIR") or None,
        # solo para pruebas del empaquetado (trabajo "_sleep"); Electron no lo define
        enable_test_jobs=os.environ.get("TINYML_ENABLE_TEST_JOBS") == "1",
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    if os.environ.get("TINYML_WATCH_STDIN", "1") == "1":
        threading.Thread(target=_watch_parent, args=(server,), daemon=True).start()

    def announce() -> None:
        # Solo cuando uvicorn ya acepta conexiones (si no, el cliente recibe ECONNREFUSED).
        while not server.started and not server.should_exit:
            time.sleep(0.02)
        if server.started:
            print(json.dumps({"event": "ready", "port": port, "pid": os.getpid()}), flush=True)

    threading.Thread(target=announce, daemon=True).start()
    server.run(sockets=[sock])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
