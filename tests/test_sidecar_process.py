"""Prueba de integración: arranca el sidecar real como proceso hijo."""

import json
import os
import subprocess
import sys
import time

import httpx

TOKEN = "s" * 40


def _spawn(dataset_dir):
    env = {**os.environ, "TINYML_SESSION_TOKEN": TOKEN, "TINYML_ALLOWED_ROOTS": str(dataset_dir)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "tinyml_studio.api"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )
    ready = json.loads(proc.stdout.readline())
    return proc, ready


def test_sidecar_random_port_token_and_stdin_shutdown(dataset_dir):
    proc, ready = _spawn(dataset_dir)
    proc2, ready2 = _spawn(dataset_dir)
    try:
        assert ready["event"] == "ready" and ready["port"] > 1024
        assert ready["port"] != ready2["port"]  # puerto aleatorio por arranque
        base = f"http://127.0.0.1:{ready['port']}"
        assert httpx.get(f"{base}/health").status_code == 401
        ok = httpx.get(f"{base}/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert ok.status_code == 200
        proc.stdin.close()  # el "padre" se va -> el sidecar debe salir solo
        assert proc.wait(timeout=15) == 0
    finally:
        for p in (proc, proc2):
            if p.poll() is None:
                p.kill()
            p.wait()
            for s in (p.stdin, p.stdout):
                if s:
                    s.close()


def test_sidecar_refuses_without_token(dataset_dir):
    env = {k: v for k, v in os.environ.items() if k != "TINYML_SESSION_TOKEN"}
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "tinyml_studio.api"],
        env=env,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert r.returncode == 2 and time.time() - t0 < 30
