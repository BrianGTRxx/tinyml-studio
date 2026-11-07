"""Instalador de componentes en Python: plan, validación y ejecución con un uv falso."""

import json
import stat
import sys

import pytest

from tinyml_studio import doctor, envmgr
from tinyml_studio.i18n import set_lang


@pytest.fixture(autouse=True)
def _reset_lang():
    yield
    set_lang(None)


def test_manifest_locks_and_doctor_ranges_agree():
    """Cada lock existe y fija versiones dentro de los rangos compatibles del doctor."""
    m = envmgr.load_manifest()
    ranges = {
        name.lower().replace("_", "-"): (lo, hi)
        for c in doctor.COMPONENTS.values()
        for name, lo, hi in c["packages"]
    }
    for c in m["components"]:
        for plat in c["platforms"]:
            lock = envmgr.env_dir() / "locks" / f"{c['id']}-{plat}.txt"
            assert lock.is_file(), lock
            for line in lock.read_text(encoding="utf-8").splitlines():
                if "==" not in line or line.startswith((" ", "#")):
                    continue
                name, _, rest = line.partition("==")
                ver = rest.split()[0].rstrip("\\").strip()
                lo_hi = ranges.get(name.strip().lower())
                if lo_hi is None:
                    continue
                lo, hi = lo_hi
                v = doctor.parse_version(ver)
                assert doctor._cmp(v, doctor.parse_version(lo)) >= 0, (name, ver, lo)
                if hi:
                    assert doctor._cmp(v, doctor.parse_version(hi)) < 0, (name, ver, hi)


def test_resolve_whitelist_and_platform():
    assert envmgr.resolve([], "linux") == ["base"]
    assert envmgr.resolve(["keras", "tui", "keras"], "linux") == ["base", "keras", "tui"]
    with pytest.raises(ValueError, match="unknown|desconocido"):
        envmgr.resolve(["numpy==1; rm -rf /"], "linux")
    with pytest.raises(ValueError, match="not available|no disponible"):
        envmgr.resolve(["tflm"], "win32")
    assert {c["id"] for c in envmgr.available_components("win32")} == {
        "base",
        "tui",
        "keras",
        "torch",
    }


def test_plan_uses_uv_or_pip_with_hashes_and_wheels_only():
    uv_plan = envmgr.plan(["tui"], python="/p/python", platform="linux", uv="/u/uv")
    assert [c[0] for c in uv_plan] == ["/u/uv", "/u/uv"]
    for c in uv_plan:
        assert {"--require-hashes", "--only-binary"} <= set(c) and c[
            c.index("--python") + 1
        ] == "/p/python"
    pip_plan = envmgr.plan(["tui"], python="/p/python", platform="linux", uv="")
    assert pip_plan[0][:4] == ["/p/python", "-m", "pip", "install"]
    assert "--require-hashes" in pip_plan[0] and pip_plan[0][-1].endswith("base-linux.txt")


@pytest.mark.skipif(sys.platform == "win32", reason="usa un script sh como falso uv")
def test_run_streams_lines_and_reports_failures(tmp_path):
    fake = tmp_path / "fakeuv"
    fake.write_text(
        '#!/bin/sh\necho "instalando $5"\n[ -n "$FAIL" ] && { echo boom; exit 4; }\nexit 0\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    lines: list[str] = []
    envmgr.run(["tui"], lines.append, python="/p/python", platform="linux", uv=str(fake))
    assert any("instalando" in line for line in lines) and lines[-1] in ("Listo.", "Done.")
    import os

    os.environ["FAIL"] = "1"
    try:
        with pytest.raises(RuntimeError, match="boom"):
            envmgr.run([], lines.append, python="/p/python", platform="linux", uv=str(fake))
    finally:
        del os.environ["FAIL"]


def test_frozen_core_is_never_modified(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError, match="no se modifica|not modified"):
        envmgr.run([], lambda _l: None)


def test_wheel_layout_is_found(tmp_path, monkeypatch):
    root = tmp_path / "_data"
    (root / "env").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "env" / "manifest.json").write_text(json.dumps({"components": []}), encoding="utf-8")
    monkeypatch.setattr(envmgr, "resource_root", lambda: root)
    assert envmgr.env_dir() == root / "env"
