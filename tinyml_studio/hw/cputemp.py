"""Temperatura de la CPU en Windows y Linux, sin instalar ni ejecutar ningún controlador.

* Linux: ``psutil.sensors_temperatures()`` (coretemp / k10temp / zenpower...). No hace falta nada más.
* Windows: la "zona térmica" que el propio equipo informa a Windows (clase WMI
  ``Win32_PerfFormattedData_Counters_ThermalZoneInformation``, sin administrador). Es una lectura aproximada y
  puede no coincidir con el sensor del procesador. Si no hay zona, se prueba la zona ACPI
  (``MSAcpi_ThermalZoneTemperature``), que en muchos equipos pide administrador.
"""
# ruff: noqa: E501
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Any

from tinyml_studio.i18n import tr

_TTL_S = 8.0  # PowerShell tarda unos cientos de ms: no se consulta en cada muestra
_cache: dict[str, Any] = {"t": 0.0, "value": None, "source": None}

_PS_ZONE = (
    "$ErrorActionPreference='Stop';"
    "try{Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ThermalZoneInformation | "
    "Select-Object -ExpandProperty HighPrecisionTemperature | ConvertTo-Json -Compress}catch{}"
)
_PS_ACPI = (
    "$ErrorActionPreference='Stop';"
    "try{Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature | "
    "Select-Object -ExpandProperty CurrentTemperature | ConvertTo-Json -Compress}catch{}"
)

def zone_to_celsius(raw: Any) -> float | None:
    """Contador "Thermal Zone Information" de Windows: décimas de kelvin (un número o una lista)."""
    vals = raw if isinstance(raw, list) else [raw]
    out = [v / 10.0 - 273.15 for v in vals if isinstance(v, int | float) and v > 0]
    out = [round(c, 1) for c in out if 15 <= c <= 125]
    return max(out) if out else None


def acpi_to_celsius(raw: Any) -> float | None:
    """MSAcpi devuelve décimas de kelvin (un número o una lista)."""
    vals = raw if isinstance(raw, list) else [raw]
    out = [v / 10.0 - 273.15 for v in vals if isinstance(v, int | float) and v > 0]
    out = [round(c, 1) for c in out if 15 <= c <= 125]
    return max(out) if out else None


def _powershell(script: str) -> Any:
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=6,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = r.stdout.strip()
    if r.returncode != 0 or not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _linux() -> tuple[float | None, str | None]:
    import psutil

    fn = getattr(psutil, "sensors_temperatures", None)
    if fn is None:
        return None, None
    try:
        data = fn()
    except Exception:  # noqa: BLE001 - sensores caprichosos según el equipo
        return None, None
    for key in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "cpu-thermal"):
        vals = [t.current for t in data.get(key, []) if t.current is not None]
        if vals:
            return round(max(vals), 1), key
    vals = [t.current for ts in data.values() for t in ts if t.current is not None]
    return (round(max(vals), 1), "sensors") if vals else (None, None)


def _windows() -> tuple[float | None, str | None]:
    # sin instalar nada ni pedir administrador: la zona térmica que informa el propio Windows (aproximada)
    v = zone_to_celsius(_powershell(_PS_ZONE))
    if v is not None:
        return v, "Thermal zone"
    v = acpi_to_celsius(_powershell(_PS_ACPI))
    return (v, "ACPI") if v is not None else (None, None)


def read_cpu_temp(force: bool = False) -> tuple[float | None, str | None]:
    """(°C, fuente). ``None`` si el equipo no la expone."""
    if os.name != "nt":
        return _linux()
    now = time.monotonic()
    if not force and now - _cache["t"] < _TTL_S:
        return _cache["value"], _cache["source"]
    v, src = _windows()
    _cache.update(t=now, value=v, source=src)
    return v, src


def hint() -> str:
    """Qué pasa cuando no hay lectura."""
    if os.name == "nt":
        return tr(
            "Este equipo no informa ninguna zona térmica a Windows, así que no se puede mostrar la temperatura "
            "de la CPU sin un controlador de hardware (la app no instala ninguno).",
            "This computer reports no thermal zone to Windows, so the CPU temperature cannot be shown without a "
            "hardware driver (the app installs none).",
        )
    return tr(
        "Este sistema no expone la temperatura de la CPU. En Linux instala lm-sensors y carga "
        "los módulos (sudo sensors-detect) si tu placa lo permite.",
        "This system does not expose the CPU temperature. On Linux install lm-sensors and load the "
        "modules (sudo sensors-detect) if your board allows it.",
    )


def approx_note() -> str:
    """Aviso cuando la lectura es la zona térmica de Windows y no el sensor de la CPU."""
    return tr(
        "La temperatura de la CPU es la zona térmica que informa tu equipo a Windows: es aproximada y puede "
        "no coincidir con la del procesador.",
        "The CPU temperature is the thermal zone your computer reports to Windows: it is approximate and may "
        "not match the processor itself.",
    )
