"""Base de placas (data/boards.seed.yaml). Los datos están SIN VERIFICAR: la interfaz lo muestra."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from tinyml_studio.i18n import get_lang, tr
from tinyml_studio.resources import resource_path

_CANDIDATES = (
    resource_path("data", "boards.yaml"),
    resource_path("data", "boards.seed.yaml"),
)


def boards_file() -> Path:
    env = os.environ.get("TINYML_BOARDS_FILE")
    if env:
        return Path(env)
    for c in _CANDIDATES:
        if c.is_file():
            return c
    raise FileNotFoundError(
        tr(
            "no se encontró data/boards.yaml ni data/boards.seed.yaml",
            "neither data/boards.yaml nor data/boards.seed.yaml was found",
        )
    )


FRAGMENTS_DIR = resource_path("data", "boards")

AI_TIERS = ("none", "classical_only", "tflm_small", "tflm_medium", "linux_runtime")
NUMERIC_FIELDS = ("clock_mhz", "cores", "sram_kb", "flash_kb", "ram_kb", "psram_kb_max")


def validate_board(b: dict[str, Any]) -> list[str]:
    """Problems of a board entry that claims to be verified (empty list = OK).

    Rule: every numeric spec field present must be backed by an ``evidence`` entry naming that field
    with an https URL; ``retrieved`` (ISO date) must exist; ``ai_tier`` must be one of AI_TIERS.
    """
    errs = []
    for k in ("id", "name", "vendor", "family", "ai_tier"):
        if not b.get(k):
            errs.append(f"missing {k}")
    if b.get("ai_tier") and b["ai_tier"] not in AI_TIERS:
        errs.append(f"ai_tier {b['ai_tier']!r} not in {AI_TIERS}")
    ev = b.get("evidence") or []
    fields = {e.get("field") for e in ev}
    for e in ev:
        if not str(e.get("url", "")).startswith("https://"):
            errs.append(f"evidence for {e.get('field')} without https url")
    for f in NUMERIC_FIELDS:
        if b.get(f) is not None and f not in fields:
            errs.append(f"{f} has no evidence")
    if b.get("verified") and not b.get("retrieved"):
        errs.append("verified without retrieved date")
    return errs


def _fragment_boards() -> list[dict[str, Any]]:
    if not FRAGMENTS_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for f in sorted(FRAGMENTS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        out += [dict(b) for b in raw.get("boards", [])]
    return out


def _english() -> dict[str, Any]:
    f = FRAGMENTS_DIR / "i18n" / "en.yaml"
    return (yaml.safe_load(f.read_text(encoding="utf-8")) or {}) if f.is_file() else {}


def _phrases(text: Any, en: dict[str, Any]) -> Any:
    """Spanish phrases inside names / cpu / evidence values -> English (longest phrase first)."""
    if not isinstance(text, str):
        return text
    for es in sorted(en.get("phrases") or {}, key=len, reverse=True):
        text = text.replace(es, en["phrases"][es])
    return text


def _translate(b: dict[str, Any], en: dict[str, Any]) -> dict[str, Any]:
    """English texts for ``b`` (the family files are in Spanish); anything missing stays as is."""
    for k in ("cpu", "soc"):
        if k in b:
            b[k] = _phrases(b[k], en)
    b["variants"] = [{**v, "name": _phrases(v.get("name"), en)} for v in b.get("variants") or []]
    t = (en.get("boards") or {}).get(b.get("id")) or {}
    if t.get("ai_note"):
        b["ai_note"] = t["ai_note"]
    if t.get("rule") and isinstance(b.get("usable"), dict):
        b["usable"] = {**b["usable"], "rule": t["rule"]}
    acc = en.get("accelerators") or {}
    b["accelerators"] = [acc.get(a, a) for a in b.get("accelerators") or []]
    notes = t.get("evidence") or {}
    used: dict[Any, int] = {}
    evidence = []
    for e in b.get("evidence") or []:
        note = notes.get(e.get("field"))
        if isinstance(
            note, list
        ):  # varias evidencias del mismo campo: una nota por cada una, en orden
            i = used.get(e.get("field"), 0)
            used[e.get("field")] = i + 1
            note = note[i] if i < len(note) else None
        evidence.append({**e, "note": note or e.get("note"), "value": _phrases(e.get("value"), en)})
    b["evidence"] = evidence
    return b


def load_boards() -> dict[str, Any]:
    path = boards_file()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    boards = []
    seen: set[str] = set()
    pieces = [] if os.environ.get("TINYML_BOARDS_FILE") else _fragment_boards()
    en = _english() if get_lang() == "en" else {}
    for b in pieces + [dict(x) for x in raw.get("boards", [])]:
        if b.get("id") in seen:  # verified fragments win over the unverified seed
            continue
        seen.add(b.get("id"))
        b = dict(b)
        b["verified"] = bool(b.get("verified", False))
        b["source"] = b.get("source") or ""
        b.setdefault("name", b.get("id"))
        b.setdefault("family", b.get("vendor"))
        if en:
            b = _translate(b, en)
        b["evidence_count"] = len(b.get("evidence") or [])
        boards.append(b)
    return {
        "file": path.name,
        "all_verified": all(b["verified"] for b in boards) if boards else False,
        "n_verified": sum(1 for b in boards if b["verified"]),
        "warning": ""
        if boards and all(b["verified"] for b in boards)
        else tr(
            "Algunas placas tienen datos SIN VERIFICAR con documentación oficial (se marcan): "
            "revisa la flash y la RAM antes de confiar en el veredicto.",
            "Some boards have data NOT VERIFIED against official documentation (they are flagged): "
            "check flash and RAM before trusting the verdict.",
        ),
        "boards": boards,
    }
