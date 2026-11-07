"""Every fragment in data/boards/ must obey docs/BOARDS_SCHEMA.md (evidence for each number)."""

from __future__ import annotations

import pytest
import yaml

from tinyml_studio.boards.loader import FRAGMENTS_DIR, load_boards, validate_board

FILES = sorted(FRAGMENTS_DIR.glob("*.yaml")) if FRAGMENTS_DIR.is_dir() else []


@pytest.mark.parametrize("f", FILES, ids=lambda p: p.name)
def test_fragment_boards_are_valid(f):
    boards = yaml.safe_load(f.read_text(encoding="utf-8"))["boards"]
    assert boards
    for b in boards:
        assert validate_board(b) == [], f"{f.name}:{b.get('id')}"
        if b.get("verified"):
            assert b.get("verification") in ("official_search", "datasheet_read")


def test_ids_unique_and_loader_merges_fragments():
    ids = []
    for f in FILES:
        ids += [b["id"] for b in yaml.safe_load(f.read_text(encoding="utf-8"))["boards"]]
    assert len(ids) == len(set(ids))
    loaded = load_boards()
    assert {b["id"] for b in loaded["boards"]} >= set(ids)


def test_english_board_texts_are_complete():
    """En inglés ninguna placa conserva texto en español (notas, regla, evidencia)."""
    from tinyml_studio.i18n import set_lang

    spanish = set("áéíóúñ¿¡")
    try:
        set_lang("es")
        es = {b["id"]: b for b in load_boards()["boards"]}
        set_lang("en")
        en = {b["id"]: b for b in load_boards()["boards"]}
    finally:
        set_lang(None)
    assert es.keys() == en.keys()
    for bid, b in en.items():
        texts = [b.get("ai_note") or "", (b.get("usable") or {}).get("rule") or ""]
        texts += list(b.get("accelerators") or []) + [e.get("note") or "" for e in b["evidence"]]
        bad = [x[:60] for x in texts if spanish & set(x)]
        assert not bad, (bid, bad)
        assert [e["field"] for e in b["evidence"]] == [e["field"] for e in es[bid]["evidence"]]
        assert b["id"] == bid and b["ai_tier"] == es[bid]["ai_tier"]
    assert es["esp32-s3"]["ai_note"].startswith("ROM 384 KB aparte")  # el original no cambia


def test_esp32_s3_checked_against_datasheets():
    """ESP32-S3: variantes de los datasheets del chip (v2.2) y del módulo WROOM-1 (v1.8), leídos."""
    from tinyml_studio.i18n import set_lang

    b = {x["id"]: x for x in load_boards()["boards"]}["esp32-s3"]
    var = {v["id"]: v for v in b["variants"]}
    assert b["verification"] == "datasheet_read"
    assert var["wroom-1-n4r2"]["psram_kb"] == 2048 and var["wroom-1-n8r2"]["flash_kb"] == 8192
    assert var["s3r16v"]["psram_kb"] == 16384 and var["s3fh4r2"]["flash_kb"] == 4096
    try:
        set_lang("en")
        en = {x["id"]: x for x in load_boards()["boards"]}["esp32-s3"]
    finally:
        set_lang(None)
    notes = [e["note"] for e in en["evidence"] if e["field"] == "psram_kb_max"]
    assert len(notes) == 2 and notes[0] != notes[1]  # cada evidencia con su propia traducción


def test_espressif_family_read_from_datasheets():
    """Familia ESP32 completa verificada con los PDF oficiales (2026-10-28)."""
    boards = {x["id"]: x for x in load_boards()["boards"] if x["family"] == "ESP32"}
    assert set(boards) >= {
        "esp32",
        "esp32-s2",
        "esp32-s3",
        "esp32-c3",
        "esp32-c6",
        "esp32-c5",
        "esp32-p4",
    }
    assert all(b["verification"] == "datasheet_read" for b in boards.values())
    c5 = {v["id"]: v for v in boards["esp32-c5"]["variants"]}
    assert c5["wroom-1-n32r8"]["flash_kb"] == 32768 and c5["wroom-1-n4r2"]["psram_kb"] == 2048
    assert (
        boards["esp32-c5"]["psram_kb_max"] == 32768 and boards["esp32-p4"]["psram_kb_max"] == 65536
    )
    assert {v["id"] for v in boards["esp32"]["variants"]} >= {"wroom-32e-n4r2", "u4wdh"}
    assert {v["id"] for v in boards["esp32-c3"]["variants"]} >= {"mini-1-h8x", "c3fh8x"}


def test_raspberry_pi_family_read_from_official_pdfs():
    """Raspberry Pi: PDF oficiales leídos (2026-10-28); el Zero original no tiene PDF oficial."""
    boards = {x["id"]: x for x in load_boards()["boards"] if x["family"] == "Raspberry Pi"}
    assert boards["rpi-zero"]["verification"] == "official_search"
    others = [b for k, b in boards.items() if k != "rpi-zero"]
    assert len(others) == 7 and all(b["verification"] == "datasheet_read" for b in others)
    assert {v["id"] for v in boards["rpi-4b"]["variants"]} == {"1gb", "2gb", "3gb", "4gb", "8gb"}
    assert boards["rpi-4b"]["clock_mhz"] == 1800 and boards["rpi-pico-2"]["flash_kb"] == 4096


def test_arduino_family_read_from_official_datasheets():
    """Arduino: datasheets de docs.arduino.cc leídos; Nicla Vision no declara su memoria."""
    boards = {x["id"]: x for x in load_boards()["boards"] if x["family"] == "Arduino"}
    assert len(boards) == 11
    assert boards["arduino-nicla-vision"]["verification"] == "official_search"
    assert all(
        b["verification"] == "datasheet_read"
        for k, b in boards.items()
        if k != "arduino-nicla-vision"
    )
    assert all("datasheets/" in b["source"] for b in boards.values())
