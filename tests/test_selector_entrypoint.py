import runpy
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest


def test_selector_entrypoint_passes_configured_phases(monkeypatch):
    received = []
    selector_module = types.ModuleType("CellClicker.image_selector_multiphase")
    selector_module.load_ui_from_folder = lambda phases: received.append(phases)
    package_module = types.ModuleType("CellClicker")
    package_module.__path__ = []
    monkeypatch.setitem(sys.modules, "CellClicker", package_module)
    monkeypatch.setitem(sys.modules, "CellClicker.image_selector_multiphase", selector_module)

    repository_root = Path(__file__).resolve().parents[1]
    runpy.run_path(repository_root / "run_selector.py", run_name="__main__")

    assert received == [[
        "prophase",
        "earlyprometaphase",
        "prometaphase",
        "metaphase",
        "anaphase",
        "telophase",
    ]]


def test_normalize_image_handles_constant_image():
    pytest.importorskip("cv2")
    from CellClicker.image_selector_multiphase import normalize_image

    normalized = normalize_image(np.full((2, 3), 7, dtype=np.uint16))

    assert normalized.dtype == np.uint8
    assert np.array_equal(normalized, np.zeros((2, 3), dtype=np.uint8))


def test_phase_selector_help_describes_selection_and_navigation():
    from CellClicker.image_selector_multiphase import PHASE_SELECTOR_HELP_TEXT

    assert "select the first thumbnail" in PHASE_SELECTOR_HELP_TEXT
    assert "Left Arrow" in PHASE_SELECTOR_HELP_TEXT
    assert "S or Skip Phase" in PHASE_SELECTOR_HELP_TEXT
    assert "Jump to Next TODO" in PHASE_SELECTOR_HELP_TEXT


def test_selector_hotkey_runs_its_action_once():
    from CellClicker.image_selector_multiphase import _run_selector_hotkey

    called = []
    assert _run_selector_hotkey(lambda: called.append(True)) == "break"
    assert called == [True]


def test_resuming_a_skipped_phase_shows_it_as_skipped(tmp_path):
    from CellClicker.convert_selections_multiphase import parse_xml_for_phases_resume

    root = ET.Element("Data")
    entry = ET.SubElement(root, "DataEntry")
    ET.SubElement(entry, "PathName").text = "frame_t007.png"
    ET.SubElement(entry, "SeriesID").text = "1"
    ET.SubElement(entry, "prophase").text = "-1"
    selection_path = tmp_path / "alice.xml"
    ET.ElementTree(root).write(selection_path)

    selections = parse_xml_for_phases_resume(selection_path, phases=["prophase"])

    assert selections[("frame_t007.png", "1")]["prophase"] == "skipped"




def test_go_back_moves_through_prior_phases_after_next_without_selection(monkeypatch):
    from CellClicker import image_selector_multiphase as selector

    displayed = []
    monkeypatch.setattr(
        selector,
        "display_set",
        lambda image_sets, image_keys, set_index, selected_indices, root, phase, phases, name_xml, window: displayed.append((set_index, phase)),
    )
    phases = ["first", "middle", "last"]
    shared_args = ([[], []], [("one", "1"), ("two", "2")], [{}, {}], object(), phases, "selections.xml")

    selector.go_back(None, shared_args[0], shared_args[1], shared_args[2], shared_args[3], "first", 1, shared_args[4], shared_args[5])
    selector.go_back(None, shared_args[0], shared_args[1], shared_args[2], shared_args[3], "last", 0, shared_args[4], shared_args[5])

    assert displayed == [(0, "last"), (0, "middle")]
