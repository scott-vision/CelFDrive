import runpy
import sys
import types
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
