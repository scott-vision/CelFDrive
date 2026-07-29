import runpy
import sys
import types
from pathlib import Path


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
