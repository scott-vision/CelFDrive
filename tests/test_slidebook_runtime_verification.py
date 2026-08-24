"""Regression tests for the acquisition-computer runtime verification tool."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY_ROOT / "tools" / "verify_slidebook_runtime.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("verify_slidebook_runtime", TOOL_PATH)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_slidebook_runtime_tool_uses_tracked_notebook_tiff_and_bridge_conversion():
    """The hardware smoke test exercises an actual tutorial TIFF and bridge axes."""
    tool = _load_tool()
    tifffile = pytest.importorskip("tifffile")

    assert tool.EXAMPLE_IMAGE.is_file()
    bridge_spec = importlib.util.spec_from_file_location(
        "slidebook_runtime_bridge_for_test",
        REPOSITORY_ROOT / "SlideBook" / "find_locations_of_interest_montage.py",
    )
    bridge = importlib.util.module_from_spec(bridge_spec)
    bridge_spec.loader.exec_module(bridge)
    image, montage = tool.load_tutorial_montage(tifffile.imread, np, bridge)

    assert image.ndim == 2
    assert image.dtype == np.uint16
    assert montage.shape == image.shape + (1,)
    np.testing.assert_array_equal(montage[:, :, 0], image)
    assert "predict.process_image" in TOOL_PATH.read_text(encoding="utf-8")


def test_windows_environment_uses_pip_for_all_native_scientific_packages():
    """Prevent reintroducing Conda/pip OpenMP DLL mixing on acquisition PCs."""
    environment = (REPOSITORY_ROOT / "Environments" / "environment-gpu-windows.yml").read_text(encoding="utf-8")
    lines = environment.splitlines()

    assert not any(line.strip() == "- conda-forge" for line in lines)
    assert not any(line.strip() == "- opencv-contrib-python" for line in lines)
    pip_dependencies = environment.split("  - pip:\n", maxsplit=1)[1]
    assert "      - numpy" in pip_dependencies
    assert "      - torch" in pip_dependencies
