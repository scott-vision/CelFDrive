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
    source = TOOL_PATH.read_text(encoding="utf-8")
    assert "predict.process_image" in source
    assert '"--device"' in source


def test_windows_environments_use_one_conda_forge_native_stack():
    """Prevent reintroducing Conda/pip OpenMP DLL mixing on acquisition PCs."""
    gpu_environment = (REPOSITORY_ROOT / "Environments" / "environment-gpu-windows.yml").read_text(encoding="utf-8")
    cpu_environment = (REPOSITORY_ROOT / "Environments" / "environment-cpu-windows.yml").read_text(encoding="utf-8")

    for environment in (gpu_environment, cpu_environment):
        assert "  - conda-forge" in environment
        assert "  - nodefaults" in environment
        assert "  - opencv" in environment
        assert "  - pytorch-" in environment
        assert "  - --no-deps" in environment
        assert "opencv-python" not in environment
        assert "  - torch\n" not in environment

    assert "  - pytorch-gpu" in gpu_environment
    assert "  - pytorch-cpu" in cpu_environment


def test_windows_venv_creator_selects_cpu_or_official_cuda_pytorch_wheels():
    """The pip alternative is isolated and selects Torch before app packages."""
    creator = (REPOSITORY_ROOT / "tools" / "create_windows_venv.ps1").read_text(encoding="utf-8")

    assert "[ValidateSet(\"cpu\", \"gpu\")]" in creator
    assert "-m\" \"venv" in creator
    assert "download.pytorch.org/whl/$CudaWheel" in creator
    assert "requirements-windows-venv.txt" in creator
    assert '".venv-celfdrive-$Device"' in creator
