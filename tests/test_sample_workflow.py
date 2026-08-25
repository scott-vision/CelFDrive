"""Tests for the bundled installation smoke test.

The first test covers the comparison helper in isolation. The second runs the
real prediction path against the synthetic fixture, which is the behaviour the
README advertises as the installation check; it skips with an explicit reason
when the inference stack or the Git LFS checkpoint is unavailable.
"""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "examples" / "run_sample_workflow.py"
SAMPLE_DIRECTORY = REPOSITORY_ROOT / "sample_data"
WEIGHTS_PATH = REPOSITORY_ROOT / "Models" / "yolo11x_p99p99_bg05" / "weights" / "best.pt"
CONFIG_PATH = REPOSITORY_ROOT / "celfdrive_predict.yaml"

GIT_LFS_POINTER_PREFIX = b"version https://git-lfs"


def load_sample_workflow_module():
    specification = importlib.util.spec_from_file_location("sample_workflow", SCRIPT_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def unavailable_checkpoint_reason():
    """Return why the bundled checkpoint cannot be loaded, or None if it can."""
    if not WEIGHTS_PATH.is_file():
        return f"{WEIGHTS_PATH} is missing"
    with WEIGHTS_PATH.open("rb") as weights_file:
        if weights_file.read(len(GIT_LFS_POINTER_PREFIX)) == GIT_LFS_POINTER_PREFIX:
            return (
                f"{WEIGHTS_PATH} is a Git LFS pointer rather than a checkpoint. "
                "Run `git lfs pull`, or download the weights from the release archive."
            )
    return None


def test_sample_fixture_matches_expected_empty_output():
    sample_workflow = load_sample_workflow_module()
    expected_path = SAMPLE_DIRECTORY / "expected_detections.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    assert sample_workflow.matches_expected([], expected)
    assert not sample_workflow.matches_expected([[0, 1, 2, 3, 4, 0.5]], expected)


def test_bundled_model_returns_the_expected_detections_for_the_fixture():
    pytest.importorskip("ultralytics", reason="Ultralytics is not installed in this environment.")
    skip_reason = unavailable_checkpoint_reason()
    if skip_reason is not None:
        pytest.skip(skip_reason)

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    import predict

    sample_workflow = load_sample_workflow_module()
    expected = json.loads((SAMPLE_DIRECTORY / "expected_detections.json").read_text(encoding="utf-8"))

    config = predict.load_predict_config(CONFIG_PATH)
    config["project"]["repo_path"] = str(REPOSITORY_ROOT)
    config["model"]["weights_path"] = str(WEIGHTS_PATH)
    config["logging"]["prediction_images"]["enabled"] = False
    predict.configure_prediction_runtime(config)

    image = np.loadtxt(
        SAMPLE_DIRECTORY / "synthetic_blank_image.csv", delimiter=",", dtype=np.uint8
    )
    detections = predict.process_image(
        image, class_info=predict.get_class_info(config["profile"])
    )

    assert len(detections) == len(expected["detections"])
    assert sample_workflow.matches_expected(detections, expected)
