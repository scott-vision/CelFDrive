"""Tests for the SlideBook raw-montage Python bridge."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


BRIDGE_PATH = Path(__file__).resolve().parents[1] / "SlideBook" / "find_locations_of_interest_montage.py"
SPEC = importlib.util.spec_from_file_location("slidebook_bridge", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def test_raw_slidebook_montage_is_reordered_to_celfdrive_convention():
    raw_image = np.arange(24).reshape(2, 3, 4)

    montage = bridge._to_celfdrive_montage(raw_image)

    assert montage.shape == (3, 4, 2)
    np.testing.assert_array_equal(montage[:, :, 1], raw_image[1])


def test_raw_slidebook_montage_rejects_non_three_dimensional_images():
    with pytest.raises(ValueError, match="position, height, width"):
        bridge._to_celfdrive_montage(np.zeros((3, 4)))


def test_callback_forwards_stage_data_and_applies_configured_objective_offset(monkeypatch):
    received = {}

    def fake_targets(**kwargs):
        received.update(kwargs)
        return 1, np.array([10.0]), np.array([20.0]), np.array([30.0]), ["highres"], ["target"], ["comment"]

    monkeypatch.setattr(bridge.predict, "get_target_locations", fake_targets)
    monkeypatch.setattr(
        bridge.predict,
        "get_config",
        lambda: {"slidebook": {"objective_offset_um": {"x": 1.5, "y": -2.0, "z": 3.0}}},
    )

    targets = bridge.find_locations_of_interest_montage(
        np.zeros((2, 3, 4)), [1, 2], [3, 4], [5, 6], 0.5, 1.0, 1, -1, 1, None, None, "name", None
    )

    assert received["image"].shape == (3, 4, 2)
    assert received["stage_x"] == [1, 2]
    np.testing.assert_allclose(targets[1], [11.5])
    np.testing.assert_allclose(targets[2], [18.0])
    np.testing.assert_allclose(targets[3], [33.0])
