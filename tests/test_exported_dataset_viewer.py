from pathlib import Path

import pytest
from PIL import Image

from CellClicker.exported_dataset_viewer import (
    box_label_position,
    find_exported_label_pairs,
    read_yolo_labels,
    yolo_box_to_display_coordinates,
)


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (40, 20)).save(path)


def test_find_exported_label_pairs_preserves_image_relative_paths(tmp_path):
    project_dir = tmp_path / "project"
    image_path = project_dir / "images" / "series_a" / "frame_001.png"
    _write_image(image_path)
    label_path = project_dir / "user_selections" / "exported_labels" / "series_a" / "frame_001.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("4 0.5 0.5 0.4 0.6\n", encoding="utf-8")

    assert find_exported_label_pairs(project_dir) == [(str(image_path), str(label_path))]


def test_find_exported_label_pairs_rejects_labels_without_images(tmp_path):
    label_path = tmp_path / "project" / "user_selections" / "exported_labels" / "frame_001.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("4 0.5 0.5 0.4 0.6\n", encoding="utf-8")
    (tmp_path / "project" / "images").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="no matching source image"):
        find_exported_label_pairs(tmp_path / "project")


def test_read_yolo_labels_and_display_coordinates(tmp_path):
    label_path = tmp_path / "labels.txt"
    label_path.write_text("4 0.5 0.25 0.4 0.2\n", encoding="utf-8")

    label = read_yolo_labels(label_path)[0]

    assert label == (4, 0.5, 0.25, 0.4, 0.2)
    assert yolo_box_to_display_coordinates(label, 200, 100) == (60.0, 15.0, 140.0, 35.0)


def test_box_label_position_stays_outside_the_box():
    assert box_label_position(20, 30, 60, 80, 100) == (20, 27, "sw")
    assert box_label_position(20, 5, 60, 20, 100) == (20, 23, "nw")


def test_read_yolo_labels_reports_invalid_normalized_values(tmp_path):
    label_path = tmp_path / "labels.txt"
    label_path.write_text("4 0.5 0.25 1.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="normalized"):
        read_yolo_labels(label_path)
