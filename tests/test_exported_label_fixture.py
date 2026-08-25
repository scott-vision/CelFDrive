"""Tests for the repository-versioned CellClicker exported-label fixture."""

from pathlib import Path

import pytest

from benchmarks.fixture_paths import resolve_exported_image_path


def _write_label_project(tmp_path, image_bytes):
    """Create the minimum project layout accepted by the exported-label loader."""
    labels_directory = tmp_path / "project" / "user_selections" / "exported_labels"
    labels_directory.mkdir(parents=True)
    (labels_directory / "P0037_t001.txt").write_text(
        "0 0.5 0.5 0.2 0.1\n", encoding="utf-8"
    )
    images_directory = tmp_path / "images"
    images_directory.mkdir()
    (images_directory / "P0037_t001.png").write_bytes(image_bytes)
    return tmp_path / "project", images_directory


def test_exported_labels_resolve_repository_png_images(tmp_path):
    project_directory, images_directory = _write_label_project(tmp_path, b"PNG fixture")

    image_path = resolve_exported_image_path(
        images_directory,
        project_directory / "user_selections" / "exported_labels" / "P0037_t001.txt",
        "0037",
        1,
    )

    assert image_path == images_directory / "P0037_t001.png"


def test_exported_labels_explain_how_to_fetch_lfs_images(tmp_path):
    project_directory, images_directory = _write_label_project(
        tmp_path, b"version https://git-lfs.github.com/spec/v1\n"
    )

    with pytest.raises(FileNotFoundError, match="git lfs pull --include"):
        resolve_exported_image_path(
            images_directory,
            project_directory / "user_selections" / "exported_labels" / "P0037_t001.txt",
            "0037",
            1,
        )
