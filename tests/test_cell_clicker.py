import pytest

from CellClicker.cell_clicker import (
    IMAGE_VIEWER_HELP_TEXT,
    ImageProcessor,
    ImageViewer,
    centered_window_position,
    display_coordinate_to_roi_coordinate,
)


def test_display_coordinate_to_roi_coordinate_reverses_display_scaling():
    assert display_coordinate_to_roi_coordinate(150, 3) == 50
    assert display_coordinate_to_roi_coordinate(151, 3) == 50


def test_display_coordinate_to_roi_coordinate_rejects_invalid_scale():
    with pytest.raises(ValueError, match="greater than zero"):
        display_coordinate_to_roi_coordinate(150, 0)


def test_centered_window_position_centers_child_over_parent():
    assert centered_window_position(100, 200, 800, 600, 300, 200) == (350, 400)


def test_image_viewer_help_covers_the_main_annotation_actions():
    assert "Left Arrow = previous image" in IMAGE_VIEWER_HELP_TEXT
    assert "I = Inspect" in IMAGE_VIEWER_HELP_TEXT
    assert "U = Update Progress" in IMAGE_VIEWER_HELP_TEXT
    assert "F = Finished" in IMAGE_VIEWER_HELP_TEXT
    assert "drag a red box" in IMAGE_VIEWER_HELP_TEXT
    assert "Right-click a green box" in IMAGE_VIEWER_HELP_TEXT


def test_inspect_hotkey_opens_the_current_box_inspector():
    viewer = ImageViewer.__new__(ImageViewer)
    inspected = []
    viewer.inspect_bbox = lambda: inspected.append(True)

    assert viewer.inspect_hotkey(None) == "break"
    assert inspected == [True]


def test_update_progress_hotkey_refreshes_annotations():
    viewer = ImageViewer.__new__(ImageViewer)
    refreshed = []
    viewer.update_progress = lambda: refreshed.append(True)

    assert viewer.update_progress_hotkey(None) == "break"
    assert refreshed == [True]


def test_finish_hotkey_ends_the_mini_clicker_session():
    processor = ImageProcessor.__new__(ImageProcessor)
    finished = []
    processor.end_session = lambda: finished.append(True)

    assert processor.finish_hotkey(None) == "break"
    assert finished == [True]


def test_complete_clicker_session_clears_inspection_and_refreshes_viewer():
    class Canvas:
        def __init__(self):
            self.deleted = []

        def delete(self, item):
            self.deleted.append(item)

    viewer = ImageViewer.__new__(ImageViewer)
    viewer.canvas = Canvas()
    viewer.rect = 42
    viewer.bbox_details = {"x": 1}
    refreshed = []
    viewer.update_progress = lambda: refreshed.append(True)
    viewer.focus_canvas = lambda: None

    viewer.complete_clicker_session()

    assert viewer.canvas.deleted == [42]
    assert viewer.rect is None
    assert viewer.bbox_details is None
    assert refreshed == [True]
