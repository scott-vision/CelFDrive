import pytest
from PIL import Image

from CellClicker import cell_clicker
from CellClicker.image_series import UnsupportedFrameNamingError, series_menu_labels
from CellClicker.cell_clicker import (
    IMAGE_VIEWER_HELP_TEXT,
    ImageProcessor,
    ImageViewer,
    PIL_RESAMPLING,
    centered_window_position,
    display_coordinate_to_roi_coordinate,
)


def test_resampling_filter_works_with_the_installed_pillow_version():
    assert PIL_RESAMPLING.LANCZOS == getattr(Image, "Resampling", Image).LANCZOS


class _FakeRoot:
    """Stand-in for the Tk window, reporting whichever widget has focus."""

    def __init__(self):
        self.focused = None

    def focus_get(self):
        return self.focused


class _FakeVariable:
    """Minimal stand-in for ``tkinter.StringVar`` in headless tests."""

    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


SERIES_A = ["project/images/series_a_t001.png", "project/images/series_a_t002.png"]
SERIES_B = ["project/images/series_b_t001.png", "project/images/series_b_t002.png", "project/images/series_b_t003.png"]


def _viewer_with_two_series():
    """Build an ImageViewer with series loaded but no Tk widgets realised."""
    viewer = ImageViewer.__new__(ImageViewer)
    viewer.series_images = {"series_a": list(SERIES_A), "series_b": list(SERIES_B)}
    viewer.series_labels = series_menu_labels(viewer.series_images)
    viewer.series_var = _FakeVariable()
    viewer.frame_number = _FakeVariable()
    viewer.images = []
    viewer.current_image = 0
    viewer.update_image = lambda: None
    viewer.focus_canvas = lambda: None
    viewer.frame_entry = object()
    viewer.root = _FakeRoot()
    return viewer


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
    viewer.frame_entry, viewer.root = object(), _FakeRoot()
    inspected = []
    viewer.inspect_bbox = lambda: inspected.append(True)

    assert viewer.inspect_hotkey(None) == "break"
    assert inspected == [True]


def test_update_progress_hotkey_refreshes_annotations():
    viewer = ImageViewer.__new__(ImageViewer)
    viewer.frame_entry, viewer.root = object(), _FakeRoot()
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


def test_select_series_switches_the_active_frame_list():
    viewer = _viewer_with_two_series()

    viewer.select_series("series_b")

    # The shared "series" token is dropped from the selector label.
    assert viewer.series_var.get() == "b"
    assert viewer.images == SERIES_B
    assert viewer.current_image == 0
    assert viewer.frame_number.get() == "0"


def test_navigation_cannot_step_out_of_the_selected_series():
    viewer = _viewer_with_two_series()
    viewer.select_series("series_a")

    for _ in range(len(SERIES_B) + 2):
        viewer.next_image()

    assert viewer.current_image == len(SERIES_A) - 1
    assert viewer.images[viewer.current_image] == SERIES_A[-1]
    assert viewer.frame_number.get() == str(len(SERIES_A) - 1)

    for _ in range(len(SERIES_B) + 2):
        viewer.prev_image()

    assert viewer.current_image == 0
    assert viewer.images[viewer.current_image] == SERIES_A[0]
    assert viewer.frame_number.get() == "0"


def test_direct_frame_entry_is_rejected_beyond_the_selected_series(monkeypatch):
    viewer = _viewer_with_two_series()
    viewer.select_series("series_a")
    errors = []
    monkeypatch.setattr(cell_clicker.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))

    viewer.frame_number.set(str(len(SERIES_B) - 1))
    viewer.go_to_frame()

    assert errors, "entering a frame that only exists in another series must be refused"
    assert viewer.current_image == 0

    viewer.frame_number.set("1")
    viewer.go_to_frame()

    assert viewer.images[viewer.current_image] == SERIES_A[1]


def test_unsupported_frame_naming_refuses_the_project_and_explains_why(monkeypatch):
    viewer = _viewer_with_two_series()
    shown = []
    monkeypatch.setattr(cell_clicker.messagebox, "showerror", lambda title, message, **kwargs: shown.append((title, message)))
    viewer.root = object()
    viewer.label = type("_Label", (), {"config": lambda self, **kwargs: shown.append(kwargs)})()

    viewer._report_unsupported_frame_naming(UnsupportedFrameNamingError("rename expt_P01_t1.png to expt_P01_t001.png"))

    assert viewer.series_images == {} and viewer.images == []
    assert shown[-1][0] == "Unsupported Image Names"
    assert "expt_P01_t001.png" in shown[-1][1]
    assert any("timepoint inconsistently" in str(entry) for entry in shown)


@pytest.mark.parametrize("key, expected_index", [("left_arrow", 0), ("right_arrow", 2)])
def test_an_arrow_keypress_moves_exactly_one_frame(key, expected_index):
    """The arrows are bound on both the canvas and the window.

    A handler that does not consume the event runs twice for one keypress, once
    from the focused canvas and again as it propagates to the window.
    """
    viewer = _viewer_with_two_series()
    viewer.select_series("series_b")
    viewer.current_image = 1

    assert getattr(viewer, key)(None) == "break", "the handler must consume the keypress"
    assert viewer.current_image == expected_index


@pytest.mark.parametrize("hotkey", ["left_arrow", "right_arrow", "inspect_hotkey", "update_progress_hotkey"])
def test_window_hotkeys_do_not_fire_while_typing_a_frame_number(hotkey):
    """The hotkeys are bound on the window, so they must stand down while typing."""
    viewer = _viewer_with_two_series()
    viewer.select_series("series_b")
    viewer.current_image = 1
    acted = []
    viewer.inspect_bbox = lambda: acted.append("inspect")
    viewer.update_progress = lambda: acted.append("progress")
    viewer.root.focused = viewer.frame_entry

    assert getattr(viewer, hotkey)(None) is None, "the keypress must be left to the text box"
    assert viewer.current_image == 1
    assert acted == []


def test_window_hotkeys_still_fire_when_the_frame_box_is_not_focused():
    viewer = _viewer_with_two_series()
    viewer.select_series("series_b")
    acted = []
    viewer.inspect_bbox = lambda: acted.append("inspect")
    viewer.update_progress = lambda: acted.append("progress")
    viewer.root.focused = None

    viewer.right_arrow(None)
    viewer.inspect_hotkey(None)
    viewer.update_progress_hotkey(None)

    assert viewer.current_image == 1
    assert acted == ["inspect", "progress"]


def test_a_lost_focus_query_does_not_block_navigation():
    """Tk raises when focus sits outside the application; navigation still works."""
    viewer = _viewer_with_two_series()
    viewer.select_series("series_b")

    def raising_focus_get():
        raise KeyError("focus is on another application")

    viewer.root.focus_get = raising_focus_get

    assert viewer.right_arrow(None) == "break"
    assert viewer.current_image == 1


class _FakeCanvas:
    """Canvas stand-in reporting the size pack gave it, not the image's size."""

    def __init__(self, width, height):
        self.width, self.height = width, height

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height


class _FakeImage:
    def __init__(self, width, height):
        self.width, self.height = width, height


def _viewer_showing(original, displayed, canvas):
    """A viewer whose image is drawn at `displayed` inside a `canvas`-sized area."""
    viewer = ImageViewer.__new__(ImageViewer)
    viewer.original_image = _FakeImage(*original)
    viewer.displayed_size = displayed
    viewer.canvas = _FakeCanvas(*canvas)
    return viewer


def test_displayed_image_size_ignores_letterbox_space_around_the_image():
    """Maximising widens the canvas past the image, which must not change scale."""
    viewer = _viewer_showing(original=(600, 450), displayed=(1200, 900), canvas=(1920, 900))

    assert viewer.displayed_image_size() == (1200, 900)
    assert viewer.image_to_canvas_scale() == (2.0, 2.0)


def test_boxes_keep_their_scale_when_the_window_stops_matching_the_image():
    """The same box must land in the same place however wide the window is."""
    square = _viewer_showing(original=(600, 450), displayed=(1200, 900), canvas=(1200, 900))
    wide = _viewer_showing(original=(600, 450), displayed=(1200, 900), canvas=(1920, 1000))

    assert wide.image_to_canvas_scale() == square.image_to_canvas_scale()


def test_a_box_drawn_on_a_letterboxed_canvas_is_stored_at_the_right_place():
    """Canvas coordinates must convert back to the image pixels they cover."""
    viewer = _viewer_showing(original=(600, 450), displayed=(1200, 900), canvas=(1920, 1000))

    scale_x, scale_y = viewer.canvas_to_image_scale()

    assert (round(1100 * scale_x), round(880 * scale_y)) == (550, 440)


def test_image_and_canvas_scales_are_inverses():
    viewer = _viewer_showing(original=(640, 512), displayed=(1000, 800), canvas=(1600, 800))
    to_canvas = viewer.image_to_canvas_scale()
    to_image = viewer.canvas_to_image_scale()

    assert [round(a * b, 9) for a, b in zip(to_canvas, to_image)] == [1.0, 1.0]


@pytest.mark.parametrize("displayed", [None, (0, 0), (1200, 0)])
def test_the_canvas_size_is_used_until_the_image_has_been_drawn(displayed):
    """Before the first draw there is no image geometry to prefer."""
    viewer = _viewer_showing(original=(600, 450), displayed=displayed, canvas=(800, 600))

    assert viewer.displayed_image_size() == (800, 600)
