import copy

import pytest

from run_config_gui import ConfigEditor, validate_prediction_config


class FakeVariable:
    """Minimal variable stand-in for testing visibility without a Tk display."""

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeFrame:
    """Records whether the conditional capture-script field is visible."""

    def __init__(self):
        self.visible = None

    def grid(self):
        self.visible = True

    def grid_remove(self):
        self.visible = False


def valid_config():
    return {
        "project": {"repo_path": "."},
        "model": {
            "weights_path": "Models/yolo11x_p99p99_bg05/weights/best.pt",
            "backend": "ultralytics_yolo",
            "suppress_stdout": True,
        },
        "logging": {
            "enabled": True,
            "root_dir": "Logging",
            "use_date_subfolder": True,
            "date_format": "%Y-%m-%d",
            "experiment_folder": {"prefix": "exp", "digits": 3},
            "output_image": {"prefix": "tmpimg", "digits": 3, "extension": ".png"},
        },
        "preprocessing": {
            "input_channel": {"mode": "first_channel_if_rgb"},
            "top_clip_percentile": 0.01,
            "normalize_min_max": True,
        },
        "inference": {
            "mode": "tiling",
            "sahi": {
                "confidence_threshold": 0.5,
                "slice_size_px": 640,
                "overlap_ratio": 0.25,
                "tile_batch_size": 6,
                "merge_iou_threshold": 0.1,
            },
        },
        "tiling": {
            "enabled": True,
            "tile_size_px": 640,
            "edge_mode": "shift_last_tile_inside_image",
            "overlap_px": 0,
            "deduplication_tolerance_px": 1.0,
        },
        "coordinate_conversion": {
            "mode": "stage",
            "default_z_offset_um": 0.0,
            "merge_tolerance_um": 20.0,
            "stage_direction": {"x": 1, "y": 1, "z": 1},
            "llsm": {"invert_y_stage_direction": True},
        },
        "slidebook": {
            "python_environment": "celfdrive-windows",
            "objective_before_target_search": "",
            "highres_objective": "20x Air",
            "objective_offset_um": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        "no_detection": {
            "mode": "empty_3i_capture_script",
            "empty_3i_capture_script": "donothing",
        },
        "plotting": {
            "enabled": True,
            "cmap": "gray",
            "bbox": {"edge_color": "red", "line_width": 1.0},
            "label": {
                "font_size": 8,
                "text_color": "white",
                "background_color": "black",
                "background_alpha": 0.5,
            },
        },
        "profile": {
            "description": "High-resolution capture profile",
            "highres_script": "floifmHighres",
            "highres_comment": "Highres",
            "name_template": "{class_name} {x} {y} {z}",
            "classes": {
                0: {
                    "name": "prophase",
                    "confidence_threshold": 0.01,
                    "priority_rank": 0,
                },
            },
        },
    }


def test_validate_prediction_config_accepts_a_valid_postscan_profile():
    validate_prediction_config(valid_config())


def test_editor_migrates_missing_inference_settings_to_tiling_sahi_defaults():
    config = valid_config()
    del config["inference"]
    editor = ConfigEditor.__new__(ConfigEditor)

    editor.migrate_config(config)

    assert config["inference"] == {
        "mode": "tiling",
        "sahi": {
            "confidence_threshold": 0.5,
            "slice_size_px": 640,
            "overlap_ratio": 0.25,
            "tile_batch_size": 6,
            "merge_iou_threshold": 0.1,
        },
    }
    assert config["logging"]["timing"] == {"enabled": True}
    assert config["logging"]["prediction_images"] == {"enabled": True}
    assert "enabled" not in config["logging"]
    assert "enabled" not in config["plotting"]


def test_editor_uses_prediction_runtime_defaults_for_new_coordinate_and_tiling_settings():
    config = valid_config()
    del config["coordinate_conversion"]["mode"]
    del config["coordinate_conversion"]["default_z_offset_um"]
    del config["coordinate_conversion"]["merge_tolerance_um"]
    del config["tiling"]["overlap_px"]
    del config["tiling"]["deduplication_tolerance_px"]
    editor = ConfigEditor.__new__(ConfigEditor)

    editor.migrate_config(config)

    assert config["coordinate_conversion"]["mode"] == "stage"
    assert config["coordinate_conversion"]["default_z_offset_um"] == 0.0
    assert config["coordinate_conversion"]["merge_tolerance_um"] == 20.0
    assert config["tiling"]["overlap_px"] == 0
    assert config["tiling"]["deduplication_tolerance_px"] == 1.0


def test_editor_rebuilds_the_image_tab_when_inference_mode_changes():
    editor = ConfigEditor.__new__(ConfigEditor)
    editor.config = valid_config()
    editor.vars = {("inference", "mode"): (FakeVariable("full_image"), str)}
    selected_tabs = []
    editor.rebuild_ui = lambda selected_main_tab=None: selected_tabs.append(selected_main_tab)

    editor.on_inference_mode_changed(None)

    assert editor.config["inference"]["mode"] == "full_image"
    assert selected_tabs == ["Image"]


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda config: config["profile"].update({"highres_script": ""}), "postscan script"),
        (lambda config: config["profile"].update({"classes": {}}), "At least one"),
        (lambda config: config["profile"]["classes"][0].update({"name": ""}), "must have a name"),
        (lambda config: config["profile"]["classes"][0].update({"confidence_threshold": 1.1}), "between 0 and 1"),
        (lambda config: config["profile"]["classes"][0].update({"priority_rank": -2}), "-1 or greater"),
        (lambda config: config["tiling"].update({"overlap_px": 640}), "overlap_px"),
        (lambda config: config["coordinate_conversion"].update({"mode": "callable"}), "API-only"),
        (lambda config: config["coordinate_conversion"].update({"stage_direction": {"x": 1, "y": 1}}), "x, y, and z"),
        (lambda config: config["slidebook"].update({"objective_offset_um": {"x": 1, "y": 1}}), "x, y, and z"),
        (lambda config: config["slidebook"].update({"python_environment": "bad\"name"}), "quotes"),
        (lambda config: config["profile"].update({"name_template": "{unknown}"}), "may only use"),
        (lambda config: config["inference"].update({"mode": "unknown"}), "inference.mode"),
        (lambda config: config["inference"]["sahi"].update({"confidence_threshold": 1.1}), "confidence_threshold"),
        (lambda config: config["inference"]["sahi"].update({"slice_size_px": 0}), "slice_size_px"),
        (lambda config: config["inference"]["sahi"].update({"overlap_ratio": 1.0}), "overlap_ratio"),
        (lambda config: config["inference"]["sahi"].update({"tile_batch_size": 0}), "tile_batch_size"),
        (lambda config: config["inference"]["sahi"].update({"merge_iou_threshold": -0.1}), "merge_iou_threshold"),
    ],
)
def test_validate_prediction_config_rejects_invalid_high_resolution_settings(update, message):
    config = copy.deepcopy(valid_config())
    update(config)

    with pytest.raises(ValueError, match=message):
        validate_prediction_config(config)


@pytest.mark.parametrize(
    ("mode", "script_is_visible"),
    [
        ("end_workflow", False),
        ("empty_3i_capture_script", True),
    ],
)
def test_no_detection_script_visibility_matches_selected_mode(mode, script_is_visible):
    editor = ConfigEditor.__new__(ConfigEditor)
    editor.vars = {("no_detection", "mode"): (FakeVariable(mode), str)}
    editor.no_detection_script_frame = FakeFrame()

    editor.update_no_detection_fields()

    assert editor.no_detection_script_frame.visible is script_is_visible
