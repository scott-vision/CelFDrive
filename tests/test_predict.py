import numpy as np
import pytest

import predict


@pytest.fixture(autouse=True)
def reset_predict_state():
    original_config = predict.config
    original_model = predict.model
    yield
    predict.config = original_config
    predict.model = original_model


def config_for_tests(**overrides):
    config = {
        "preprocessing": {
            "input_channel": {"mode": "first_channel_if_rgb"},
            "top_clip_percentile": 0.01,
            "normalize_min_max": True,
        },
        "tiling": {
            "enabled": True,
            "tile_size_px": 640,
            "overlap_px": 0,
            "deduplication_tolerance_px": 1.0,
        },
        "coordinate_conversion": {
            "default_z_offset_um": 0.0,
            "llsm": {"invert_y_stage_direction": True},
        },
        "logging": {"enabled": False},
        "plotting": {"enabled": False},
        "model": {"suppress_stdout": False},
        "profile": {
            "highres_script": "highres",
            "highres_comment": "High resolution",
            "name_template": "{class_name} {x} {y} {z}",
            "classes": {
                0: {"name": "prophase", "confidence_threshold": 0.5, "priority_rank": 1},
                1: {"name": "prometaphase", "confidence_threshold": 0.2, "priority_rank": 0},
            },
        },
        "no_detection": {"mode": "end_workflow"},
    }
    for key, value in overrides.items():
        config[key] = value
    return config


def test_load_predict_config_migrates_legacy_profile_and_no_detection(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        """schema_version: 1
profiles:
  sdc:
    classes:
      0:
        name: prophase
        confidence_threshold: 0.2
        priority_rank: 0
no_detection:
  return_original_first_position: false
""",
        encoding="utf-8",
    )

    config = predict.load_predict_config(config_path)

    assert "profiles" not in config
    assert config["profile"]["classes"][0]["name"] == "prophase"
    assert config["no_detection"]["mode"] == "end_workflow"
    assert config["coordinate_conversion"]["mode"] == "stage"
    assert config["tiling"]["deduplication_tolerance_px"] == 1.0


def test_get_logging_directory_resolves_relative_log_directory_from_project_root(tmp_path):
    predict.config = {
        "project": {"repo_path": str(tmp_path)},
        "logging": {"root_dir": "Logging", "use_date_subfolder": False},
    }

    assert predict.get_logging_directory() == tmp_path / "Logging"


@pytest.mark.parametrize(
    ("shape", "expected_offsets", "expected_tile_shapes"),
    [
        ((300, 300), [(0, 0)], [(300, 300)]),
        ((640, 640), [(0, 0)], [(640, 640)]),
        (
            (700, 800),
            [(0, 0), (160, 0), (0, 60), (160, 60)],
            [(640, 640), (640, 640), (640, 640), (640, 640)],
        ),
    ],
)
def test_split_image_returns_complete_tiles_at_all_boundaries(shape, expected_offsets, expected_tile_shapes):
    predict.config = config_for_tests()

    tiles = predict.split_image(np.zeros(shape, dtype=np.uint8))

    assert [(x, y) for _, x, y in tiles] == expected_offsets
    assert [tile.shape for tile, _, _ in tiles] == expected_tile_shapes


def test_split_image_honours_overlap_and_covers_the_final_edge():
    config = config_for_tests()
    config["tiling"] = {
        "enabled": True,
        "tile_size_px": 200,
        "overlap_px": 50,
        "deduplication_tolerance_px": 1.0,
    }
    predict.config = config

    tiles = predict.split_image(np.zeros((320, 430), dtype=np.uint8))

    assert [(x, y) for _, x, y in tiles] == [
        (0, 0), (150, 0), (230, 0),
        (0, 120), (150, 120), (230, 120),
    ]


def test_split_image_rejects_overlap_as_large_as_the_tile():
    config = config_for_tests()
    config["tiling"]["overlap_px"] = 640
    predict.config = config

    with pytest.raises(ValueError, match="overlap_px"):
        predict.split_image(np.zeros((640, 640), dtype=np.uint8))


def test_deduplicate_detections_keeps_highest_confidence_same_class_detection():
    detections = [
        [0, 10, 10, 10, 10, 0.8],
        [0, 10.5, 10.5, 10, 10, 0.9],
        [1, 10.5, 10.5, 10, 10, 0.7],
    ]

    deduplicated = predict.deduplicate_detections(detections, tolerance_px=1.0)

    assert deduplicated == [
        [0, 10.5, 10.5, 10.0, 10.0, 0.9],
        [1, 10.5, 10.5, 10.0, 10.0, 0.7],
    ]


def test_preprocess_image_rejects_unsupported_input_shape():
    predict.config = config_for_tests()

    with pytest.raises(ValueError, match="Unsupported image shape"):
        predict.preprocess_image(np.zeros((2, 3, 4, 5), dtype=np.uint8))


def test_preprocess_image_rejects_invalid_percentile():
    config = config_for_tests()
    config["preprocessing"]["top_clip_percentile"] = 100
    predict.config = config

    with pytest.raises(ValueError, match="top_clip_percentile"):
        predict.preprocess_image(np.zeros((3, 3), dtype=np.uint8))


def test_class_specific_filtering_honours_threshold_and_priority():
    class_info = predict.get_class_info(config_for_tests()["profile"])
    detections = [
        [0, 0, 0, 10, 10, 0.6],
        [1, 0, 0, 10, 10, 0.3],
        [0, 0, 0, 10, 10, 0.4],
    ]

    filtered = predict.filter_and_sort_detections(detections, class_info)

    assert [detection[0] for detection in filtered] == [1, 0]


def test_global_filter_excludes_classes_disabled_by_priority():
    class_info = {
        0: ("enabled", 0.1, 0),
        1: ("disabled", 0.1, -1),
    }
    detections = [
        [1, 2, 3, 0.9, 0, "enabled"],
        [1, 2, 3, 0.9, 1, "disabled"],
    ]

    filtered = predict.global_filter_and_sort_detections(detections, class_info)

    assert filtered == [[1, 2, 3, 0.9, 0, "enabled"]]


def test_process_montage_returns_empty_result_when_global_filter_removes_every_detection(monkeypatch):
    predict.config = config_for_tests()
    monkeypatch.setattr(
        predict,
        "process_single_location",
        lambda *args, **kwargs: [[1.0, 2.0, 3.0, 0.1, 0, "prophase"]],
    )

    result = predict.process_montage(
        [predict.CapturePosition(0, 0, 0)],
        np.zeros((2, 2, 1), dtype=np.uint8),
        {0: ("prophase", 0.5, 0)},
        xy_pixel_spacing_um=1,
        z_offset_um=0,
        coordinate_mode="pixel",
        coordinate_converter=None,
        x_stage_direction=1,
        y_stage_direction=1,
        legacy_llsm_y_inversion=False,
    )

    assert all(len(values) == 0 for values in result)


def test_coordinate_conversion_applies_spacing_and_llsm_y_inversion():
    predict.config = config_for_tests()

    converted = predict.image_cordinates_to_physical(
        x=10,
        y=20,
        im_x=70,
        im_y=40,
        w=100,
        h=100,
        new_z=5,
        xy_pixel_spacing=0.5,
        z_spacing=1,
        x_stage_direction=1,
        y_stage_direction=1,
        z_stage_direction=1,
        LLSM=True,
        class_id=1,
        conf=0.9,
        class_name="prometaphase",
    )

    assert converted == [20.0, 25.0, 5, 0.9, 1, "prometaphase"]


def test_get_target_location_ends_workflow_when_no_detection(monkeypatch):
    predict.config = config_for_tests()
    monkeypatch.setattr(
        predict,
        "process_montage",
        lambda *args, **kwargs: (np.array([]), np.array([]), np.array([]), []),
    )

    output = predict.get_target_location(
        1.0,
        2.0,
        3.0,
        np.zeros((4, 4, 1), dtype=np.uint8),
        1.0,
        1.0,
        1,
        1,
        1,
    )

    count, x_values, y_values, z_values, scripts, names, comments = output
    assert count == 0
    assert x_values.size == y_values.size == z_values.size == 0
    assert scripts == names == comments == []


def mock_process_image(monkeypatch, detections):
    monkeypatch.setattr(predict, "process_image", lambda *args, **kwargs: detections)


def test_named_stage_api_matches_legacy_wrapper(monkeypatch):
    predict.config = config_for_tests()
    mock_process_image(monkeypatch, [[0, 70, 40, 20, 10, 0.9]])
    image = np.zeros((100, 200, 1), dtype=np.uint8)

    named = predict.get_target_locations(
        stage_x=10,
        stage_y=20,
        stage_z=5,
        image=image,
        xy_pixel_spacing_um=0.5,
        x_stage_direction=1,
        y_stage_direction=1,
    )
    legacy = predict.get_target_location(10, 20, 5, image, 0.5, 1, 1, 1, 1)

    assert named[0] == legacy[0]
    assert all(np.array_equal(named[index], legacy[index]) for index in range(1, 4))
    assert named[4:] == legacy[4:]
    assert named[1].tolist() == [-5.0]
    assert named[2].tolist() == [15.0]


def test_pixel_mode_returns_detection_centre(monkeypatch):
    predict.config = config_for_tests()
    mock_process_image(monkeypatch, [[0, 70, 40, 20, 10, 0.9]])

    output = predict.get_target_locations(
        stage_x=10,
        stage_y=20,
        stage_z=5,
        image=np.zeros((100, 200, 1), dtype=np.uint8),
        z_offset_um=2,
        coordinate_mode="pixel",
    )

    assert output[1].tolist() == [80.0]
    assert output[2].tolist() == [45.0]
    assert output[3].tolist() == [7.0]


def test_legacy_llsm_wrapper_preserves_y_direction_inversion(monkeypatch):
    predict.config = config_for_tests()
    mock_process_image(monkeypatch, [[0, 70, 40, 20, 10, 0.9]])
    image = np.zeros((100, 200, 1), dtype=np.uint8)

    regular = predict.get_target_location(10, 20, 5, image, 0.5, 1, 1, 1, 1)
    llsm = predict.get_target_location(10, 20, 5, image, 0.5, 1, 1, 1, 1, LLSM=True)

    assert regular[2].tolist() == [15.0]
    assert llsm[2].tolist() == [25.0]


def test_callable_mode_receives_documented_arguments(monkeypatch):
    predict.config = config_for_tests()
    mock_process_image(monkeypatch, [[0, 70, 40, 20, 10, 0.9]])
    received = {}

    def converter(**kwargs):
        received.update(kwargs)
        return 1, 2, 3

    output = predict.get_target_locations(
        stage_x=10,
        stage_y=20,
        stage_z=5,
        image=np.zeros((100, 200, 1), dtype=np.uint8),
        xy_pixel_spacing_um=0.5,
        z_offset_um=2,
        coordinate_mode="callable",
        coordinate_converter=converter,
    )

    assert output[1].tolist() == [1.0]
    assert output[2].tolist() == [2.0]
    assert output[3].tolist() == [3.0]
    assert received["detection_x_px"] == 80.0
    assert received["detection_y_px"] == 45.0
    assert received["image_width_px"] == 200
    assert received["z_offset_um"] == 2.0


@pytest.mark.parametrize(
    ("coordinate_mode", "coordinate_converter", "message"),
    [
        ("unknown", None, "coordinate_mode"),
        ("callable", None, "coordinate_converter"),
        ("pixel", lambda **kwargs: (1, 2, 3), "coordinate_converter"),
    ],
)
def test_coordinate_mode_validation(coordinate_mode, coordinate_converter, message):
    predict.config = config_for_tests()

    with pytest.raises((TypeError, ValueError), match=message):
        predict.get_target_locations(
            stage_x=1,
            stage_y=2,
            stage_z=3,
            image=np.zeros((4, 4, 1), dtype=np.uint8),
            xy_pixel_spacing_um=1,
            coordinate_mode=coordinate_mode,
            coordinate_converter=coordinate_converter,
        )


def test_callable_mode_rejects_invalid_return(monkeypatch):
    predict.config = config_for_tests()
    mock_process_image(monkeypatch, [[0, 1, 1, 1, 1, 0.9]])

    with pytest.raises(ValueError, match="exactly three"):
        predict.get_target_locations(
            stage_x=1,
            stage_y=2,
            stage_z=3,
            image=np.zeros((4, 4, 1), dtype=np.uint8),
            xy_pixel_spacing_um=1,
            coordinate_mode="callable",
            coordinate_converter=lambda **kwargs: (1, 2),
        )

    with pytest.raises(TypeError, match="finite number"):
        predict.get_target_locations(
            stage_x=1,
            stage_y=2,
            stage_z=3,
            image=np.zeros((4, 4, 1), dtype=np.uint8),
            xy_pixel_spacing_um=1,
            coordinate_mode="callable",
            coordinate_converter=lambda **kwargs: (1, 2, float("nan")),
        )


def test_named_api_validates_montage_shape_spacing_and_directions():
    predict.config = config_for_tests()

    with pytest.raises(ValueError, match="position axis"):
        predict.get_target_locations(
            stage_x=[1, 2],
            stage_y=[3, 4],
            stage_z=[5, 6],
            image=np.zeros((4, 4, 1), dtype=np.uint8),
            xy_pixel_spacing_um=1,
        )
    with pytest.raises(ValueError, match="positive"):
        predict.get_target_locations(
            stage_x=1,
            stage_y=2,
            stage_z=3,
            image=np.zeros((4, 4, 1), dtype=np.uint8),
            xy_pixel_spacing_um=0,
        )
    with pytest.raises(ValueError, match="x_stage_direction"):
        predict.get_target_locations(
            stage_x=1,
            stage_y=2,
            stage_z=3,
            image=np.zeros((4, 4, 1), dtype=np.uint8),
            xy_pixel_spacing_um=1,
            x_stage_direction=0,
        )
