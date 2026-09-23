from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile

from CellClicker import tiff_project_import
from CellClicker.clicker_utils import (
    get_previous_image_name,
    get_relative_image_name,
    get_relative_label_name,
)
from CellClicker.image_series import (
    UnsupportedFrameNamingError,
    check_frame_styles,
    discover_image_series,
    frame_marker_and_digits,
    series_menu_labels,
    series_minimum_width,
    split_series_and_frame,
)
from CellClicker.manageXML import append_cell_regions_xml, cell_xml_to_dataframe_absfilenames
from CellClicker.tiff_project_import import create_projects_from_tiff_folder, frame_filename


def _write_tiff(path, data, axes):
    # ``minisblack`` keeps a three-frame grayscale stack from being written as
    # an RGB image, which would misdeclare its axes.
    photometric = {"photometric": "minisblack"} if data.ndim == 3 else {}
    tifffile.imwrite(path, data, metadata={"axes": axes}, **photometric)


def _two_series_source(tmp_path, frames=2):
    """Create a source folder holding two independent two-frame TIFF series."""
    source = tmp_path / "source"
    source.mkdir()
    data = np.arange(frames * 4, dtype=np.uint16).reshape(frames, 2, 2)
    _write_tiff(source / "series_a.tif", data, "TYX")
    _write_tiff(source / "series_b.tif", data, "TYX")
    return source


def test_imports_tyx_tiff_as_flat_normalized_timepoints(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiff(source / "position 1.tif", np.array([[[0, 1], [2, 3]], [[10, 20], [30, 40]]], dtype=np.uint16), "TYX")

    result = create_projects_from_tiff_folder(source, tmp_path / "project")

    images = tmp_path / "project" / "images"
    assert result["series"] == 1
    assert result["frames"] == 2
    assert (images / "cell_regions.xml").is_file()
    assert sorted(path.name for path in images.iterdir()) == ["cell_regions.xml", "position_1_t001.png", "position_1_t002.png"]
    assert np.array_equal(np.asarray(Image.open(images / "position_1_t001.png")), [[0, 85], [170, 255]])


def test_import_never_creates_series_subdirectories(tmp_path):
    """The canonical project layout is a flat ``images/`` directory."""
    source = _two_series_source(tmp_path)

    create_projects_from_tiff_folder(source, tmp_path / "project")

    images_dir = tmp_path / "project" / "images"
    assert not (images_dir / "series_a").exists()
    assert not (images_dir / "series_b").exists()
    assert not any(path.is_dir() for path in images_dir.iterdir())
    assert sorted(path.name for path in images_dir.glob("*.png")) == [
        "series_a_t001.png", "series_a_t002.png", "series_b_t001.png", "series_b_t002.png",
    ]


def test_import_selects_channel_and_max_projects_z(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = np.zeros((2, 2, 3, 2, 2), dtype=np.uint16)
    data[:, 1, :, :, :] = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 3], [4, 5]]], [[[10, 20], [30, 40]], [[50, 60], [70, 80]], [[20, 30], [40, 50]]]])
    _write_tiff(source / "P1.tiff", data, "TCZYX")

    create_projects_from_tiff_folder(source, tmp_path / "project", channel_index=1)

    image = np.asarray(Image.open(tmp_path / "project" / "images" / "P1_t002.png"))
    assert np.array_equal(image, [[0, 85], [170, 255]])


def test_import_max_projects_tzyx_tiff(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint16)
    _write_tiff(source / "P1.tif", data, "TZYX")

    create_projects_from_tiff_folder(source, tmp_path / "project")

    image = np.asarray(Image.open(tmp_path / "project" / "images" / "P1_t001.png"))
    assert np.array_equal(image, [[0, 85], [170, 255]])


def test_import_can_map_a_z_dimension_to_chronological_frames(tmp_path):
    """Users can override metadata when an archive stored frames along Z."""
    source = tmp_path / "source"
    source.mkdir()
    data = np.array([[[1, 2], [3, 4]], [[10, 20], [30, 40]], [[5, 6], [7, 8]]], dtype=np.uint16)
    _write_tiff(source / "archive_export.tif", data, "ZYX")

    result = create_projects_from_tiff_folder(source, tmp_path / "project", time_axis=0)

    images = tmp_path / "project" / "images"
    assert result["frames"] == 3
    assert sorted(path.name for path in images.glob("*.png")) == [
        "archive_export_t001.png", "archive_export_t002.png", "archive_export_t003.png",
    ]
    assert np.array_equal(np.asarray(Image.open(images / "archive_export_t002.png")), [[0, 85], [170, 255]])


def test_import_requires_mapping_of_non_singleton_dimensions(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiff(source / "unlabelled.tif", np.ones((2, 3, 2, 2), dtype=np.uint16), "QRYX")

    with pytest.raises(ValueError, match="unassigned dimension 1"):
        create_projects_from_tiff_folder(source, tmp_path / "project")
    assert not (tmp_path / "project").exists()

    result = create_projects_from_tiff_folder(
        source, tmp_path / "project", time_axis=0, channel_axis=1, channel_index=2,
    )
    assert result["frames"] == 2


def test_import_accepts_xy_spatial_order(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiff(source / "xy.tif", np.array([[1, 2], [3, 4]], dtype=np.uint16), "XY")

    create_projects_from_tiff_folder(source, tmp_path / "project")

    image = np.asarray(Image.open(tmp_path / "project" / "images" / "xy_t001.png"))
    assert np.array_equal(image, [[0, 170], [85, 255]])


def test_import_uses_inferred_trailing_spatial_axes_without_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = np.arange(8, dtype=np.uint16).reshape(2, 2, 2)
    tifffile.imwrite(source / "unlabelled.tif", data, photometric="minisblack")

    description = tiff_project_import.describe_tiff_axes(source)
    result = create_projects_from_tiff_folder(source, tmp_path / "project")

    assert description["time_axis"] == 0
    assert result["frames"] == 2


def test_import_separate_projects_are_each_flat(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiff(source / "a.tif", np.ones((2, 2), dtype=np.uint16), "YX")
    _write_tiff(source / "b.tif", np.ones((2, 2), dtype=np.uint16), "YX")

    output = tmp_path / "projects"
    result = create_projects_from_tiff_folder(source, output, separate_projects=True)

    assert result["series"] == 2
    assert (output / "a" / "images" / "a_t001.png").is_file()
    assert (output / "b" / "images" / "b_t001.png").is_file()
    assert not (output / "a" / "images" / "a").exists()


def test_import_refuses_to_overwrite_an_existing_output(tmp_path):
    source = _two_series_source(tmp_path)
    output = tmp_path / "project"
    create_projects_from_tiff_folder(source, output)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_projects_from_tiff_folder(source, output)


def test_import_rejects_invalid_channel_without_creating_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiff(source / "a.tif", np.ones((1, 2, 2, 2), dtype=np.uint16), "TZYX")

    with pytest.raises(ValueError, match="not available"):
        create_projects_from_tiff_folder(source, tmp_path / "project", channel_index=1)
    assert not (tmp_path / "project").exists()


def test_failed_conversion_leaves_no_partial_project(tmp_path, monkeypatch):
    """A failure part-way through publishes nothing and removes the staging tree."""
    source = _two_series_source(tmp_path)
    original = tiff_project_import.preprocess_image
    calls = {"count": 0}

    def failing_preprocess(frame):
        calls["count"] += 1
        if calls["count"] > 2:
            raise ValueError("synthetic conversion failure")
        return original(frame)

    monkeypatch.setattr(tiff_project_import, "preprocess_image", failing_preprocess)

    with pytest.raises(ValueError, match="synthetic conversion failure"):
        create_projects_from_tiff_folder(source, tmp_path / "project")

    assert not (tmp_path / "project").exists()
    assert not list(tmp_path.glob("celfdrive-tiff-import-*"))


def test_colliding_tiff_names_get_distinct_frame_prefixes(tmp_path):
    """Two stems that sanitize identically must not overwrite each other's frames."""
    source = tmp_path / "source"
    source.mkdir()
    _write_tiff(source / "pos 1.tif", np.full((2, 2), 1, dtype=np.uint16), "YX")
    _write_tiff(source / "pos+1.tif", np.full((2, 2), 2, dtype=np.uint16), "YX")

    result = create_projects_from_tiff_folder(source, tmp_path / "project")

    images_dir = tmp_path / "project" / "images"
    assert result["frames"] == 2
    assert sorted(path.name for path in images_dir.glob("*.png")) == ["pos_1_2_t001.png", "pos_1_t001.png"]
    assert list(discover_image_series(images_dir)) == ["pos_1", "pos_1_2"]


def test_frame_numbers_are_padded_to_sort_chronologically():
    assert frame_filename("P01", 7) == "P01_t007.png"
    assert frame_filename("P01", 7, frame_count=12) == "P01_t007.png"
    assert frame_filename("P01", 7, frame_count=1200) == "P01_t0007.png"
    assert sorted(frame_filename("P01", n, 1200) for n in (2, 10, 1100)) == ["P01_t0002.png", "P01_t0010.png", "P01_t1100.png"]


def test_wide_frame_numbers_still_step_backwards_within_their_series():
    assert get_previous_image_name("P01_t0010.png") == "P01_t0009.png"
    assert get_relative_image_name("P01_t1100.png", 100) == "P01_t1000.png"


def test_discover_image_series_groups_flat_filenames_by_series(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("series_b_t010.png", "series_b_t002.png", "series_a_t001.png", "cell_regions.xml"):
        (images / name).write_bytes(b"data")

    discovered = discover_image_series(images)

    assert list(discovered) == ["series_a", "series_b"]
    assert [Path(path).name for path in discovered["series_b"]] == ["series_b_t002.png", "series_b_t010.png"]
    assert [Path(path).name for path in discovered["series_a"]] == ["series_a_t001.png"]


def test_discovered_series_never_share_frames(tmp_path):
    """Selecting a series must expose that series' frames and nothing else."""
    source = _two_series_source(tmp_path, frames=3)
    create_projects_from_tiff_folder(source, tmp_path / "project")

    discovered = discover_image_series(tmp_path / "project" / "images")

    assert list(discovered) == ["series_a", "series_b"]
    assert all(len(paths) == 3 for paths in discovered.values())
    assert not set(discovered["series_a"]) & set(discovered["series_b"])
    for series, paths in discovered.items():
        assert all(Path(path).name.startswith(series + "_t") for path in paths)


def test_legacy_flat_project_loads_as_a_single_series(tmp_path):
    """An ordinary pre-existing project keeps behaving as one flat series."""
    images = tmp_path / "images"
    images.mkdir()
    for timepoint in (3, 1, 2):
        (images / f"20241022_MC225_slide1_max_P01_t{timepoint:03}.png").write_bytes(b"data")

    discovered = discover_image_series(images)

    assert list(discovered) == ["20241022_MC225_slide1_max_P01"]
    assert [Path(path).name for path in discovered["20241022_MC225_slide1_max_P01"]] == [
        "20241022_MC225_slide1_max_P01_t001.png",
        "20241022_MC225_slide1_max_P01_t002.png",
        "20241022_MC225_slide1_max_P01_t003.png",
    ]


def test_images_without_a_timepoint_convention_form_one_default_series(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("frame_b.png", "frame_a.png"):
        (images / name).write_bytes(b"data")

    discovered = discover_image_series(images)

    assert list(discovered) == ["default"]
    assert [Path(path).name for path in discovered["default"]] == ["frame_a.png", "frame_b.png"]


def test_nested_projects_from_the_superseded_importer_remain_readable(tmp_path):
    """Projects generated before the flat layout still load, by directory."""
    images = tmp_path / "images"
    for name in ("position_b/t010.png", "position_b/t002.png", "position_a/t001.png"):
        path = images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")

    discovered = discover_image_series(images)

    assert list(discovered) == ["position_a", "position_b"]
    assert [Path(path).name for path in discovered["position_b"]] == ["t002.png", "t010.png"]


def test_split_series_and_frame_reads_the_project_naming_convention():
    assert split_series_and_frame("P0037_t001.png") == ("P0037", 1)
    assert split_series_and_frame("20220329_4_3_P1 - 1t027.png") == ("20220329_4_3_P1 - 1", 27)
    assert split_series_and_frame("mc191_series4_MAX_t0012.png") == ("mc191_series4_MAX", 12)
    assert split_series_and_frame("t001.png") == ("default", 1)
    assert split_series_and_frame("overview.png") == (None, None)


def test_annotations_reference_flat_paths_and_reload_within_one_series(tmp_path):
    """XML anchors walk back through flat filenames without crossing series."""
    source = _two_series_source(tmp_path, frames=3)
    create_projects_from_tiff_folder(source, tmp_path / "project")
    images_dir = tmp_path / "project" / "images"
    xml_path = images_dir / "cell_regions.xml"
    anchor = str(images_dir / "series_a_t003.png")

    for class_id in range(3):
        append_cell_regions_xml(str(xml_path), anchor, class_id, 1.0, 1.0, 2.0, 2.0, 2, 2, 1)

    frame = cell_xml_to_dataframe_absfilenames(str(xml_path))
    referenced = [Path(name).name for name in frame["PathName"]]

    assert referenced == ["series_a_t003.png", "series_a_t002.png", "series_a_t001.png"]
    assert all((images_dir / name).is_file() for name in referenced)
    assert not any("series_b" in name for name in referenced)


def _flat_project(tmp_path, names):
    images = tmp_path / "images"
    images.mkdir()
    for name in names:
        (images / name).write_bytes(b"data")
    return images


@pytest.mark.parametrize("position", ["P01", "p01", "P1", "p1", "Pos1", ""])
def test_series_grouping_does_not_depend_on_the_position_token_style(tmp_path, position):
    """Acquisitions spell positions inconsistently, and some omit them entirely."""
    prefix = f"20240520_C2_MC191_{position}" if position else "20240520_C2_MC191"
    images = _flat_project(tmp_path, [f"{prefix}_t{n:03}.png" for n in (2, 1)])

    discovered = discover_image_series(images)

    assert list(discovered) == [prefix]
    assert [Path(path).name for path in discovered[prefix]] == [f"{prefix}_t001.png", f"{prefix}_t002.png"]


def test_series_are_ordered_numerically_for_unpadded_positions(tmp_path):
    """``p2`` must precede ``p10`` in the selector rather than sorting as text."""
    images = _flat_project(tmp_path, [f"expt_p{position}_t001.png" for position in (10, 2, 1, 11)])

    assert list(discover_image_series(images)) == ["expt_p1", "expt_p2", "expt_p10", "expt_p11"]


def test_padded_and_unpadded_positions_stay_separate_series(tmp_path):
    images = _flat_project(tmp_path, ["expt_P01_t001.png", "expt_P1_t001.png", "expt_t001.png"])

    assert list(discover_image_series(images)) == ["expt", "expt_P01", "expt_P1"]


def test_series_menu_labels_drop_the_shared_experiment_prefix():
    labels = series_menu_labels(["20240520_C2_MC191_P01", "20240520_C2_MC191_P02", "20240520_C2_MC191_P10"])

    assert labels == {
        "20240520_C2_MC191_P01": "P01",
        "20240520_C2_MC191_P02": "P02",
        "20240520_C2_MC191_P10": "P10",
    }


@pytest.mark.parametrize("names", [
    ["20240520_C2_MC191_P01"],                     # a single series needs no shortening
    ["position_a", "capture_b"],                   # nothing shared to drop
    ["expt", "expt_P02"],                          # one name is a prefix of the other
])
def test_series_menu_labels_fall_back_to_full_names_when_shortening_is_unsafe(names):
    assert series_menu_labels(names) == {name: name for name in names}


@pytest.mark.parametrize("names", [
    ["expt_P01_t001.png", "expt_P01_t002.png"],    # the canonical form
    ["expt_P01_t1.png", "expt_P01_t2.png"],        # unpadded
    ["expt_P01_T01.png", "expt_P01_T02.png"],      # upper case
    ["expt_P01_t0001.png", "expt_P01_t0002.png"],  # wide padding
])
def test_reading_accepts_any_consistent_timepoint_style(tmp_path, names):
    """Loading is permissive about letter case and padding width."""
    images = _flat_project(tmp_path, names)

    discovered = discover_image_series(images)

    assert list(discovered) == ["expt_P01"]
    assert [Path(path).name for path in discovered["expt_P01"]] == names


@pytest.mark.parametrize("template, minimum", [
    ("expt_P01_t{}.png", 1),        # unpadded, widening as it counts
    ("expt_P01_t{:02}.png", 2),     # padded to two, widening past t99
    ("expt_P01_t{:03}.png", 3),     # the canonical form
    ("expt_P01_T{}.png", 1),        # unpadded and upper case
])
def test_a_series_reads_and_steps_back_whatever_its_padding(tmp_path, template, minimum):
    """``t1 ... t200`` must load and step back as readily as ``t001 ... t200``."""
    names = [template.format(frame) for frame in range(1, 201)]
    images = _flat_project(tmp_path, names)

    assert series_minimum_width(names) == minimum
    frames = discover_image_series(images)["expt_P01"]
    assert [Path(path).name for path in frames] == names, "frames must be in chronological order"

    # Every step back, including across each width change, must name a real file.
    for frame in (200, 101, 100, 11, 10, 2):
        earlier = get_previous_image_name(str(images / template.format(frame)))
        assert Path(earlier).name == template.format(frame - 1)
        assert Path(earlier).is_file(), f"{earlier} does not exist"

    assert get_previous_image_name(str(images / template.format(1))) is None
    jumped = get_relative_image_name(str(images / template.format(200)), 150)
    assert Path(jumped).name == template.format(50) and Path(jumped).is_file()


@pytest.mark.parametrize("name", ["expt_slot12.png", "expt_point7.png", "expt_slot123.png"])
def test_a_word_ending_in_t_and_digits_is_not_a_timepoint(name):
    """``slot12`` must not be read as series ``slo`` at frame 12."""
    assert split_series_and_frame(name) == (None, None)


def test_a_series_padding_inconsistently_fails_loudly(tmp_path):
    """``t9`` beside ``t010`` follows no single padding rule, so refuse it."""
    images = _flat_project(tmp_path, ["expt_P01_t9.png", "expt_P01_t010.png"])

    with pytest.raises(UnsupportedFrameNamingError) as error:
        discover_image_series(images)

    message = str(error.value)
    assert "expt_P01" in message
    assert "expt_P01_t010.png" in message
    assert "same minimum width" in message


def test_mixed_timepoint_letter_case_in_one_series_fails_loudly(tmp_path):
    images = _flat_project(tmp_path, ["expt_P01_t001.png", "expt_P01_T002.png"])

    with pytest.raises(UnsupportedFrameNamingError, match="both `T` and `t`"):
        discover_image_series(images)


def test_different_series_may_use_different_timepoint_styles(tmp_path):
    """The rule is per series; separate positions are rewritten independently."""
    images = _flat_project(tmp_path, ["expt_P01_t1.png", "expt_P01_t2.png", "expt_P02_t001.png"])

    discovered = discover_image_series(images)

    assert list(discovered) == ["expt_P01", "expt_P02"]


def test_images_without_timepoints_never_trigger_the_consistency_check(tmp_path):
    images = _flat_project(tmp_path, ["overview.png", "expt_P01_t001.png", "expt_P01_t002.png"])

    assert list(discover_image_series(images)) == ["default", "expt_P01"]


def test_frame_marker_and_digits_reports_what_was_written():
    assert frame_marker_and_digits("expt_P01_t001.png") == ("t", "001")
    assert frame_marker_and_digits("expt_P01_T27.png") == ("T", "27")
    assert frame_marker_and_digits("overview.png") is None


def test_stepping_back_preserves_the_style_it_read():
    """Without a series on disk to consult, the written width is preserved."""
    assert get_previous_image_name("expt_P01_T027.png") == "expt_P01_T026.png"
    assert get_previous_image_name("expt_P01_t27.png") == "expt_P01_t26.png"
    assert get_relative_image_name("expt_P01_t0100.png", 5) == "expt_P01_t0095.png"
    assert get_relative_label_name("expt_P01_T027.txt", 7) == "expt_P01_T020.txt"


def test_stepping_back_off_the_start_of_a_series_is_still_quiet():
    assert get_previous_image_name("expt_P01_t001.png") is None
    assert get_relative_image_name("expt_P01_t002.png", 5) is None
    assert get_previous_image_name("overview.png") is None
    assert get_previous_image_name("expt_slot12.png") is None


def test_an_imported_project_uses_one_consistent_style(tmp_path):
    source = _two_series_source(tmp_path, frames=3)
    create_projects_from_tiff_folder(source, tmp_path / "project")

    for series, paths in discover_image_series(tmp_path / "project" / "images").items():
        names = [Path(path).name for path in paths]
        check_frame_styles(series, names)
        assert {frame_marker_and_digits(name)[0] for name in names} == {"t"}
        assert series_minimum_width(names) == 3
