"""Memory behaviour of the region-XML readers used by the phase selector."""

import cv2
import numpy as np
import pytest

from CellClicker.manageXML import append_cell_regions_xml, find_labels_and_extract_rois, get_all_images


FRAME_SIDE = 240


def _project(tmp_path, frames=3, labels=3):
    """Build a small project with one annotated track spanning `labels` frames."""
    images = tmp_path / "images"
    images.mkdir()
    for frame in range(1, frames + 1):
        cv2.imwrite(str(images / f"expt_P01_t{frame:03}.png"), np.full((FRAME_SIDE, FRAME_SIDE, 3), frame, np.uint8))

    xml_path = images / "cell_regions.xml"
    anchor = str(images / f"expt_P01_t{frames:03}.png")
    for class_id in range(labels):
        append_cell_regions_xml(str(xml_path), anchor, class_id, 120, 120, 40, 40, FRAME_SIDE, FRAME_SIDE, 1)
    return xml_path, anchor


def test_extracted_rois_do_not_hold_their_whole_frame_in_memory(tmp_path):
    """A crop kept as a numpy view retains the entire decoded frame behind it.

    The phase selector stores one crop per label, so a view would multiply the
    selector's memory by the ratio of frame area to crop area and exhaust it.
    """
    xml_path, anchor = _project(tmp_path)

    rois = find_labels_and_extract_rois(str(xml_path), anchor, anchor)["1"]

    assert rois, "the track should yield one crop per label"
    for roi in rois:
        assert roi.base is None, "the crop still references the frame it was cut from"
        assert roi.nbytes < FRAME_SIDE * FRAME_SIDE * 3


def test_loading_a_project_keeps_only_the_crops(tmp_path):
    """Total retained memory must scale with crop area, not with frame area."""
    xml_path, _anchor = _project(tmp_path, frames=3, labels=3)

    images_dict = get_all_images(str(xml_path))

    retained = sum(roi.nbytes for rois in images_dict.values() for roi in rois)
    one_frame = FRAME_SIDE * FRAME_SIDE * 3
    assert retained < one_frame, f"{retained} bytes retained is more than a single frame"


def test_extracted_crops_carry_the_pixels_of_their_own_frame(tmp_path):
    """Copying must not change which frame each crop came from."""
    xml_path, anchor = _project(tmp_path, frames=3, labels=3)

    rois = find_labels_and_extract_rois(str(xml_path), anchor, anchor)["1"]

    # Labels step backwards from the anchor and are reversed, so the crops run
    # from the earliest frame to the latest, whose pixels are its frame number.
    assert [int(roi[0, 0, 0]) for roi in rois] == [1, 2, 3]
