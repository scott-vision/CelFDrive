import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from CellClicker.manageXML import (
    get_next_series_id,
    prepare_series_extension,
    remove_entry_from_xml,
)
from CellClicker.project_reconciliation import delete_track_from_project, reconcile_tracking_records, tracking_is_current
from CellClicker.tracking_xml import read_tracking_xml, write_tracking_xml
from CellClicker.user_xml import update_xml_multiclass
from CellClicker.workflow_state import raw_track_revisions, selection_fingerprint, stale_selection_report
from CellClicker.duplicate_tracks import find_near_duplicate_tracks
from CellClicker.convert_selections_multiphase import calculate_median_handling_negatives
from CellClicker.phase_settings import DEFAULT_PHASES, load_phases, save_phases


PHASES = ["prophase", "earlyprometaphase", "prometaphase", "metaphase", "anaphase", "telophase"]


def _raw_xml(path: Path, anchor: str, series_ids=(1,)):
    root = ET.Element("annotations")
    parent = ET.SubElement(root, "path")
    ET.SubElement(parent, "name").text = anchor
    for series_id in series_ids:
        series = ET.SubElement(parent, "series", id=str(series_id), revision="0")
        for class_id in (0, 1):
            label = ET.SubElement(series, "label")
            ET.SubElement(label, "class_id").text = str(class_id)
            for name, value in (("x_center", "0.5"), ("y_center", "0.5"), ("width", "0.2"), ("height", "0.2")):
                ET.SubElement(label, name).text = value
    ET.ElementTree(root).write(path)


def _selection(path, anchor, series_id, revision=0):
    root = ET.Element("Data")
    ET.ElementTree(root).write(path)
    update_xml_multiclass(anchor, series_id, {phase: -1 for phase in PHASES}, path, PHASES, revision)


def _point(source_class_id, box_type="original"):
    return {
        "timepoint_index": source_class_id,
        "frame_index": source_class_id,
        "image_path": f"frame_{source_class_id}.png",
        "class_id": 0,
        "phase_name": "prophase",
        "source_class_id": source_class_id,
        "preferred_box_type": box_type,
        "boxes": [{"box_type": box_type, "format": "yolo_xywh_norm", "x_center": .5, "y_center": .5, "width": .2, "height": .2, "source": "test"}],
    }


def test_extension_revision_and_stable_series_ids(tmp_path):
    raw = tmp_path / "cell_reigons.xml"
    anchor = str(tmp_path / "images" / "series_t003.png")
    _raw_xml(raw, anchor, (1, 3))

    assert prepare_series_extension(raw, anchor, 1) == 1
    assert raw_track_revisions(raw)[(anchor, "1")] == 1
    assert [item.get("id") for item in ET.parse(raw).findall("./path/series")] == ["3", "1"]
    remove_entry_from_xml(raw, anchor, 1)
    assert get_next_series_id(raw, anchor) == 4


def test_stale_selection_and_deletion_reconcile_all_canonical_files(tmp_path):
    project = tmp_path / "project"
    images = project / "images"
    selections = project / "user_selections"
    images.mkdir(parents=True)
    selections.mkdir()
    anchor = str(images / "series_t003.png")
    raw = images / "cell_reigons.xml"
    _raw_xml(raw, anchor)
    annotator = selections / "alice.xml"
    _selection(annotator, anchor, 1)
    aggregate = selections / "aggregated_tracking.xml"
    _selection(aggregate, anchor, 1)
    tracking = selections / "tracking_review.xml"
    write_tracking_xml(tracking, [{"track_id": "T00001", "source_path": anchor, "series_id": "1", "timepoints": [_point(0)]}])

    prepare_series_extension(raw, anchor, 1)
    assert stale_selection_report(raw, [str(annotator)]) == [(str(annotator), (anchor, "1"))]
    delete_track_from_project(str(project), anchor, 1)

    assert raw_track_revisions(raw) == {}
    assert ET.parse(annotator).findall("DataEntry") == []
    assert ET.parse(aggregate).findall("DataEntry") == []
    data = read_tracking_xml(tracking)
    assert data["tracks"] == []
    assert data["metadata"]["exports_stale"] == "true"


def test_reconciliation_preserves_matching_reviewed_frames(tmp_path):
    raw = tmp_path / "cell_reigons.xml"
    anchor = str(tmp_path / "images" / "series_t003.png")
    _raw_xml(raw, anchor)
    output = tmp_path / "tracking_review.xml"
    prior = {"track_id": "T00001", "source_path": anchor, "series_id": "1", "raw_revision": 0, "review_state": "reviewed", "timepoints": [_point(0, "tightened")]}
    write_tracking_xml(output, [prior])
    candidate = {"track_id": "T00002", "source_path": anchor, "series_id": "1", "raw_revision": 1, "timepoints": [_point(0), _point(2)]}

    tracks, _ = reconcile_tracking_records([candidate], output, raw, {})
    assert tracks[0]["track_id"] == "T00001"
    assert tracks[0]["review_state"] == "pending"
    assert tracks[0]["timepoints"][0]["preferred_box_type"] == "tightened"
    assert tracks[0]["timepoints"][1]["preferred_box_type"] == "original"


def test_legacy_tracking_is_current_until_a_raw_track_is_extended(tmp_path):
    raw = tmp_path / "cell_reigons.xml"
    anchor = str(tmp_path / "images" / "series_t003.png")
    _raw_xml(raw, anchor)
    tracking = tmp_path / "tracking_review.xml"
    write_tracking_xml(tracking, [{"track_id": "T00001", "source_path": anchor, "series_id": "1", "timepoints": [_point(0)]}])

    assert tracking_is_current(tracking, raw)
    prepare_series_extension(raw, anchor, 1)
    assert not tracking_is_current(tracking, raw)


def test_tracking_is_stale_when_annotator_phase_selection_changes(tmp_path):
    project = tmp_path / "project"
    images = project / "images"
    selections = project / "user_selections"
    images.mkdir(parents=True)
    selections.mkdir()
    raw = images / "cell_reigons.xml"
    anchor = str(images / "series_t003.png")
    _raw_xml(raw, anchor)
    annotator = selections / "alice.xml"
    ET.ElementTree(ET.Element("Data")).write(annotator)
    tracking = selections / "tracking_review.xml"
    write_tracking_xml(
        tracking,
        [{"track_id": "T00001", "source_path": anchor, "series_id": "1", "timepoints": [_point(0)]}],
        metadata={"selection_fingerprint": selection_fingerprint([annotator])},
    )

    assert tracking_is_current(tracking, raw)
    ET.ElementTree(ET.Element("Data", saved="updated")).write(annotator)
    assert not tracking_is_current(tracking, raw)


def test_duplicate_check_requires_two_high_overlap_preferred_frames():
    first = {
        "track_id": "T00001", "series_id": "1", "timepoints": [_point(0), _point(1)],
    }
    second = {
        "track_id": "T00002", "series_id": "2", "timepoints": [_point(0), _point(1)],
    }
    near_but_distinct = {
        "track_id": "T00003", "series_id": "3", "timepoints": [_point(0)],
    }
    near_but_distinct["timepoints"][0]["boxes"][0]["x_center"] = .7

    duplicates = find_near_duplicate_tracks([first, second, near_but_distinct])

    assert len(duplicates) == 1
    assert duplicates[0]["first"]["track_id"] == "T00001"
    assert duplicates[0]["second"]["track_id"] == "T00002"


def test_phase_skip_requires_a_strict_annotator_majority():
    assert calculate_median_handling_negatives(pd.Series([-1, 4, 6])) == 5
    assert calculate_median_handling_negatives(pd.Series([-1, -1, 6])) == -1
    assert calculate_median_handling_negatives(pd.Series([-1, 4])) == 4


def test_project_phase_settings_default_and_custom_mapping(tmp_path):
    project = tmp_path / "project"
    assert load_phases(project) == list(DEFAULT_PHASES)

    assert save_phases(project, [{"id": 0, "name": "entry"}, {"id": 1, "name": "exit"}]) == ["entry", "exit"]
    assert load_phases(project) == ["entry", "exit"]
