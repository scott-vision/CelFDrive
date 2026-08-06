from pathlib import Path

from PIL import Image
import pytest

from CellClicker.tracking_export import (
    export_tracking_xml_to_coco,
    export_tracking_xml_to_miniseries,
    export_tracking_xml_to_yolo,
)
from CellClicker.tracking_xml import read_tracking_xml, write_tracking_xml
from CellClicker.yolo_training_ui import _sync_project_labels, prepare_yolo_sources
from CellClicker.tracking_sam2 import _supports_cuda_capability


def make_track(project_dir: Path, image_path: Path):
    return {
        "track_id": "T00001",
        "source_path": str(image_path),
        "series_id": "1",
        "timepoints": [
            {
                "timepoint_index": 0,
                "frame_index": 7,
                "image_path": str(image_path),
                "class_id": 2,
                "phase_name": "prometaphase",
                "source_class_id": 7,
                "preferred_box_type": "otsu",
                "boxes": [
                    {
                        "box_type": "original",
                        "format": "yolo_xywh_norm",
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.5,
                        "height": 0.5,
                        "source": "cell_reigons.xml",
                    },
                    {
                        "box_type": "otsu",
                        "format": "yolo_xywh_norm",
                        "x_center": 0.4,
                        "y_center": 0.4,
                        "width": 0.2,
                        "height": 0.2,
                        "source": "otsu_threshold",
                    },
                ],
            }
        ],
    }


def make_project(tmp_path: Path):
    project_dir = tmp_path / "project"
    image_path = project_dir / "images" / "series" / "frame_007.png"
    image_path.parent.mkdir(parents=True)
    Image.new("L", (100, 80), color=128).save(image_path)
    tracking_path = project_dir / "user_selections" / "tracking_review.xml"
    write_tracking_xml(
        tracking_path,
        [make_track(project_dir, image_path)],
        metadata={"dataset_root": str(project_dir)},
    )
    return project_dir, image_path, tracking_path


def test_tracking_xml_round_trip_and_exports(tmp_path):
    project_dir, _, tracking_path = make_project(tmp_path)

    tracking_data = read_tracking_xml(tracking_path)
    timepoint = tracking_data["tracks"][0]["timepoints"][0]
    assert timepoint["frame_index"] == 7
    assert timepoint["preferred_box_type"] == "otsu"
    assert [box["box_type"] for box in timepoint["boxes"]] == ["original", "otsu"]

    labels_dir = project_dir / "user_selections" / "exported_labels"
    labels = export_tracking_xml_to_yolo(tracking_path, labels_dir)
    assert labels == {"series/frame_007.txt": ["2 0.4 0.4 0.2 0.2"]}
    assert (labels_dir / "series" / "frame_007.txt").read_text(encoding="utf-8") == "2 0.4 0.4 0.2 0.2\n"

    coco = export_tracking_xml_to_coco(tracking_path, project_dir / "annotations.json")
    assert coco["images"][0]["file_name"] == "series/frame_007.png"
    assert coco["annotations"][0]["bbox"] == [30.0, 24.0, 20.0, 16.0]

    crops = export_tracking_xml_to_miniseries(tracking_path, project_dir / "miniseries")
    assert len(crops) == 1
    assert Path(crops[0]).is_file()


def test_tracking_export_rejects_missing_preferred_box(tmp_path):
    project_dir, _, tracking_path = make_project(tmp_path)
    tracking_data = read_tracking_xml(tracking_path)
    tracking_data["tracks"][0]["timepoints"][0]["preferred_box_type"] = "sam2"
    write_tracking_xml(
        tracking_path,
        tracking_data["tracks"],
        classes=tracking_data["classes"],
        box_types=tracking_data["box_types"],
        metadata=tracking_data["metadata"],
    )

    with pytest.raises(ValueError, match="sam2"):
        export_tracking_xml_to_yolo(tracking_path, project_dir / "labels")


def test_tracking_xml_preserves_empty_schema_mappings_and_exports_to_current_directory(tmp_path, monkeypatch):
    project_dir, _, tracking_path = make_project(tmp_path)
    write_tracking_xml(tracking_path, [], classes={}, box_types={}, metadata={})

    tracking_data = read_tracking_xml(tracking_path)
    assert tracking_data["classes"] == {}
    assert tracking_data["box_types"] == {}

    monkeypatch.chdir(tmp_path)
    export_tracking_xml_to_coco(tracking_path, "annotations.json")
    assert (tmp_path / "annotations.json").is_file()


def test_label_sync_keeps_existing_labels_when_no_replacements_exist(tmp_path):
    project_dir, _, _ = make_project(tmp_path)
    existing_label = project_dir / "labels" / "series" / "frame_007.txt"
    existing_label.parent.mkdir(parents=True)
    existing_label.write_text("old annotation\n", encoding="utf-8")
    (project_dir / "user_selections" / "exported_labels").mkdir()

    with pytest.raises(ValueError, match="no images with exported labels"):
        _sync_project_labels(project_dir)

    assert existing_label.read_text(encoding="utf-8") == "old annotation\n"


def test_builder_orders_timepoints_and_assigns_phases(monkeypatch):
    pytest.importorskip("cv2")
    from CellClicker import build_tracking_xml

    monkeypatch.setattr(
        build_tracking_xml,
        "resolve_image_path",
        lambda anchor_path, step_back, dataset_root=None: f"/project/images/frame_{step_back}.png",
    )
    labels = [
        {"class_id": "0", "x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1},
        {"class_id": "2", "x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1},
    ]
    track = build_tracking_xml.build_track_record(
        1,
        "anchor.png",
        "series-1",
        labels,
        {"prophase": 0, "metaphase": 1},
    )

    assert [point["source_class_id"] for point in track["timepoints"]] == [2, 0]
    assert [point["phase_name"] for point in track["timepoints"]] == ["prophase", "metaphase"]
    assert [point["class_id"] for point in track["timepoints"]] == [0, 3]


def test_otsu_generation_updates_existing_variant(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from CellClicker.tracking_otsu import run_otsu_on_tracking_xml

    _, _, tracking_path = make_project(tmp_path)

    monkeypatch.setattr(
        "CellClicker.tracking_otsu.adjust_bbox_via_threshold",
        lambda *args: (2, 0.3, 0.3, 0.1, 0.1),
    )
    stats = run_otsu_on_tracking_xml(tracking_path, overwrite=True)
    assert stats["updated"] == 1
    assert read_tracking_xml(tracking_path)["tracks"][0]["timepoints"][0]["boxes"][1]["x_center"] == 0.3


def test_training_source_preparation(tmp_path):
    project_dir, _, tracking_path = make_project(tmp_path)
    export_tracking_xml_to_yolo(tracking_path, project_dir / "user_selections" / "exported_labels")
    result = prepare_yolo_sources(
        [project_dir],
        [project_dir],
        tmp_path / "dataset.yaml",
    )
    assert result["train_count"] == result["val_count"] == 1
    assert (project_dir / "labels" / "series" / "frame_007.txt").is_file()
    assert "names:" in (tmp_path / "dataset.yaml").read_text(encoding="utf-8")


def test_cuda_architecture_check_accepts_compatible_minor_version():
    assert _supports_cuda_capability((8, 9), {"sm_75", "sm_86", "sm_90"})
    assert not _supports_cuda_capability((8, 9), {"sm_75", "sm_90"})
