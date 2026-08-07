from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from CellClicker.mitotic_tightener import (
    TIGHTENER_BOX_TYPE,
    TIGHTENER_IMGSZ_METADATA_KEY,
    TIGHTENER_SELECTION_METADATA_KEY,
    _select_prediction_by_center_confidence,
    TIGHTENER_MODEL_METADATA_KEY,
    configure_tightener_weights,
    crop_label_from_boxes,
    prepare_tightener_dataset,
    recommended_imgsz,
    run_tightener_on_tracking_xml,
    train_tightener_model,
)
from CellClicker.tracking_xml import read_tracking_xml, write_tracking_xml
from CellClicker.mitotic_tightener_ui import read_tightener_training_settings, write_tightener_training_settings


def _box(kind, x=0.5, y=0.5, width=0.2, height=0.2):
    return {"box_type": kind, "format": "yolo_xywh_norm", "x_center": x, "y_center": y, "width": width, "height": height}


def _project(tmp_path, name, preferred="otsu"):
    project = tmp_path / name
    image_path = project / "images" / "frame.png"
    image_path.parent.mkdir(parents=True)
    Image.new("L", (120, 100), 128).save(image_path)
    point = {"timepoint_index": 0, "frame_index": 0, "image_path": str(image_path), "class_id": 3,
             "phase_name": "metaphase", "source_class_id": 3, "preferred_box_type": preferred,
             "boxes": [_box("original"), _box("otsu", x=.45, y=.5, width=.1, height=.1)]}
    track = {"track_id": "T1", "source_path": str(image_path), "series_id": "1", "timepoints": [point]}
    xml_path = project / "user_selections" / "tracking_review.xml"
    write_tracking_xml(xml_path, [track], metadata={"dataset_root": str(project)})
    return project, xml_path


def test_crop_label_uses_review_bounds_and_rejects_target_outside():
    bounds, label = crop_label_from_boxes((120, 100), _box("original"), _box("otsu", width=.1, height=.1))
    assert bounds == (12, 2, 108, 98)
    assert label.startswith("0 ")
    with pytest.raises(ValueError, match="not fully contained"):
        crop_label_from_boxes((120, 100), _box("original", x=.1, y=.1, width=.05, height=.05), _box("otsu", x=.9, y=.9))


def test_prepare_dataset_writes_all_splits_and_recommends_compact_size(tmp_path):
    train, _ = _project(tmp_path, "train")
    val, _ = _project(tmp_path, "val")
    test, _ = _project(tmp_path, "test")
    result = prepare_tightener_dataset([train], [val], [test], tmp_path / "dataset")
    assert result["imgsz"] == 96
    assert result["splits"]["train"]["valid"] == 1
    assert (tmp_path / "dataset" / "images" / "test" / "test_000000.png").is_file()
    dataset_yaml = (tmp_path / "dataset" / "dataset.yaml").read_text(encoding="utf-8")
    assert "test:" in dataset_yaml and ".staging" not in dataset_yaml
    manifest = (tmp_path / "dataset" / "training_manifest.csv").read_text(encoding="utf-8")
    assert "project_dir" in manifest and "included" in manifest and "images/train/train_000000.png" in manifest
    assert recommended_imgsz(321) == 320
    with pytest.raises(ValueError, match="both train and validation"):
        prepare_tightener_dataset([train], [train], [test], tmp_path / "duplicate")


def test_moved_project_paths_rebase_from_old_dataset_root(tmp_path):
    project, xml_path = _project(tmp_path, "moved")
    data = read_tracking_xml(xml_path)
    old_root = tmp_path / "old_location" / "moved"
    old_image = old_root / "images" / "frame.png"
    data["metadata"]["dataset_root"] = str(old_root)
    data["tracks"][0]["source_path"] = str(old_image)
    data["tracks"][0]["timepoints"][0]["image_path"] = str(old_image)
    write_tracking_xml(xml_path, data["tracks"], classes=data["classes"], box_types=data["box_types"], metadata=data["metadata"])
    rebased = read_tracking_xml(xml_path)
    assert rebased["metadata"]["dataset_root"] == str(project)
    assert rebased["tracks"][0]["timepoints"][0]["image_path"] == str(project / "images" / "frame.png")
    assert prepare_tightener_dataset([project], [_project(tmp_path, "val2")[0]], [_project(tmp_path, "test2")[0]], tmp_path / "moved_dataset")["splits"]["train"]["valid"] == 1


def test_tightener_training_settings_round_trip_and_validate(tmp_path):
    settings_path = tmp_path / "settings.json"
    projects = {"train": [str(tmp_path / "train")], "val": [str(tmp_path / "val")], "test": [str(tmp_path / "test")]}
    settings = {"output_root": "Models/tightener_runs", "run_name": "experiment", "epochs": "20", "batch": "4", "patience": "5", "device": "cpu"}
    write_tightener_training_settings(settings_path, projects, settings)
    loaded_projects, loaded_settings = read_tightener_training_settings(settings_path)
    assert loaded_projects == projects
    assert loaded_settings == settings
    settings_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        read_tightener_training_settings(settings_path)


def test_configured_model_inference_maps_prediction_and_falls_back(tmp_path, monkeypatch):
    project, xml_path = _project(tmp_path, "project")
    weights = tmp_path / "best.pt"; weights.write_bytes(b"weights")
    configure_tightener_weights(xml_path, weights)
    assert read_tracking_xml(xml_path)["metadata"][TIGHTENER_MODEL_METADATA_KEY] == str(weights)
    assert read_tracking_xml(xml_path)["metadata"][TIGHTENER_IMGSZ_METADATA_KEY] == "320"
    assert read_tracking_xml(xml_path)["metadata"][TIGHTENER_SELECTION_METADATA_KEY] == "center_confidence"

    class Boxes:
        # The higher-confidence candidate is a neighbouring cell; selection
        # must retain the lower-confidence box that overlaps the prompt.
        data = np.array([[10, 8, 30, 28, .9, 0], [38, 40, 58, 60, .2, 0]])
    class Result: boxes = Boxes()
    call_kwargs = []
    class Model:
        def __call__(self, *args, **kwargs):
            call_kwargs.append(kwargs)
            return [Result()]
    monkeypatch.setattr("CellClicker.mitotic_tightener._ensure_ultralytics", lambda: lambda _: Model())
    stats = run_tightener_on_tracking_xml(xml_path)
    box = read_tracking_xml(xml_path)["tracks"][0]["timepoints"][0]["boxes"][-1]
    assert stats["created"] == 1 and box["box_type"] == TIGHTENER_BOX_TYPE
    assert box["x_center"] == pytest.approx((12 + 48) / 120)
    assert call_kwargs[0]["imgsz"] == 320 and call_kwargs[0]["conf"] == .05

    class EmptyResult: boxes = None
    class EmptyModel:
        def __call__(self, *args, **kwargs): return [EmptyResult()]
    monkeypatch.setattr("CellClicker.mitotic_tightener._ensure_ultralytics", lambda: lambda _: EmptyModel())
    stats = run_tightener_on_tracking_xml(xml_path)
    box = read_tracking_xml(xml_path)["tracks"][0]["timepoints"][0]["boxes"][-1]
    assert stats["fallback_original"] == 1 and box["width"] == pytest.approx(.2)


def test_center_confidence_selection_falls_back_to_overlap():
    original = _box("original")
    crop_bounds = (12, 2, 108, 98)
    # Neither candidate contains the prompt centre (48, 48), so the more
    # overlapping candidate wins despite the lower confidence.
    selected = _select_prediction_by_center_confidence(
        np.array([[4, 4, 25, 25, .95, 0], [35, 40, 45, 56, .2, 0]]), original, (120, 100), crop_bounds
    )
    assert selected[4] == pytest.approx(.2)

def test_training_reports_epoch_and_runs_test_split(tmp_path, monkeypatch):
    train, _ = _project(tmp_path, "train")
    val, _ = _project(tmp_path, "val")
    test, _ = _project(tmp_path, "test")
    calls, epochs = [], []
    class FakeResult: save_dir = tmp_path / "runs" / "run"
    class FakeTrainer: save_dir = tmp_path / "runs" / "run"
    class FakeModel:
        trainer = FakeTrainer()
        def __init__(self, path): self.path = path; self.callback = None
        def add_callback(self, _, callback): self.callback = callback
        def train(self, **kwargs):
            calls.append(("train", kwargs)); self.trainer.epoch = 0; self.trainer.epochs = 2; self.trainer.metrics = {"metrics/mAP50(B)": .5}; self.callback(self.trainer)
            return FakeResult()
        def val(self, **kwargs): calls.append(("val", kwargs)); return type("Metrics", (), {"results_dict": {"metrics/mAP50(B)": .6}})()
    monkeypatch.setattr("CellClicker.mitotic_tightener._ensure_ultralytics", lambda: FakeModel)
    result = train_tightener_model([train], [val], [test], tmp_path / "dataset", tmp_path / "runs", "run", 2, 1, 1, "cpu", epoch_callback=lambda *args: epochs.append(args))
    assert epochs[0][0] == 1
    assert calls[-1][0] == "val" and calls[-1][1]["split"] == "test"
    assert result["test_metrics"]["metrics/mAP50(B)"] == .6
