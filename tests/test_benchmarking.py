import pandas as pd
import numpy as np
import pytest
import yaml

import benchmarking


CLASS_MAP = {
    "p": "prophase",
    "ep": "earlyprometaphase",
    "pm": "prometaphase",
    "m": "metaphase",
    "a": "anaphase",
    "t": "telophase",
}


def _config(tmp_path, labels_path, **overrides):
    config = {
        "schema_version": 1,
        "dataset": {"images_root": str(tmp_path)},
        "labels": {"csv_path": str(labels_path), "coordinate_format": "xyxy_px", "class_map": CLASS_MAP},
        "model": {"weights_path": "model.pt"},
        "inference": {"confidence": .3, "threshold_selection_source": "validation-run-01"},
        "split": {"name": "test", "test_group_ids": ["day-1", "day-2"], "excluded_group_ids": ["training-day"]},
    }
    config.update(overrides)
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _labels(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame([
        {"image_id": "one", "image_path": "one.tif", "group_id": "day-1", "object_id": "1", "source_label": "p", "x_min": 1, "y_min": 2, "x_max": 9, "y_max": 12},
        {"image_id": "two", "image_path": "two.tif", "group_id": "day-2", "object_id": "1", "source_label": "m", "x_min": 4, "y_min": 4, "x_max": 12, "y_max": 12},
    ]).to_csv(path, index=False)
    for name in ("one.tif", "two.tif"):
        (tmp_path / name).touch()
    return path


def test_load_internal_labels_maps_boxes_and_preserves_provenance(tmp_path):
    config = benchmarking.load_benchmark_config(_config(tmp_path, _labels(tmp_path)))
    labels = benchmarking.load_internal_labels(config)

    assert labels.class_name.tolist() == ["prophase", "metaphase"]
    assert labels.class_id.tolist() == [0, 3]
    assert labels.centre_x.tolist() == [5, 8]
    assert labels.annotation_provenance.tolist() == ["unspecified", "unspecified"]


def test_load_internal_labels_rejects_unknown_label(tmp_path):
    label_path = _labels(tmp_path)
    labels = pd.read_csv(label_path)
    labels.loc[0, "source_label"] = "unknown"
    labels.to_csv(label_path, index=False)

    with pytest.raises(ValueError, match="missing from labels.class_map"):
        benchmarking.load_internal_labels(benchmarking.load_benchmark_config(_config(tmp_path, label_path)))


def test_load_internal_labels_rejects_duplicate_object_id(tmp_path):
    label_path = _labels(tmp_path)
    labels = pd.read_csv(label_path)
    labels.loc[1, "image_id"] = "one"
    labels["object_id"] = labels["object_id"].astype(str)
    labels.loc[1, "object_id"] = "1"
    labels.to_csv(label_path, index=False)

    with pytest.raises(ValueError, match="Duplicate object_id"):
        benchmarking.load_internal_labels(benchmarking.load_benchmark_config(_config(tmp_path, label_path)))


def test_load_internal_labels_rejects_groups_outside_test_split(tmp_path):
    label_path = _labels(tmp_path)
    labels = pd.read_csv(label_path)
    labels.loc[1, "group_id"] = "training-day"
    labels.to_csv(label_path, index=False)

    with pytest.raises(ValueError, match="outside the frozen test split"):
        benchmarking.load_internal_labels(benchmarking.load_benchmark_config(_config(tmp_path, label_path)))


def _boxes():
    ground_truth = pd.DataFrame([
        {"image_id": "one", "group_id": "g1", "object_id": "a", "class_name": "prophase", "x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10, "centre_x": 5, "centre_y": 5},
        {"image_id": "two", "group_id": "g2", "object_id": "b", "class_name": "metaphase", "x_min": 10, "y_min": 10, "x_max": 20, "y_max": 20, "centre_x": 15, "centre_y": 15},
    ])
    predictions = pd.DataFrame([
        {"image_id": "one", "group_id": "g1", "class_name": "prophase", "confidence": .9, "x_min": 1, "y_min": 1, "x_max": 9, "y_max": 9, "centre_x": 5, "centre_y": 5},
        {"image_id": "two", "group_id": "g2", "class_name": "anaphase", "confidence": .8, "x_min": 10, "y_min": 10, "x_max": 20, "y_max": 20, "centre_x": 15, "centre_y": 15},
    ])
    return predictions, ground_truth


def test_metrics_keep_detection_and_classification_failures_separate():
    predictions, ground_truth = _boxes()
    metrics = benchmarking.detection_metrics(predictions, ground_truth)
    matched, confusion = benchmarking.classification_outputs(predictions, ground_truth)

    assert metrics.loc[metrics.class_name == "prophase", "recall"].item() == 1
    assert metrics.loc[metrics.class_name == "metaphase", "recall"].item() == 0
    assert len(matched) == 2
    assert confusion.loc["metaphase", "anaphase"] == 1


def test_target_centre_errors_converts_to_micrometres():
    predictions, ground_truth = _boxes()
    errors = benchmarking.target_centre_errors(predictions, ground_truth, pixel_size_um=.5)

    assert errors.centre_error_px.tolist() == [0.0, 0.0]
    assert errors.centre_error_um.tolist() == [0.0, 0.0]


def test_prepare_model_input_replicates_a_single_selected_channel():
    image = np.array([[1, 2], [3, 4]], dtype="uint8")
    prepared = benchmarking.prepare_model_input(image, 3)

    assert prepared.shape == (2, 2, 3)
    assert (prepared[:, :, 0] == image).all()
    assert (prepared[:, :, 1] == image).all()


def test_prepare_model_input_rejects_unrecognised_channel_count():
    with pytest.raises(ValueError, match="input-channel"):
        benchmarking.prepare_model_input([[1]], 2)


def test_manual_review_queue_is_stage_stratified_and_reproducible():
    _, ground_truth = _boxes()
    queue = benchmarking.build_manual_review_queue(ground_truth, minimum_per_source=2, seed=3)

    assert queue.object_id.tolist() == ["b", "a"]
    assert queue.review_status.tolist() == ["pending", "pending"]


def test_public_dataset_inventory_adapters_do_not_claim_stage_ground_truth(tmp_path):
    ctc_image = tmp_path / "01" / "t000.tif"
    ctc_mask = tmp_path / "01_GT" / "SEG" / "man_seg000.tif"
    ctc_image.parent.mkdir()
    ctc_mask.parent.mkdir(parents=True)
    ctc_image.touch()
    ctc_mask.touch()
    cellcognition_image = tmp_path / "cellcognition" / "0013" / "tubulin_P0013_T00001_Crfp_Z1_S1.tif"
    cellcognition_image.parent.mkdir(parents=True)
    cellcognition_image.touch()

    ctc = benchmarking.inventory_ctc_dataset(tmp_path)
    cellcognition = benchmarking.inventory_cellcognition_dataset(tmp_path / "cellcognition")

    assert ctc.loc[0, "segmentation_mask_path"].endswith("man_seg000.tif")
    assert ctc.loc[0, "stage_label_available"] == False
    assert cellcognition.loc[0, "channel"] == "H2B-mCherry"
    assert cellcognition.loc[0, "stage_label_available"] == False


def test_bootstrap_requires_independent_groups():
    predictions, ground_truth = _boxes()
    ground_truth["group_id"] = "one"
    predictions["group_id"] = "one"

    with pytest.raises(ValueError, match="At least two"):
        benchmarking.bootstrap_group_metrics(predictions, ground_truth)
