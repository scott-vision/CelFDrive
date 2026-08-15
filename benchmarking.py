"""Reproducible evaluation utilities for frozen CelFDrive detection models.

Inputs are read-only. All derived labels, predictions, metrics, and manifests
are written to a newly-created benchmark run directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml


SCHEMA_VERSION = 1
CANONICAL_CLASSES = ("prophase", "earlyprometaphase", "prometaphase", "metaphase", "anaphase", "telophase")
REQUIRED_LABEL_COLUMNS = {"image_id", "image_path", "group_id", "object_id", "source_label"}


@dataclass(frozen=True)
class Match:
    """One prediction-to-ground-truth assignment in a single image."""

    image_id: str
    prediction_index: int
    ground_truth_index: int
    iou: float


def load_benchmark_config(config_path):
    """Load a paper-test YAML and validate its explicit split/mapping contract."""
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Benchmark config must declare schema_version: 1")
    for key in ("dataset", "labels", "model", "inference", "split"):
        if key not in config:
            raise ValueError(f"Benchmark config is missing required '{key}' section")
    if config["split"].get("name") != "test":
        raise ValueError("Paper benchmark config split.name must be 'test'")
    test_groups = set(map(str, config["split"].get("test_group_ids", [])))
    if not test_groups:
        raise ValueError("split.test_group_ids must explicitly freeze at least one group")
    overlap = test_groups & set(map(str, config["split"].get("excluded_group_ids", [])))
    if overlap:
        raise ValueError(f"Test groups overlap excluded training/tuning groups: {sorted(overlap)}")
    class_map = {str(key): value for key, value in config["labels"].get("class_map", {}).items()}
    if set(class_map.values()) != set(CANONICAL_CLASSES):
        raise ValueError("labels.class_map must explicitly cover all six CelFDrive classes")
    invalid = {name: mapped for name, mapped in class_map.items() if mapped not in CANONICAL_CLASSES}
    if invalid:
        raise ValueError(f"labels.class_map has unsupported values: {invalid}")
    if config["labels"].get("coordinate_format") not in {"xyxy_px", "yolo_normalized"}:
        raise ValueError("labels.coordinate_format must be 'xyxy_px' or 'yolo_normalized'")
    confidence = float(config["inference"].get("confidence", -1))
    if not 0 <= confidence <= 1:
        raise ValueError("inference.confidence must be between 0 and 1")
    if not config["inference"].get("threshold_selection_source"):
        raise ValueError("inference.threshold_selection_source is required")
    config["_config_path"] = str(config_path)
    return config


def _config_path(value, config_path):
    path = Path(value)
    return path if path.is_absolute() else Path(config_path).parent / path


def select_intensity_image(array, channel_index=0, channel_axis=None):
    """Select one declared channel; reject ambiguous multi-dimensional images."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim != 3 or channel_axis is None:
        raise ValueError("Image must be 2-D, or 3-D with labels.channel_axis declared")
    if channel_axis not in (0, 1, 2):
        raise ValueError("labels.channel_axis must be 0, 1, or 2")
    if not 0 <= int(channel_index) < array.shape[channel_axis]:
        raise ValueError(f"channel_index {channel_index} is outside image shape {array.shape}")
    return np.take(array, int(channel_index), axis=channel_axis)


def image_shape(image_path, channel_axis=None, channel_index=0):
    """Read just enough image data to establish the selected 2-D image shape."""
    import tifffile

    return select_intensity_image(tifffile.imread(image_path), channel_index, channel_axis).shape


def load_internal_labels(config):
    """Read label CSV into canonical pixel-coordinate rows without editing it."""
    labels_cfg = config["labels"]
    csv_path = _config_path(labels_cfg["csv_path"], config["_config_path"])
    if not csv_path.is_file():
        raise FileNotFoundError(f"Label CSV does not exist: {csv_path}")
    labels = pd.read_csv(csv_path, dtype={column: str for column in REQUIRED_LABEL_COLUMNS})
    missing = REQUIRED_LABEL_COLUMNS - set(labels.columns)
    if missing:
        raise ValueError(f"Label CSV is missing required columns: {sorted(missing)}")
    if labels[list(REQUIRED_LABEL_COLUMNS)].isna().any().any() or (labels[list(REQUIRED_LABEL_COLUMNS)] == "").any().any():
        raise ValueError("Label CSV has empty required identifiers")
    duplicates = labels.duplicated(["image_id", "object_id"], keep=False)
    if duplicates.any():
        sample = labels.loc[duplicates, ["image_id", "object_id"]].head().to_dict("records")
        raise ValueError(f"Duplicate object_id within image: {sample}")
    class_map = {str(key): value for key, value in labels_cfg["class_map"].items()}
    unknown = sorted(set(labels.source_label.astype(str)) - set(class_map))
    if unknown:
        raise ValueError(f"Source labels missing from labels.class_map: {unknown}")
    images_root = _config_path(config["dataset"]["images_root"], config["_config_path"])
    resolved = []
    for image_id, image_path in zip(labels.image_id, labels.image_path):
        path = Path(image_path)
        path = path if path.is_absolute() else images_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Label image_id '{image_id}' does not exist: {path}")
        resolved.append(str(path.resolve()))
    labels["resolved_image_path"] = resolved
    test_groups = set(map(str, config["split"]["test_group_ids"]))
    unexpected_groups = sorted(set(labels.group_id.astype(str)) - test_groups)
    if unexpected_groups:
        raise ValueError(
            "Label CSV contains groups outside the frozen test split: "
            f"{unexpected_groups}. Supply a test-only label CSV."
        )
    coordinate_format = labels_cfg["coordinate_format"]
    if coordinate_format == "xyxy_px":
        box_columns = {"x_min", "y_min", "x_max", "y_max"}
        if box_columns - set(labels.columns):
            raise ValueError(f"xyxy_px labels are missing columns: {sorted(box_columns - set(labels.columns))}")
        for column in box_columns:
            labels[column] = pd.to_numeric(labels[column], errors="raise")
    else:
        box_columns = {"x_center", "y_center", "width", "height"}
        if box_columns - set(labels.columns):
            raise ValueError(f"yolo_normalized labels are missing columns: {sorted(box_columns - set(labels.columns))}")
        for column in box_columns:
            labels[column] = pd.to_numeric(labels[column], errors="raise")
        for path, rows in labels.groupby("resolved_image_path", sort=False):
            height, width = image_shape(path, labels_cfg.get("channel_axis"), labels_cfg.get("channel_index", 0))
            index = rows.index
            labels.loc[index, "x_min"] = (rows.x_center - rows.width / 2) * width
            labels.loc[index, "x_max"] = (rows.x_center + rows.width / 2) * width
            labels.loc[index, "y_min"] = (rows.y_center - rows.height / 2) * height
            labels.loc[index, "y_max"] = (rows.y_center + rows.height / 2) * height
    numeric = ["x_min", "y_min", "x_max", "y_max"]
    if not np.isfinite(labels[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Label boxes must contain finite numeric coordinates")
    invalid = (labels.x_min < 0) | (labels.y_min < 0) | (labels.x_max <= labels.x_min) | (labels.y_max <= labels.y_min)
    if invalid.any():
        raise ValueError(f"Invalid label boxes at rows: {labels.index[invalid].tolist()[:10]}")
    labels["class_name"] = labels.source_label.astype(str).map(class_map)
    labels["class_id"] = labels.class_name.map({name: index for index, name in enumerate(CANONICAL_CLASSES)})
    labels["centre_x"] = (labels.x_min + labels.x_max) / 2
    labels["centre_y"] = (labels.y_min + labels.y_max) / 2
    labels["annotation_provenance"] = labels_cfg.get("annotation_provenance", "unspecified")
    return labels


def preprocess_image(image, top_clip_percentile=0.01):
    """Apply CelFDrive's upper percentile clip and min--max normalization."""
    if not 0 <= top_clip_percentile < 100:
        raise ValueError("top_clip_percentile must be in [0, 100)")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Inference image must be a finite two-dimensional array")
    upper = np.percentile(image, 100 - top_clip_percentile)
    clipped = np.minimum(image, upper)
    if upper <= clipped.min():
        return np.zeros_like(clipped, dtype=np.uint8)
    return np.round((clipped - clipped.min()) / (upper - clipped.min()) * 255).astype(np.uint8)


def prepare_model_input(image, expected_channels):
    """Adapt a selected grayscale microscopy channel to a model's declared input.

    A three-channel model receives three identical copies of the same selected
    fluorescence channel; no colour composite or second biological channel is
    introduced.  Other input channel counts are rejected rather than guessed.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Model input preparation requires one two-dimensional intensity image")
    if expected_channels == 1:
        return image
    if expected_channels == 3:
        return np.repeat(image[:, :, np.newaxis], 3, axis=2)
    raise ValueError(f"Unsupported checkpoint input-channel count: {expected_channels}")


def box_iou(first, second):
    """Calculate IoU for two ``x_min, y_min, x_max, y_max`` boxes."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return 0.0 if union <= 0 else intersection / union


def match_boxes(predictions, ground_truth, iou_threshold=0.5, class_aware=True):
    """Greedily make one-to-one matches in descending confidence order."""
    predictions, ground_truth = predictions.reset_index(drop=True), ground_truth.reset_index(drop=True)
    matches, used_ground_truth = [], set()
    for prediction_index in predictions.sort_values("confidence", ascending=False).index:
        prediction = predictions.loc[prediction_index]
        candidates = []
        for ground_truth_index, reference in ground_truth.iterrows():
            if ground_truth_index in used_ground_truth or (class_aware and prediction.class_name != reference.class_name):
                continue
            iou = box_iou((prediction.x_min, prediction.y_min, prediction.x_max, prediction.y_max), (reference.x_min, reference.y_min, reference.x_max, reference.y_max))
            if iou >= iou_threshold:
                candidates.append((iou, ground_truth_index))
        if candidates:
            iou, ground_truth_index = max(candidates)
            used_ground_truth.add(ground_truth_index)
            matches.append(Match(str(prediction.image_id), int(prediction_index), int(ground_truth_index), float(iou)))
    return matches


def _average_precision(predictions, ground_truth, class_name, iou_threshold):
    predictions = predictions[predictions.class_name == class_name].sort_values("confidence", ascending=False).reset_index(drop=True)
    reference = ground_truth[ground_truth.class_name == class_name]
    if len(reference) == 0:
        return np.nan
    matched, values = set(), []
    for _, prediction in predictions.iterrows():
        best_iou, best_index = 0.0, None
        for index, target in reference[reference.image_id == prediction.image_id].iterrows():
            if index not in matched:
                iou = box_iou((prediction.x_min, prediction.y_min, prediction.x_max, prediction.y_max), (target.x_min, target.y_min, target.x_max, target.y_max))
                if iou > best_iou:
                    best_iou, best_index = iou, index
        values.append(best_iou >= iou_threshold)
        if values[-1]:
            matched.add(best_index)
    if not values:
        return 0.0
    values = np.asarray(values, dtype=float)
    recall = np.cumsum(values) / len(reference)
    precision = np.cumsum(values) / np.arange(1, len(values) + 1)
    recall, precision = np.r_[0, recall, 1], np.r_[1, precision, 0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def detection_metrics(predictions, ground_truth, classes=CANONICAL_CLASSES):
    """Calculate class-specific AP and metrics at the configured confidence."""
    rows = []
    for class_name in classes:
        predicted, reference = predictions[predictions.class_name == class_name], ground_truth[ground_truth.class_name == class_name]
        tp = len(match_boxes(predicted, reference))
        fp, fn = len(predicted) - tp, len(reference) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        ap_values = [_average_precision(predictions, ground_truth, class_name, threshold) for threshold in np.arange(.5, 1, .05)]
        rows.append({"class_name": class_name, "support": len(reference), "predictions": len(predicted), "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "ap50": _average_precision(predictions, ground_truth, class_name, .5), "ap50_95": float(np.mean(ap_values)) if len(reference) else np.nan})
    metrics = pd.DataFrame(rows)
    macro = {"class_name": "macro", "support": int(metrics.support.sum()), "predictions": int(metrics.predictions.sum())}
    macro.update({field: float(metrics[field].mean()) for field in ("precision", "recall", "f1", "ap50", "ap50_95")})
    return pd.concat((metrics, pd.DataFrame([macro])), ignore_index=True)


def classification_outputs(predictions, ground_truth, classes=CANONICAL_CLASSES):
    """Return matched classification rows and a stage confusion matrix."""
    rows = []
    for image_id in sorted(set(predictions.image_id) | set(ground_truth.image_id)):
        predicted = predictions[predictions.image_id == image_id].reset_index(drop=True)
        reference = ground_truth[ground_truth.image_id == image_id].reset_index(drop=True)
        for match in match_boxes(predicted, reference, class_aware=False):
            rows.append({"image_id": image_id, "group_id": reference.loc[match.ground_truth_index, "group_id"], "ground_truth_class": reference.loc[match.ground_truth_index, "class_name"], "prediction_class": predicted.loc[match.prediction_index, "class_name"], "iou": match.iou})
    matched = pd.DataFrame(rows, columns=["image_id", "group_id", "ground_truth_class", "prediction_class", "iou"])
    confusion = pd.crosstab(matched.ground_truth_class if len(matched) else pd.Series(dtype=str), matched.prediction_class if len(matched) else pd.Series(dtype=str)).reindex(index=classes, columns=classes, fill_value=0)
    return matched, confusion


def target_operating_points(predictions, ground_truth, false_positives_per_field=(.1, .5, 1.0)):
    """Measure any-stage target recall at fixed false-positive-per-field rates."""
    image_count, rows = max(1, ground_truth.image_id.nunique()), []
    thresholds = sorted(set(predictions.confidence), reverse=True) + [1.01]
    for target_rate in false_positives_per_field:
        eligible = []
        for threshold in thresholds:
            selected = predictions[predictions.confidence >= threshold]
            matches = sum((match_boxes(selected[selected.image_id == image].reset_index(drop=True), ground_truth[ground_truth.image_id == image].reset_index(drop=True), class_aware=False) for image in set(selected.image_id) | set(ground_truth.image_id)), [])
            tp, fp = len(matches), len(selected) - len(matches)
            if fp / image_count <= target_rate:
                eligible.append((threshold, tp, fp))
        threshold, tp, fp = min(eligible, key=lambda entry: entry[0]) if eligible else (1.01, 0, 0)
        rows.append({"target_false_positives_per_field": target_rate, "confidence_threshold": threshold, "true_positives": tp, "false_positives": fp, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / len(ground_truth) if len(ground_truth) else 0.0})
    return pd.DataFrame(rows)


def bootstrap_group_metrics(predictions, ground_truth, iterations=1000, seed=42):
    """Bootstrap overall precision, recall, and F1 by independent acquisition group."""
    groups = np.asarray(sorted(ground_truth.group_id.unique()))
    if len(groups) < 2:
        raise ValueError("At least two independent group_id values are required for bootstrap confidence intervals")
    group_scores = []
    for group in groups:
        reference = ground_truth[ground_truth.group_id == group]
        predicted = predictions[predictions.group_id == group]
        tp = sum((len(match_boxes(predicted[predicted.image_id == image].reset_index(drop=True), reference[reference.image_id == image].reset_index(drop=True))) for image in set(predicted.image_id) | set(reference.image_id)))
        precision, recall = tp / len(predicted) if len(predicted) else 0.0, tp / len(reference) if len(reference) else 0.0
        group_scores.append((precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0))
    group_scores = np.asarray(group_scores)
    rng, values = np.random.default_rng(seed), []
    for _ in range(iterations):
        values.append(group_scores[rng.integers(len(group_scores), size=len(group_scores))].mean(axis=0))
    values = pd.DataFrame(values, columns=("precision", "recall", "f1"))
    return pd.DataFrame({"metric": values.columns, "ci_low": values.quantile(.025).values, "ci_high": values.quantile(.975).values})


def _sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_commit(repo_root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def create_run_directory(output_root, run_name=None):
    """Create a unique output directory; existing result directories are never reused."""
    run_id = f"{run_name or 'benchmark'}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    run_directory = Path(output_root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _write_table(table, path):
    table.to_csv(path.with_suffix(".csv"), index=False)
    try:
        table.to_parquet(path.with_suffix(".parquet"), index=False)
    except ImportError as error:
        raise RuntimeError("Parquet output requires pyarrow; install the documented environment dependencies") from error


def target_centre_errors(predictions, ground_truth, pixel_size_um=None):
    """Return centre offsets for class-agnostic matched target objects."""
    rows = []
    for image_id in sorted(set(predictions.image_id) | set(ground_truth.image_id)):
        predicted = predictions[predictions.image_id == image_id].reset_index(drop=True)
        reference = ground_truth[ground_truth.image_id == image_id].reset_index(drop=True)
        for match in match_boxes(predicted, reference, class_aware=False):
            prediction, target = predicted.loc[match.prediction_index], reference.loc[match.ground_truth_index]
            distance_px = float(np.hypot(prediction.centre_x - target.centre_x, prediction.centre_y - target.centre_y))
            rows.append({"image_id": image_id, "group_id": target.group_id, "object_id": target.object_id,
                         "iou": match.iou, "centre_error_px": distance_px,
                         "centre_error_um": distance_px * float(pixel_size_um) if pixel_size_um is not None else np.nan})
    return pd.DataFrame(rows, columns=("image_id", "group_id", "object_id", "iou", "centre_error_px", "centre_error_um"))


def build_manual_review_queue(ground_truth, minimum_per_source=50, seed=42):
    """Select a deterministic, stage-stratified manual label-review queue."""
    desired = min(len(ground_truth), minimum_per_source)
    if desired == 0:
        return ground_truth.assign(review_status=pd.Series(dtype=str), review_reason=pd.Series(dtype=str))
    rng = np.random.default_rng(seed)
    selected = []
    for _, rows in ground_truth.groupby("class_name", sort=True):
        selected.extend(rng.choice(rows.index.to_numpy(), size=min(len(rows), max(1, desired // len(CANONICAL_CLASSES))), replace=False).tolist())
    remaining = [index for index in ground_truth.index if index not in selected]
    if len(selected) < desired:
        selected.extend(rng.choice(remaining, size=desired - len(selected), replace=False).tolist())
    queue = ground_truth.loc[selected].copy()
    queue["review_status"] = "pending"
    queue["review_reason"] = "stratified_stage_sample"
    return queue.sort_values(["class_name", "image_id", "object_id"]).reset_index(drop=True)


def write_quality_overlays(predictions, ground_truth, output_directory, image_ids):
    """Write deterministic PNG/SVG overlays for matches and visible error cases."""
    import matplotlib.pyplot as plt
    import tifffile

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for image_id in sorted(set(image_ids)):
        reference = ground_truth[ground_truth.image_id == image_id]
        predicted = predictions[predictions.image_id == image_id]
        image = tifffile.imread(reference.resolved_image_path.iloc[0])
        figure, axis = plt.subplots(figsize=(8, 8))
        axis.imshow(image, cmap="gray")
        for _, row in reference.iterrows():
            axis.add_patch(plt.Rectangle((row.x_min, row.y_min), row.x_max - row.x_min, row.y_max - row.y_min, fill=False, edgecolor="lime", linewidth=1))
            axis.text(row.x_min, row.y_min, str(row.class_name), color="lime", fontsize=5, va="bottom", bbox={"facecolor": "black", "alpha": .6, "pad": 1, "edgecolor": "none"})
        for _, row in predicted.iterrows():
            axis.add_patch(plt.Rectangle((row.x_min, row.y_min), row.x_max - row.x_min, row.y_max - row.y_min, fill=False, edgecolor="red", linewidth=1))
            axis.text(row.x_min, row.y_max, f"{row.class_name} {row.confidence:.2f}", color="red", fontsize=5, va="top", bbox={"facecolor": "black", "alpha": .6, "pad": 1, "edgecolor": "none"})
        axis.set_title(f"{image_id}: green ground truth, red prediction")
        axis.axis("off")
        for extension in ("png", "svg"):
            figure.savefig(output_directory / f"{image_id}.{extension}", bbox_inches="tight", dpi=160)
        plt.close(figure)


def run_internal_benchmark(config_path, output_root=r"D:\CelFDriveBenchmark\runs", run_name=None):
    """Run frozen inference and return the newly-created audit directory."""
    config, labels = load_benchmark_config(config_path), None
    labels = load_internal_labels(config)
    run_directory = create_run_directory(output_root, run_name)
    try:
        from ultralytics import YOLO
        import tifffile
        import torch

        model_path = _config_path(config["model"]["weights_path"], config["_config_path"])
        actual_hash = _sha256(model_path)
        expected_hash = config["model"].get("sha256", "").upper()
        if expected_hash and expected_hash != actual_hash:
            raise ValueError(f"Model SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")
        model = YOLO(str(model_path))
        names = {int(key): str(value) for key, value in model.names.items()}
        if tuple(names[index] for index in sorted(names)) != CANONICAL_CLASSES:
            raise ValueError(f"Checkpoint class map does not match CelFDrive classes: {names}")
        expected_channels = int(model.model.yaml.get("ch", 3))
        rows = []
        for image_id, reference in labels.groupby("image_id", sort=True):
            image = select_intensity_image(tifffile.imread(reference.resolved_image_path.iloc[0]), config["labels"].get("channel_index", 0), config["labels"].get("channel_axis"))
            normalized = preprocess_image(image, float(config["inference"].get("top_clip_percentile", .01)))
            result = model.predict(prepare_model_input(normalized, expected_channels), imgsz=int(config["inference"].get("imgsz", 640)), conf=float(config["inference"]["confidence"]), iou=float(config["inference"].get("iou", .7)), device=config["inference"].get("device"), verbose=False)[0]
            if result.boxes is not None:
                for box, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                    rows.append({"image_id": image_id, "group_id": reference.group_id.iloc[0], "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3], "centre_x": (box[0] + box[2]) / 2, "centre_y": (box[1] + box[3]) / 2, "confidence": confidence, "class_id": class_id, "class_name": names[class_id]})
        predictions = pd.DataFrame(rows, columns=("image_id", "group_id", "x_min", "y_min", "x_max", "y_max", "centre_x", "centre_y", "confidence", "class_id", "class_name"))
        _write_table(labels, run_directory / "ground_truth")
        _write_table(predictions, run_directory / "predictions")
        _write_table(detection_metrics(predictions, labels), run_directory / "detection_metrics")
        matches, confusion = classification_outputs(predictions, labels)
        _write_table(matches, run_directory / "classification_matches")
        confusion.to_csv(run_directory / "confusion_matrix.csv")
        _write_table(target_operating_points(predictions, labels), run_directory / "target_operating_points")
        _write_table(bootstrap_group_metrics(predictions, labels, int(config.get("bootstrap_iterations", 1000))), run_directory / "bootstrap_confidence_intervals")
        pixel_size_um = config["dataset"].get("pixel_size_um")
        if pixel_size_um is not None and float(pixel_size_um) <= 0:
            raise ValueError("dataset.pixel_size_um must be positive when supplied")
        _write_table(target_centre_errors(predictions, labels, pixel_size_um), run_directory / "target_centre_errors")
        review_queue = build_manual_review_queue(labels, int(config.get("minimum_manual_review_labels", 50)), int(config.get("review_seed", 42)))
        _write_table(review_queue, run_directory / "manual_review_queue")
        write_quality_overlays(predictions, labels, run_directory / "overlays", review_queue.image_id)
        manifest = {"schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "config_path": str(Path(config_path).resolve()), "model_path": str(model_path), "model_sha256": actual_hash, "model_class_map": names, "model_input_channels": expected_channels, "model_input_conversion": "selected_grayscale_replicated_to_rgb" if expected_channels == 3 else "selected_grayscale", "inference": config["inference"], "test_group_ids": config["split"]["test_group_ids"], "excluded_group_ids": config["split"].get("excluded_group_ids", []), "pixel_size_um": pixel_size_um, "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "git_commit": _git_commit(Path(__file__).resolve().parent)}
        (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        (run_directory / "FAILED.txt").write_text("Benchmark did not complete; inspect the command error before using this directory.", encoding="utf-8")
        raise
    return run_directory


def inventory_ctc_dataset(dataset_root):
    """Inventory CTC images/masks for pipeline checks; CTC has no stage labels."""
    root, rows = Path(dataset_root), []
    for image_path in sorted(root.glob("*/t*.tif")):
        sequence, frame = image_path.parent.name, image_path.stem
        mask = root / f"{sequence}_GT" / "SEG" / f"man_seg{frame[1:]}.tif"
        rows.append({"image_path": str(image_path.resolve()), "sequence_id": sequence, "frame_id": frame, "segmentation_mask_path": str(mask.resolve()) if mask.is_file() else None, "source_label": "nucleus", "stage_label_available": False})
    if not rows:
        raise FileNotFoundError(f"No CTC images matching <sequence>/t*.tif below {root}")
    return pd.DataFrame(rows)


def inventory_cellcognition_dataset(images_root, analysis_root=None):
    """Inventory H2B-mCherry TIFFs without claiming analysis tables are ground truth."""
    rows = [{"image_path": str(path.resolve()), "position_id": path.parent.name, "frame_id": path.stem, "channel": "H2B-mCherry", "stage_label_available": False, "analysis_root": str(Path(analysis_root).resolve()) if analysis_root else None} for path in sorted(Path(images_root).rglob("*Crfp*.tif"))]
    if not rows:
        raise FileNotFoundError(f"No CellCognition Crfp TIFF images below {images_root}")
    return pd.DataFrame(rows)
