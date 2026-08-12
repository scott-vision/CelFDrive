"""Train and apply a local YOLO11 cell bounding-box tightener.

The detector sees the same contextual crop used by the tracking-review UI.  It
learns one class (``cell``) and maps its crop-local prediction back to a
normalized full-image tracking box.
"""

import csv
import json
import os
import shutil
import tempfile
from datetime import datetime

from PIL import Image

from .tracking_xml import read_tracking_xml, write_tracking_data


CROP_MARGIN_FACTOR = 1.8
MIN_CROP_SIZE = 96
TIGHTENER_BOX_TYPE = "yolo11_tightened"
TIGHTENER_MODEL_METADATA_KEY = "mitotic_tightener_weights"
TIGHTENER_IMGSZ_METADATA_KEY = "mitotic_tightener_imgsz"
TIGHTENER_SELECTION_METADATA_KEY = "mitotic_tightener_selection"
TIGHTENER_CLASS_MODE_METADATA_KEY = "mitotic_tightener_class_mode"
TIGHTENER_CLASS_MAPPING_METADATA_KEY = "mitotic_tightener_class_mapping"
MODEL_METADATA_FILENAME = "mitotic_tightener_model.json"
DEFAULT_TIGHTENER_SELECTION = "center_confidence"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_TIGHTENER_STORAGE_ROOT = os.environ.get(
    "CELLCLICKER_TIGHTENER_STORAGE_ROOT",
    r"D:\Scott\home\Brook\TrainingData\cell_tightener",
)
DEFAULT_TIGHTENER_DATASETS_ROOT = os.path.join(DEFAULT_TIGHTENER_STORAGE_ROOT, "datasets")
DEFAULT_TIGHTENER_RUNS_ROOT = os.path.join(DEFAULT_TIGHTENER_STORAGE_ROOT, "runs")
DEFAULT_TIGHTENER_MODELS_ROOT = os.path.join(DEFAULT_TIGHTENER_STORAGE_ROOT, "models")


def _ensure_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required for cell tightener training and inference.") from exc
    return YOLO


def _normalise_path(path):
    return os.path.normpath(os.fspath(path))


def _trained_model_imgsz(weights_path, default=320):
    """Read the Ultralytics run image size stored next to ``weights/best.pt``."""
    args_path = os.path.join(os.path.dirname(os.path.dirname(weights_path)), "args.yaml")
    if not os.path.isfile(args_path):
        return default
    with open(args_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("imgsz:"):
                try:
                    value = int(line.split(":", 1)[1].strip())
                except ValueError:
                    break
                if value > 0:
                    return value
                break
    return default


def _find_box(timepoint, box_type):
    for box in timepoint.get("boxes", []):
        if box.get("box_type") == box_type:
            return box
    return None


def _preferred_box(timepoint):
    preferred_type = timepoint.get("preferred_box_type")
    return _find_box(timepoint, preferred_type) if preferred_type else None


def review_crop_bounds(image_size, box):
    """Return clipped integer crop bounds matching ``TrackingReviewUI``."""
    image_width, image_height = image_size
    center_x = float(box["x_center"]) * image_width
    center_y = float(box["y_center"]) * image_height
    width = max(MIN_CROP_SIZE, float(box["width"]) * image_width * CROP_MARGIN_FACTOR)
    height = max(MIN_CROP_SIZE, float(box["height"]) * image_height * CROP_MARGIN_FACTOR)
    x1 = max(0, int(round(center_x - width / 2)))
    y1 = max(0, int(round(center_y - height / 2)))
    x2 = min(image_width, int(round(center_x + width / 2)))
    y2 = min(image_height, int(round(center_y + height / 2)))
    if x2 <= x1:
        x2 = min(image_width, x1 + MIN_CROP_SIZE)
    if y2 <= y1:
        y2 = min(image_height, y1 + MIN_CROP_SIZE)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Review crop has no pixels after clipping to image bounds.")
    return x1, y1, x2, y2


def _box_xyxy(box, image_size):
    image_width, image_height = image_size
    width = float(box["width"]) * image_width
    height = float(box["height"]) * image_height
    center_x = float(box["x_center"]) * image_width
    center_y = float(box["y_center"]) * image_height
    return center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2


def _iou_xyxy(first, second):
    """Return intersection-over-union for two pixel ``(x1, y1, x2, y2)`` boxes."""
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _select_prediction_by_original_overlap(predictions, original_box, image_size, crop_bounds):
    """Choose the crop-local prediction with greatest overlap with the prompt box."""
    crop_x1, crop_y1, _, _ = crop_bounds
    original_x1, original_y1, original_x2, original_y2 = _box_xyxy(original_box, image_size)
    prompt_in_crop = (original_x1 - crop_x1, original_y1 - crop_y1, original_x2 - crop_x1, original_y2 - crop_y1)
    return max(predictions, key=lambda row: _iou_xyxy(row[:4], prompt_in_crop))


def _select_prediction_by_center_confidence(predictions, original_box, image_size, crop_bounds):
    """Prefer confidence among boxes containing the prompt centre, else best IoU."""
    crop_x1, crop_y1, _, _ = crop_bounds
    original_x1, original_y1, original_x2, original_y2 = _box_xyxy(original_box, image_size)
    center_x = (original_x1 + original_x2) / 2 - crop_x1
    center_y = (original_y1 + original_y2) / 2 - crop_y1
    containing = [row for row in predictions if row[0] <= center_x <= row[2] and row[1] <= center_y <= row[3]]
    if containing:
        return max(containing, key=lambda row: float(row[4]))
    return _select_prediction_by_original_overlap(predictions, original_box, image_size, crop_bounds)


def crop_label_from_boxes(image_size, original_box, preferred_box, yolo_class_id=0):
    """Return review crop bounds and one crop-local YOLO label."""
    crop_x1, crop_y1, crop_x2, crop_y2 = review_crop_bounds(image_size, original_box)
    target_x1, target_y1, target_x2, target_y2 = _box_xyxy(preferred_box, image_size)
    crop_width, crop_height = crop_x2 - crop_x1, crop_y2 - crop_y1
    if not (crop_x1 <= target_x1 < target_x2 <= crop_x2 and crop_y1 <= target_y1 < target_y2 <= crop_y2):
        raise ValueError("Preferred box is not fully contained by the original-box review crop.")
    center_x = ((target_x1 + target_x2) / 2 - crop_x1) / crop_width
    center_y = ((target_y1 + target_y2) / 2 - crop_y1) / crop_height
    width = (target_x2 - target_x1) / crop_width
    height = (target_y2 - target_y1) / crop_height
    if not (0 < width <= 1 and 0 < height <= 1):
        raise ValueError("Preferred box has invalid crop-relative dimensions.")
    return (crop_x1, crop_y1, crop_x2, crop_y2), f"{int(yolo_class_id)} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"


def _project_tracking_path(project_dir):
    path = os.path.join(_normalise_path(project_dir), "user_selections", "tracking_review.xml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Tightener project `{project_dir}` is missing `user_selections/tracking_review.xml`.")
    return path


def _class_mapping_for_projects(project_dirs, class_aware):
    """Return stable YOLO-to-tracking class metadata for selected projects."""
    if not class_aware:
        return "single", [{"yolo_class_id": 0, "tracking_class_id": None, "name": "cell"}]
    expected_classes = None
    for project_dir in project_dirs:
        classes = read_tracking_xml(_project_tracking_path(project_dir)).get("classes", {})
        ordered = [(int(class_id), str(name)) for class_id, name in sorted(classes.items())]
        if not ordered:
            raise ValueError(f"Multiclass tightener project `{project_dir}` has no tracking class map.")
        if expected_classes is None:
            expected_classes = ordered
        elif ordered != expected_classes:
            raise ValueError("Multiclass tightener projects must share one identical ordered tracking class map.")
    return "multiclass", [
        {"yolo_class_id": index, "tracking_class_id": class_id, "name": name}
        for index, (class_id, name) in enumerate(expected_classes or [])
    ]


def summarise_projects(project_dirs):
    """Count valid crop examples and record invalid timepoints for project folders."""
    summary, total_valid, total_skipped = [], 0, 0
    for project_dir in project_dirs:
        project_dir = _normalise_path(project_dir)
        tracking_data = read_tracking_xml(_project_tracking_path(project_dir))
        valid = skipped = 0
        for track in tracking_data.get("tracks", []):
            for timepoint in track.get("timepoints", []):
                try:
                    original_box = _find_box(timepoint, "original")
                    preferred_box = _preferred_box(timepoint)
                    if original_box is None or preferred_box is None:
                        raise ValueError("Both original and preferred boxes are required.")
                    image_path = _normalise_path(timepoint["image_path"])
                    if not os.path.isfile(image_path):
                        raise FileNotFoundError(image_path)
                    with Image.open(image_path) as image:
                        crop_label_from_boxes(image.size, original_box, preferred_box)
                    valid += 1
                except (KeyError, OSError, ValueError, FileNotFoundError):
                    skipped += 1
        summary.append({"project_dir": project_dir, "valid": valid, "skipped": skipped})
        total_valid += valid
        total_skipped += skipped
    return {"projects": summary, "valid": total_valid, "skipped": total_skipped}


def recommended_imgsz(max_crop_side):
    """Round crop extent up to YOLO stride 32 and cap compact training at 320."""
    if max_crop_side <= 0:
        raise ValueError("Cannot choose an image size without valid training crops.")
    return min(320, max(32, ((int(max_crop_side) + 31) // 32) * 32))


def _validate_distinct_splits(train_dirs, val_dirs, test_dirs):
    split_dirs = {"train": train_dirs, "validation": val_dirs, "test": test_dirs}
    seen = {}
    for split_name, dirs in split_dirs.items():
        for directory in dirs:
            normalised = _normalise_path(directory)
            if normalised in seen:
                raise ValueError(f"Project `{normalised}` is in both {seen[normalised]} and {split_name} splits.")
            seen[normalised] = split_name


def _write_yaml(dataset_dir, class_mapping):
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        for split in ("train", "val", "test"):
            image_dir = os.path.join(dataset_dir, "images", split).replace("\\", "/")
            handle.write(f"{split}: {image_dir}\n")
        handle.write(f"nc: {len(class_mapping)}\nnames:\n")
        for item in class_mapping:
            handle.write(f"  {item['yolo_class_id']}: {item['name']}\n")
    return yaml_path


MANIFEST_FIELDS = [
    "split", "project_dir", "tracking_xml", "track_id", "timepoint_index", "image_path",
    "original_box_type", "preferred_box_type", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
    "yolo_label", "crop_path", "label_path", "status", "skip_reason",
    "target_tracking_class_id", "target_yolo_class_id", "target_class_name",
]


def _write_manifest(manifest_path, rows):
    """Write crop-level training provenance, including intentionally skipped rows."""
    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_tightener_dataset(train_dirs, val_dirs, test_dirs, dataset_dir, progress_callback=None, class_aware=False):
    """Create a self-contained crop dataset and return split counts and image size."""
    if not train_dirs or not val_dirs or not test_dirs:
        raise ValueError("Add at least one project folder to each train, validation, and test split.")
    _validate_distinct_splits(train_dirs, val_dirs, test_dirs)
    class_mode, class_mapping = _class_mapping_for_projects(train_dirs + val_dirs + test_dirs, class_aware)
    yolo_class_by_tracking_id = {item["tracking_class_id"]: item["yolo_class_id"] for item in class_mapping}
    class_name_by_tracking_id = {item["tracking_class_id"]: item["name"] for item in class_mapping}
    dataset_dir = _normalise_path(dataset_dir)
    if os.path.exists(dataset_dir):
        raise FileExistsError(f"Tightener dataset directory already exists: `{dataset_dir}`.")
    split_dirs = {"train": train_dirs, "val": val_dirs, "test": test_dirs}
    summaries = {name: summarise_projects(paths) for name, paths in split_dirs.items()}
    total = sum(item["valid"] for item in summaries.values())
    if total == 0:
        raise ValueError("No valid tightener examples were found in the selected projects.")
    staging_dir = f"{dataset_dir}.staging"
    if os.path.exists(staging_dir):
        raise FileExistsError(f"Tightener staging directory already exists: `{staging_dir}`.")
    current = max_crop_side = 0
    manifest_rows = []
    try:
        for split, project_dirs in split_dirs.items():
            image_output = os.path.join(staging_dir, "images", split)
            label_output = os.path.join(staging_dir, "labels", split)
            os.makedirs(image_output, exist_ok=True)
            os.makedirs(label_output, exist_ok=True)
            example_index = 0
            for project_dir in project_dirs:
                tracking_xml = _project_tracking_path(project_dir)
                tracking_data = read_tracking_xml(tracking_xml)
                for track in tracking_data.get("tracks", []):
                    for timepoint in track.get("timepoints", []):
                        manifest_row = {
                            "split": split, "project_dir": _normalise_path(project_dir), "tracking_xml": tracking_xml,
                            "track_id": track.get("track_id"), "timepoint_index": timepoint.get("timepoint_index"),
                            "image_path": timepoint.get("image_path"), "original_box_type": "original",
                            "preferred_box_type": timepoint.get("preferred_box_type"), "crop_x1": "", "crop_y1": "",
                            "crop_x2": "", "crop_y2": "", "yolo_label": "", "crop_path": "", "label_path": "",
                            "status": "skipped", "skip_reason": "",
                            "target_tracking_class_id": "", "target_yolo_class_id": "", "target_class_name": "",
                        }
                        try:
                            original_box = _find_box(timepoint, "original")
                            preferred_box = _preferred_box(timepoint)
                            image_path = _normalise_path(timepoint["image_path"])
                            if original_box is None or preferred_box is None:
                                raise ValueError("Missing source or target box")
                            tracking_class_id = int(timepoint["class_id"])
                            if class_mode == "multiclass" and tracking_class_id not in yolo_class_by_tracking_id:
                                raise ValueError(f"Class {tracking_class_id} is not in the multiclass mapping")
                            yolo_class_id = yolo_class_by_tracking_id.get(tracking_class_id, 0)
                            with Image.open(image_path) as image:
                                crop_bounds, label = crop_label_from_boxes(image.size, original_box, preferred_box, yolo_class_id)
                                crop = image.crop(crop_bounds)
                                max_crop_side = max(max_crop_side, *crop.size)
                                filename = f"{split}_{example_index:06d}.png"
                                crop_path = os.path.join(image_output, filename)
                                crop.save(crop_path)
                            label_path = os.path.join(label_output, filename[:-4] + ".txt")
                            with open(label_path, "w", encoding="utf-8") as handle:
                                handle.write(label + "\n")
                            manifest_row.update({
                                "image_path": image_path, "crop_x1": crop_bounds[0], "crop_y1": crop_bounds[1],
                                "crop_x2": crop_bounds[2], "crop_y2": crop_bounds[3], "yolo_label": label,
                                "crop_path": os.path.relpath(crop_path, staging_dir).replace("\\", "/"),
                                "label_path": os.path.relpath(label_path, staging_dir).replace("\\", "/"), "status": "included",
                                "target_tracking_class_id": tracking_class_id if class_mode == "multiclass" else "",
                                "target_yolo_class_id": yolo_class_id,
                                "target_class_name": class_name_by_tracking_id.get(tracking_class_id, "cell"),
                            })
                            example_index += 1
                            current += 1
                            if progress_callback:
                                progress_callback(current, total, image_path)
                        except (KeyError, OSError, ValueError, FileNotFoundError) as exc:
                            manifest_row["skip_reason"] = f"{type(exc).__name__}: {exc}"
                        manifest_rows.append(manifest_row)
        _write_manifest(os.path.join(staging_dir, "training_manifest.csv"), manifest_rows)
        os.makedirs(os.path.dirname(dataset_dir), exist_ok=True)
        os.replace(staging_dir, dataset_dir)
        _write_yaml(dataset_dir, class_mapping)
    except Exception:
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)
        raise
    return {"dataset_dir": dataset_dir, "yaml_path": os.path.join(dataset_dir, "dataset.yaml"),
            "manifest_path": os.path.join(dataset_dir, "training_manifest.csv"), "splits": summaries,
            "imgsz": recommended_imgsz(max_crop_side), "max_crop_side": max_crop_side,
            "class_mode": class_mode, "class_mapping": class_mapping}


def _read_last_results_row(path):
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8", newline="") as handle:
        return next(reversed(list(csv.DictReader(handle))), {})


def _write_model_metadata(run_dir, dataset_info):
    """Write class provenance beside weights for later project configuration."""
    os.makedirs(run_dir, exist_ok=True)
    metadata_path = os.path.join(run_dir, MODEL_METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"class_mode": dataset_info["class_mode"], "class_mapping": dataset_info["class_mapping"], "imgsz": dataset_info["imgsz"]},
            handle,
            indent=2,
        )
    return metadata_path


def _read_model_metadata(weights_path):
    """Read tightener provenance stored beside an in-project ``best.pt``."""
    metadata_path = os.path.join(os.path.dirname(os.path.dirname(weights_path)), MODEL_METADATA_FILENAME)
    if not os.path.isfile(metadata_path):
        return None
    # PowerShell's UTF-8 output may include a BOM; accept both UTF-8 variants.
    with open(metadata_path, encoding="utf-8-sig") as handle:
        metadata = json.load(handle)
    class_mode = metadata.get("class_mode")
    class_mapping = metadata.get("class_mapping")
    if class_mode not in {"single", "multiclass"} or not isinstance(class_mapping, list):
        raise ValueError(f"Tightener model metadata `{metadata_path}` is invalid.")
    if class_mode == "multiclass" and not class_mapping:
        raise ValueError(f"Tightener model metadata `{metadata_path}` has no multiclass mapping.")
    return metadata


def _legacy_single_model_metadata(weights_path):
    """Recognise earlier one-class tightener runs that predate provenance JSON."""
    args_path = os.path.join(os.path.dirname(os.path.dirname(weights_path)), "args.yaml")
    if not os.path.isfile(args_path):
        return None
    dataset_path = None
    with open(args_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("data:"):
                dataset_path = line.split(":", 1)[1].strip().strip("'\"")
                break
    if not dataset_path or not os.path.isfile(dataset_path):
        return None
    with open(dataset_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("nc:") and line.split(":", 1)[1].strip() == "1":
                return {"class_mode": "single", "class_mapping": [{"yolo_class_id": 0, "tracking_class_id": None, "name": "cell"}]}
    return None


def train_tightener_model(train_dirs, val_dirs, test_dirs, dataset_dir, output_root, run_name, epochs, batch, patience, device, progress_callback=None, epoch_callback=None, dataset_callback=None, class_aware=False):
    """Prepare crops, train YOLO11n, then evaluate its best checkpoint on test."""
    YOLO = _ensure_ultralytics()
    dataset_info = prepare_tightener_dataset(train_dirs, val_dirs, test_dirs, dataset_dir, progress_callback, class_aware=class_aware)
    if dataset_callback:
        dataset_callback(dataset_info)
    output_root = _normalise_path(output_root)
    run_dir = os.path.join(output_root, run_name)
    if os.path.exists(run_dir):
        raise FileExistsError(f"Tightener training run already exists: `{run_dir}`.")
    os.makedirs(DEFAULT_TIGHTENER_MODELS_ROOT, exist_ok=True)
    model = YOLO(os.path.join(DEFAULT_TIGHTENER_MODELS_ROOT, "yolo11n.pt"))
    if epoch_callback:
        def report_epoch(trainer):
            metrics = getattr(trainer, "metrics", {}) or {}
            epoch_callback(int(getattr(trainer, "epoch", -1)) + 1, int(getattr(trainer, "epochs", epochs)), dict(metrics))
        model.add_callback("on_fit_epoch_end", report_epoch)
    results = model.train(data=dataset_info["yaml_path"], project=output_root, name=run_name, imgsz=dataset_info["imgsz"],
                          batch=batch, epochs=epochs, patience=patience, device=device)
    save_dir = _normalise_path(str(getattr(results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", run_dir)))
    best_weights = os.path.join(save_dir, "weights", "best.pt")
    model_metadata_path = _write_model_metadata(save_dir, dataset_info)
    test_model = YOLO(best_weights)
    test_results = test_model.val(data=dataset_info["yaml_path"], split="test", imgsz=dataset_info["imgsz"], device=device)
    return {"dataset_info": dataset_info, "run_dir": save_dir, "best_weights": best_weights,
            "last_weights": os.path.join(save_dir, "weights", "last.pt"), "validation_metrics": _read_last_results_row(os.path.join(save_dir, "results.csv")),
            "test_metrics": dict(getattr(test_results, "results_dict", {}) or {}), "model_metadata_path": model_metadata_path,
            "timestamp": datetime.now().isoformat(timespec="seconds")}


def configure_tightener_weights(tracking_xml_path, weights_path, selection_strategy=DEFAULT_TIGHTENER_SELECTION):
    """Persist the model used for this project's future tightener generation."""
    weights_path = _normalise_path(weights_path)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Tightener weights do not exist: `{weights_path}`.")
    if selection_strategy not in {"center_confidence", "overlap", "confidence"}:
        raise ValueError("Tightener selection strategy must be `center_confidence`, `overlap`, or `confidence`.")
    model_metadata = _read_model_metadata(weights_path) or _legacy_single_model_metadata(weights_path)
    if model_metadata is None:
        raise ValueError(
            "Tightener weights are missing model provenance. Configure only weights produced by Cell Tightener Training."
        )
    data = read_tracking_xml(tracking_xml_path)
    if model_metadata["class_mode"] == "multiclass":
        project_mapping = [
            {"yolo_class_id": index, "tracking_class_id": int(class_id), "name": str(name)}
            for index, (class_id, name) in enumerate(sorted(data.get("classes", {}).items()))
        ]
        if project_mapping != model_metadata["class_mapping"]:
            raise ValueError("Multiclass tightener weights do not match this project's tracking class map.")
    metadata = data.setdefault("metadata", {})
    metadata[TIGHTENER_MODEL_METADATA_KEY] = weights_path
    metadata[TIGHTENER_IMGSZ_METADATA_KEY] = str(model_metadata.get("imgsz", _trained_model_imgsz(weights_path)))
    metadata[TIGHTENER_SELECTION_METADATA_KEY] = selection_strategy
    metadata[TIGHTENER_CLASS_MODE_METADATA_KEY] = model_metadata["class_mode"]
    metadata[TIGHTENER_CLASS_MAPPING_METADATA_KEY] = json.dumps(model_metadata["class_mapping"])
    write_tracking_data(tracking_xml_path, data)


def run_tightener_on_tracking_xml(tracking_xml_path, overwrite=True, confidence=0.05, device=None, progress_callback=None):
    """Generate ``yolo11_tightened`` variants using this project's configured model."""
    data = read_tracking_xml(tracking_xml_path)
    weights_path = data.get("metadata", {}).get(TIGHTENER_MODEL_METADATA_KEY)
    if not weights_path:
        raise ValueError("Configure Cell Tightener Model before running the tightener.")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Configured tightener weights do not exist: `{weights_path}`.")
    imgsz = int(data.get("metadata", {}).get(TIGHTENER_IMGSZ_METADATA_KEY, _trained_model_imgsz(weights_path)))
    if imgsz <= 0:
        raise ValueError("Configured cell tightener image size must be positive.")
    selection_strategy = data.get("metadata", {}).get(TIGHTENER_SELECTION_METADATA_KEY, DEFAULT_TIGHTENER_SELECTION)
    if selection_strategy not in {"center_confidence", "overlap", "confidence"}:
        raise ValueError("Configured cell tightener selection strategy must be `center_confidence`, `overlap`, or `confidence`.")
    class_mode = data.get("metadata", {}).get(TIGHTENER_CLASS_MODE_METADATA_KEY, "single")
    if class_mode not in {"single", "multiclass"}:
        raise ValueError("Configured cell tightener class mode must be `single` or `multiclass`.")
    try:
        class_mapping = json.loads(data.get("metadata", {}).get(TIGHTENER_CLASS_MAPPING_METADATA_KEY, "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("Configured cell tightener class mapping is invalid.") from exc
    class_by_yolo_id = {int(item["yolo_class_id"]): item for item in class_mapping}
    if class_mode == "multiclass" and not class_by_yolo_id:
        raise ValueError("Configured multiclass tightener has no class mapping.")
    model = _ensure_ultralytics()(weights_path)
    data.setdefault("box_types", {})[TIGHTENER_BOX_TYPE] = "Box adjusted by a trained YOLO11 cell tightener."
    timepoints = [point for track in data.get("tracks", []) for point in track.get("timepoints", [])]
    stats = {"timepoints": len(timepoints), "created": 0, "updated": 0, "fallback_original": 0, "skipped_existing": 0, "failed": 0}
    for index, timepoint in enumerate(timepoints, start=1):
        if progress_callback:
            progress_callback(index, len(timepoints), timepoint.get("image_path"))
        existing = _find_box(timepoint, TIGHTENER_BOX_TYPE)
        if existing and not overwrite:
            stats["skipped_existing"] += 1
            continue
        try:
            original = _find_box(timepoint, "original")
            if original is None:
                raise ValueError("Original box is required for tightener inference.")
            image_path = _normalise_path(timepoint["image_path"])
            with Image.open(image_path) as image:
                bounds = review_crop_bounds(image.size, original)
                crop = image.crop(bounds)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                    crop_path = temp.name
                try:
                    crop.save(crop_path)
                    result = model(crop_path, imgsz=imgsz, conf=confidence, device=device)[0]
                finally:
                    if os.path.exists(crop_path):
                        os.remove(crop_path)
                boxes = getattr(result, "boxes", None)
                values = getattr(boxes, "data", None) if boxes is not None else None
                if values is None or len(values) == 0:
                    predicted = None
                else:
                    rows = values.cpu().numpy() if hasattr(values, "cpu") else values
                    if selection_strategy == "center_confidence":
                        predicted = _select_prediction_by_center_confidence(rows, original, image.size, bounds)
                    elif selection_strategy == "overlap":
                        predicted = _select_prediction_by_original_overlap(rows, original, image.size, bounds)
                    else:
                        predicted = max(rows, key=lambda row: float(row[4]))
                if predicted is None:
                    box = dict(original)
                    box["box_type"] = TIGHTENER_BOX_TYPE
                    box["source"] = f"yolo11_tightener_fallback:{weights_path}"
                    stats["fallback_original"] += 1
                else:
                    x1, y1, x2, y2 = [float(value) for value in predicted[:4]]
                    crop_width, crop_height = crop.size
                    x1, x2 = max(0.0, min(x1, crop_width)), max(0.0, min(x2, crop_width))
                    y1, y2 = max(0.0, min(y1, crop_height)), max(0.0, min(y2, crop_height))
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError("Tightener prediction has no area within its crop.")
                    crop_x1, crop_y1, _, _ = bounds
                    full_x1, full_y1, full_x2, full_y2 = x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1
                    image_width, image_height = image.size
                    box = {"box_type": TIGHTENER_BOX_TYPE, "format": "yolo_xywh_norm",
                           "x_center": ((full_x1 + full_x2) / 2) / image_width, "y_center": ((full_y1 + full_y2) / 2) / image_height,
                           "width": (full_x2 - full_x1) / image_width, "height": (full_y2 - full_y1) / image_height,
                           "source": f"yolo11_tightener:{weights_path}"}
                    if class_mode == "multiclass":
                        predicted_class = class_by_yolo_id.get(int(predicted[5]))
                        if predicted_class is None:
                            raise ValueError("Tightener prediction returned a class outside its configured mapping.")
                        box["predicted_class_id"] = int(predicted_class["tracking_class_id"])
                        box["predicted_class_name"] = str(predicted_class["name"])
            if existing:
                timepoint["boxes"][timepoint["boxes"].index(existing)] = box
                stats["updated"] += 1
            else:
                timepoint.setdefault("boxes", []).append(box)
                stats["created"] += 1
        except Exception:
            stats["failed"] += 1
    write_tracking_data(tracking_xml_path, data)
    return stats
