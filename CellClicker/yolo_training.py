"""Configuration-driven preparation and Ultralytics YOLO training.

This module contains no GUI imports so the same workflow can be used from the
CellClicker interface and from headless command-line installations.
"""

import csv
from copy import deepcopy
from datetime import datetime
import os
import shutil
import tempfile

import yaml

from .tracking_export import exported_labels_are_current
from .tracking_xml import read_tracking_xml


TRAINING_CONFIG_SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_CLASS_NAMES = [
    "prophase",
    "earlyprometaphase",
    "prometaphase",
    "metaphase",
    "anaphase",
    "telophase",
]


def _normalize_path(path):
    return os.path.normpath(os.fspath(path))


def _resolve_config_path(path, config_directory):
    path = os.path.expanduser(os.fspath(path))
    if not os.path.isabs(path):
        path = os.path.join(config_directory, path)
    return _normalize_path(os.path.abspath(path))


def _yaml_path(path):
    return _normalize_path(path).replace("\\", "/")


def _require_mapping(value, section_name):
    if not isinstance(value, dict):
        raise ValueError(f"Training config section `{section_name}` must be a mapping.")
    return value


def _require_nonempty_text(value, field_name):
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value).strip():
        raise ValueError(f"Training config field `{field_name}` must be a non-empty string.")
    return os.fspath(value).strip()


def _require_positive_int(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"Training config field `{field_name}` must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Training config field `{field_name}` must be a positive integer.") from exc
    if integer <= 0 or isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Training config field `{field_name}` must be a positive integer.")
    return integer


def _validate_distinct_splits(train_dirs, val_dirs, test_dirs):
    """Reject projects that would leak between training, validation, and test."""
    seen = {}
    for split_name, project_dirs in (("training", train_dirs), ("validation", val_dirs), ("test", test_dirs)):
        for project_dir in project_dirs:
            normalized_project_dir = os.path.normcase(os.path.abspath(_normalize_path(project_dir)))
            if normalized_project_dir in seen:
                raise ValueError(
                    f"Project `{_normalize_path(project_dir)}` is in both "
                    f"{seen[normalized_project_dir]} and {split_name} splits."
                )
            seen[normalized_project_dir] = split_name


def validate_training_config(config, config_directory=None, check_paths=True):
    """Validate and resolve one schema-versioned training configuration.

    Relative filesystem paths are resolved from ``config_directory``. The
    returned mapping is an independent, normalized configuration suitable for
    :func:`run_training_config`.
    """
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping.")
    if config.get("schema_version") != TRAINING_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Training config must declare `schema_version: {TRAINING_CONFIG_SCHEMA_VERSION}`."
        )

    config_directory = _normalize_path(config_directory or os.getcwd())
    splits = _require_mapping(config.get("splits"), "splits")
    model = _require_mapping(config.get("model"), "model")
    run = _require_mapping(config.get("run"), "run")
    training = _require_mapping(config.get("training"), "training")

    resolved_splits = {}
    for split_name in ("train", "val", "test"):
        projects = splits.get(split_name)
        if not isinstance(projects, list) or not projects:
            raise ValueError(f"Training config split `splits.{split_name}` must contain at least one project path.")
        resolved_splits[split_name] = [
            _resolve_config_path(_require_nonempty_text(path, f"splits.{split_name}"), config_directory)
            for path in projects
        ]

    model_path = _resolve_config_path(
        _require_nonempty_text(model.get("path"), "model.path"), config_directory
    )
    output_root = _resolve_config_path(
        _require_nonempty_text(run.get("output_root"), "run.output_root"), config_directory
    )
    run_name = _require_nonempty_text(run.get("name"), "run.name")
    if os.path.basename(run_name) != run_name or run_name in {".", ".."}:
        raise ValueError("Training config field `run.name` must be a single directory name.")

    device = training.get("device")
    if isinstance(device, bool) or not isinstance(device, (str, int)) or not str(device).strip():
        raise ValueError("Training config field `training.device` must be a non-empty string or integer.")

    _validate_distinct_splits(
        resolved_splits["train"], resolved_splits["val"], resolved_splits["test"]
    )
    if check_paths:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Training model does not exist: `{model_path}`.")
        for split_name, project_dirs in resolved_splits.items():
            for project_dir in project_dirs:
                if not os.path.isdir(project_dir):
                    raise FileNotFoundError(
                        f"Training config split `{split_name}` project does not exist: `{project_dir}`."
                    )

    return {
        "schema_version": TRAINING_CONFIG_SCHEMA_VERSION,
        "splits": resolved_splits,
        "model": {"path": model_path},
        "run": {"output_root": output_root, "name": run_name},
        "training": {
            "epochs": _require_positive_int(training.get("epochs"), "training.epochs"),
            "imgsz": _require_positive_int(training.get("imgsz"), "training.imgsz"),
            "batch": _require_positive_int(training.get("batch"), "training.batch"),
            "patience": _require_positive_int(training.get("patience"), "training.patience"),
            "device": str(device).strip(),
        },
    }


def load_training_config(config_path, check_paths=True):
    """Load, validate, and resolve a YOLO training YAML file."""
    config_path = os.path.abspath(_normalize_path(config_path))
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Training config is not valid YAML: `{config_path}`.") from exc
    return validate_training_config(
        config, config_directory=os.path.dirname(config_path), check_paths=check_paths
    )


def _portable_path(path, config_directory):
    try:
        return _yaml_path(os.path.relpath(path, config_directory))
    except ValueError:
        # Windows cannot make paths on different drives relative.
        return _yaml_path(path)


def _serializable_training_config(config, config_directory, relative_paths):
    serializable = deepcopy(config)
    path_value = (
        (lambda value: _portable_path(value, config_directory))
        if relative_paths
        else _yaml_path
    )
    for split_name in ("train", "val", "test"):
        serializable["splits"][split_name] = [
            path_value(path) for path in config["splits"][split_name]
        ]
    serializable["model"]["path"] = path_value(config["model"]["path"])
    serializable["run"]["output_root"] = path_value(config["run"]["output_root"])
    return serializable


def write_training_config(config_path, config, relative_paths=True):
    """Validate and write a training config, using portable paths when possible."""
    config_path = os.path.abspath(_normalize_path(config_path))
    config_directory = os.path.dirname(config_path)
    normalized = validate_training_config(config, config_directory=os.getcwd())
    os.makedirs(config_directory, exist_ok=True)
    serializable = _serializable_training_config(normalized, config_directory, relative_paths)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)
    return config_path


def _ensure_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for YOLO training. Install `ultralytics` in the active environment."
        ) from exc
    return YOLO


def _iter_image_files(images_dir):
    for dirpath, _, filenames in os.walk(images_dir):
        for filename in filenames:
            suffix = os.path.splitext(filename)[1].lower()
            if suffix in IMAGE_EXTENSIONS:
                yield os.path.join(dirpath, filename)


def _project_key(project_dir):
    return os.path.basename(_normalize_path(os.fspath(project_dir).rstrip("\\/")))


def _load_project_classes(project_dir):
    """Read ordered class names from a project's tracking XML."""
    tracking_xml = os.path.join(project_dir, "user_selections", "tracking_review.xml")
    if not os.path.exists(tracking_xml):
        raise FileNotFoundError(
            f"Training project `{project_dir}` is missing `user_selections/tracking_review.xml`."
        )
    tracking_data = read_tracking_xml(tracking_xml)
    classes = tracking_data.get("classes", {})
    if not classes:
        return DEFAULT_CLASS_NAMES
    return [str(classes[class_id]) for class_id in sorted(classes, key=lambda value: int(value))]


def _resolve_project_pairs(project_dir):
    """Return image/label pairs whose image-relative paths match exactly."""
    project_dir = _normalize_path(project_dir)
    images_dir = os.path.join(project_dir, "images")
    labels_dir = os.path.join(project_dir, "user_selections", "exported_labels")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Training project `{project_dir}` is missing `images/`.")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(
            f"Training project `{project_dir}` is missing `user_selections/exported_labels/`. Export labels first."
        )
    pairs = []
    for image_path in sorted(_iter_image_files(images_dir)):
        relative_image = os.path.relpath(image_path, images_dir)
        relative_label = os.path.splitext(relative_image)[0] + ".txt"
        label_path = os.path.join(labels_dir, relative_label)
        if os.path.exists(label_path):
            pairs.append((_normalize_path(image_path), _normalize_path(label_path), relative_image))
    return pairs


def _summarize_projects(project_dirs):
    summary = []
    class_names = None
    total_pairs = 0
    for project_dir in project_dirs:
        pairs = _resolve_project_pairs(project_dir)
        if pairs and not exported_labels_are_current(project_dir):
            raise ValueError(
                f"Training project `{project_dir}` has stale exported labels. Rebuild YOLO labels from tracking review first."
            )
        project_classes = _load_project_classes(project_dir)
        if class_names is None:
            class_names = project_classes
        elif class_names != project_classes:
            raise ValueError("Selected projects do not share the same class list in `tracking_review.xml`.")
        summary.append({
            "project_dir": _normalize_path(project_dir),
            "project_name": _project_key(project_dir),
            "count": len(pairs),
            "pairs": pairs,
        })
        total_pairs += len(pairs)
    return summary, (class_names or list(DEFAULT_CLASS_NAMES)), total_pairs


def _write_dataset_yaml(yaml_path, train_image_dirs, val_image_dirs, test_image_dirs, class_names):
    lines = ["train:"]
    for train_dir in train_image_dirs:
        lines.append(f"  - {_yaml_path(train_dir)}")
    lines.append("val:")
    for val_dir in val_image_dirs:
        lines.append(f"  - {_yaml_path(val_dir)}")
    lines.append("test:")
    for test_dir in test_image_dirs:
        lines.append(f"  - {_yaml_path(test_dir)}")
    lines.extend([f"nc: {len(class_names)}", "names:"])
    for index, class_name in enumerate(class_names):
        lines.append(f"  {index}: {class_name}")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _sync_project_labels(project_dir, progress_callback=None, current_offset=0, total_items=1):
    project_dir = _normalize_path(project_dir)
    images_dir = os.path.join(project_dir, "images")
    exported_labels_dir = os.path.join(project_dir, "user_selections", "exported_labels")
    labels_dir = os.path.join(project_dir, "labels")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Training project `{project_dir}` is missing `images/`.")
    if not os.path.isdir(exported_labels_dir):
        raise FileNotFoundError(
            f"Training project `{project_dir}` is missing `user_selections/exported_labels/`. Export labels first."
        )
    pairs = _resolve_project_pairs(project_dir)
    if not pairs:
        raise ValueError(
            f"Training project `{project_dir}` has no images with exported labels. Export labels first."
        )
    if not exported_labels_are_current(project_dir):
        raise ValueError(
            f"Training project `{project_dir}` has stale exported labels. Rebuild YOLO labels from tracking review first."
        )

    staging_dir = tempfile.mkdtemp(prefix="celfdrive-labels-", dir=project_dir)
    copied_successfully = False
    try:
        for index, (_, label_path, relative_image) in enumerate(pairs, start=1):
            if progress_callback is not None:
                progress_callback(current_offset + index, total_items, label_path)
            relative_label = os.path.splitext(relative_image)[0] + ".txt"
            target_label_path = os.path.join(staging_dir, relative_label)
            os.makedirs(os.path.dirname(target_label_path), exist_ok=True)
            shutil.copy2(label_path, target_label_path)
        if os.path.isdir(labels_dir):
            shutil.rmtree(labels_dir)
        os.replace(staging_dir, labels_dir)
        copied_successfully = True
    finally:
        if not copied_successfully and os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)
    return len(pairs)


def prepare_yolo_sources(train_dirs, val_dirs, test_dirs, yaml_path, progress_callback=None):
    """Synchronize project labels and write an Ultralytics dataset YAML file."""
    if not train_dirs or not val_dirs or not test_dirs:
        raise ValueError("Add at least one project to each training, validation, and test split.")
    _validate_distinct_splits(train_dirs, val_dirs, test_dirs)
    train_summary, class_names, train_count = _summarize_projects(train_dirs)
    val_summary, val_class_names, val_count = _summarize_projects(val_dirs)
    test_summary, test_class_names, test_count = _summarize_projects(test_dirs)
    if class_names != val_class_names or class_names != test_class_names:
        raise ValueError("Training, validation, and test projects do not share the same class list.")

    total_items = train_count + val_count + test_count
    current = 0
    image_dirs = {"train": [], "val": [], "test": []}
    for split_name, split_summary in (("train", train_summary), ("val", val_summary), ("test", test_summary)):
        for project in split_summary:
            copied = _sync_project_labels(
                project["project_dir"], progress_callback=progress_callback,
                current_offset=current, total_items=total_items,
            )
            current += copied
            image_dirs[split_name].append(os.path.join(project["project_dir"], "images"))

    yaml_path = _normalize_path(yaml_path)
    yaml_directory = os.path.dirname(yaml_path)
    if yaml_directory:
        os.makedirs(yaml_directory, exist_ok=True)
    _write_dataset_yaml(yaml_path, image_dirs["train"], image_dirs["val"], image_dirs["test"], class_names)
    return {
        "yaml_path": yaml_path,
        "class_names": class_names,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "train_image_dirs": image_dirs["train"],
        "val_image_dirs": image_dirs["val"],
        "test_image_dirs": image_dirs["test"],
    }


def _read_last_results_row(results_csv):
    if not os.path.exists(results_csv):
        return {}
    with open(results_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        last_row = None
        for row in reader:
            last_row = {key.strip(): value.strip() for key, value in row.items() if key is not None}
    return last_row or {}


def _append_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_yolo_model(
    train_dirs, val_dirs, test_dirs, output_root, run_name, model_path,
    epochs, imgsz, batch, patience, device, progress_callback=None,
):
    """Prepare projects, train, evaluate the best checkpoint, and summarize."""
    YOLO = _ensure_ultralytics()
    output_root = _normalize_path(output_root)
    run_dir = os.path.join(output_root, run_name)
    yaml_path = os.path.join(output_root, "_yamls", f"{run_name}.yaml")
    if os.path.exists(run_dir):
        raise FileExistsError(f"Training run directory already exists: `{run_dir}`.")
    if os.path.exists(yaml_path):
        raise FileExistsError(f"Training YAML already exists: `{yaml_path}`.")

    dataset_info = prepare_yolo_sources(
        train_dirs, val_dirs, test_dirs, yaml_path, progress_callback=progress_callback
    )
    model = YOLO(model_path)
    train_results = model.train(
        data=dataset_info["yaml_path"], project=output_root, name=run_name,
        imgsz=imgsz, batch=batch, epochs=epochs, patience=patience, device=device,
    )
    save_dir = getattr(train_results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is None:
        save_dir = run_dir
    save_dir = _normalize_path(str(save_dir))

    results_csv = os.path.join(save_dir, "results.csv")
    metrics = _read_last_results_row(results_csv)
    best_weights = os.path.join(save_dir, "weights", "best.pt")
    test_results = YOLO(best_weights).val(
        data=dataset_info["yaml_path"], split="test", imgsz=imgsz, device=device,
    )
    test_metrics = dict(getattr(test_results, "results_dict", {}) or {})
    summary_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "model_path": _normalize_path(model_path),
        "epochs": int(epochs), "imgsz": int(imgsz), "batch": int(batch),
        "patience": int(patience), "device": str(device),
        "dataset_yaml": _normalize_path(dataset_info["yaml_path"]),
        "train_images": int(dataset_info["train_count"]),
        "val_images": int(dataset_info["val_count"]),
        "test_images": int(dataset_info["test_count"]),
        "train_projects": ";".join(project["project_dir"] for project in dataset_info["train_summary"]),
        "val_projects": ";".join(project["project_dir"] for project in dataset_info["val_summary"]),
        "test_projects": ";".join(project["project_dir"] for project in dataset_info["test_summary"]),
        "train_image_dirs": ";".join(dataset_info["train_image_dirs"]),
        "val_image_dirs": ";".join(dataset_info["val_image_dirs"]),
        "test_image_dirs": ";".join(dataset_info["test_image_dirs"]),
        "class_names": ";".join(dataset_info["class_names"]),
        "run_dir": save_dir,
        "best_weights": best_weights,
        "last_weights": os.path.join(save_dir, "weights", "last.pt"),
        "results_csv": results_csv,
        "results_png": os.path.join(save_dir, "results.png"),
        "confusion_matrix_png": os.path.join(save_dir, "confusion_matrix.png"),
        "confusion_matrix_normalized_png": os.path.join(save_dir, "confusion_matrix_normalized.png"),
        "pr_curve_png": os.path.join(save_dir, "PR_curve.png"),
        "p_curve_png": os.path.join(save_dir, "P_curve.png"),
        "r_curve_png": os.path.join(save_dir, "R_curve.png"),
        "f1_curve_png": os.path.join(save_dir, "F1_curve.png"),
    }
    summary_row.update(metrics)
    summary_row.update({f"test_{name}": value for name, value in test_metrics.items()})
    run_summary_csv = os.path.join(save_dir, "training_summary.csv")
    global_summary_csv = os.path.join(output_root, "training_runs.csv")
    _append_csv_row(run_summary_csv, summary_row)
    _append_csv_row(global_summary_csv, summary_row)
    return {
        "dataset_info": dataset_info, "run_dir": save_dir,
        "summary_csv": run_summary_csv, "global_summary_csv": global_summary_csv,
        "summary_row": summary_row, "test_metrics": test_metrics,
    }


def run_training_config(config, progress_callback=None):
    """Run one validated configuration and save its resolved run snapshot."""
    config = validate_training_config(config)
    result = train_yolo_model(
        train_dirs=config["splits"]["train"],
        val_dirs=config["splits"]["val"],
        test_dirs=config["splits"]["test"],
        output_root=config["run"]["output_root"],
        run_name=config["run"]["name"],
        model_path=config["model"]["path"],
        epochs=config["training"]["epochs"],
        imgsz=config["training"]["imgsz"],
        batch=config["training"]["batch"],
        patience=config["training"]["patience"],
        device=config["training"]["device"],
        progress_callback=progress_callback,
    )
    snapshot_path = os.path.join(result["run_dir"], "training_config.yaml")
    write_training_config(snapshot_path, config, relative_paths=False)
    result["training_config"] = snapshot_path
    return result
