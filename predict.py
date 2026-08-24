"""Predict cell states in overview montages and produce capture targets.

The named public API is :func:`get_target_locations`. Montages use NumPy axis
order ``(height, width, position)``; image X is the column axis and image Y is
the row axis. Pixel spacing, stage positions, and Z offsets are micrometres.
"""

from pathlib import Path
import re
from datetime import datetime
import contextlib
import math
import os
from time import perf_counter
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Real
from typing import Optional

import numpy as np
import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "celfdrive_predict.yaml"
CONFIG_REF_PATTERN = re.compile(r"\$\{([^}]+)\}")
SUPPORTED_BACKENDS = {"ultralytics_yolo"}


# Runtime state

@dataclass
class PredictionRuntime:
    """Mutable resources belonging to one prediction workflow invocation.

    Keeping these resources together prevents one microscope workflow from
    changing another workflow's model, configuration, or logging directory.
    """

    config: Optional[dict] = None
    model: Optional[object] = None
    experiment_path: Optional[Path] = None
    last_timings: Optional["PredictionTimings"] = None


@dataclass
class PredictionTimings:
    """Wall-clock durations accumulated for one target-selection callback.

    Durations are in seconds and include every overview plane in a montage.
    ``inference_s`` synchronizes CUDA before and after each model invocation so
    it represents completed GPU work rather than merely queued CUDA kernels.
    """

    preprocessing_s: float = 0.0
    inference_s: float = 0.0
    postprocessing_s: float = 0.0
    logging_s: float = 0.0
    total_s: float = 0.0

    def format_report(self):
        """Return the stable, single-line timing report used by acquisition logs."""
        return (
            "CelFDrive timing (s): "
            f"preprocessing={self.preprocessing_s:.3f}, "
            f"inference={self.inference_s:.3f}, "
            f"postprocessing={self.postprocessing_s:.3f}, "
            f"logging={self.logging_s:.3f}, "
            f"total={self.total_s:.3f}"
        )


_runtime: ContextVar[Optional[PredictionRuntime]] = ContextVar("prediction_runtime", default=None)


@dataclass(frozen=True)
class Detection:
    """A model detection in image pixel coordinates."""

    class_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def centre_x(self):
        return self.x + self.width / 2

    @property
    def centre_y(self):
        return self.y + self.height / 2

@dataclass(frozen=True)
class CapturePosition:
    """The physical stage position associated with one overview image."""

    x: float
    y: float
    z: float

# Configuration

def _get_nested_value(data, dotted_key):
    value = data
    for part in dotted_key.split("."):
        value = value[part]
    return value

def _expand_config_value(value, root_config):
    if isinstance(value, str):
        def replace_match(match):
            referenced_value = _get_nested_value(root_config, match.group(1))
            return str(_expand_config_value(referenced_value, root_config))

        return CONFIG_REF_PATTERN.sub(replace_match, value)
    if isinstance(value, dict):
        return {key: _expand_config_value(item, root_config) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_value(item, root_config) for item in value]
    return value

def migrate_predict_config(raw_config):
    """Migrate a schema-version-1 configuration mapping in place for legacy keys."""
    if "profile" not in raw_config and "profiles" in raw_config:
        raw_config["profile"] = raw_config["profiles"].get("sdc", next(iter(raw_config["profiles"].values())))
        raw_config.pop("profiles", None)
        raw_config["profile"].pop("llsm", None)
    no_detection = raw_config.get("no_detection", {})
    if "mode" not in no_detection:
        if no_detection.get("return_original_first_position", True):
            no_detection["mode"] = "empty_3i_capture_script"
        else:
            no_detection["mode"] = "end_workflow"
    if no_detection.get("mode") == "do_nothing":
        no_detection["mode"] = "end_workflow"
    if "empty_3i_capture_script" not in no_detection:
        no_detection["empty_3i_capture_script"] = no_detection.get("script", "donothing")
    for key in ["n_returned_locations", "script", "name", "comment", "return_original_first_position"]:
        no_detection.pop(key, None)
    coordinate_conversion = raw_config.setdefault("coordinate_conversion", {})
    coordinate_conversion.setdefault("mode", "stage")
    coordinate_conversion.setdefault("default_z_offset_um", 0.0)
    coordinate_conversion.setdefault("merge_tolerance_um", 20.0)
    slidebook = raw_config.setdefault("slidebook", {})
    slidebook.setdefault("objective_offset_um", {"x": 0.0, "y": 0.0, "z": 0.0})
    tiling = raw_config.setdefault("tiling", {})
    legacy_tiling_enabled = tiling.pop("enabled", True)
    tiling.setdefault("overlap_px", 0)
    tiling.setdefault("deduplication_tolerance_px", 1.0)
    logging = raw_config.setdefault("logging", {})
    prediction_images = logging.setdefault("prediction_images", {})
    legacy_logging_enabled = logging.pop("enabled", True)
    legacy_plotting_enabled = raw_config.setdefault("plotting", {}).pop("enabled", True)
    prediction_images.setdefault("enabled", legacy_logging_enabled and legacy_plotting_enabled)
    timing = logging.setdefault("timing", {})
    timing.setdefault("enabled", True)
    inference = raw_config.setdefault("inference", {})
    if inference.get("mode", "standard") == "standard":
        inference["mode"] = "tiling" if legacy_tiling_enabled else "full_image"
    else:
        inference.setdefault("mode", "tiling")
    sahi = inference.setdefault("sahi", {})
    sahi.setdefault("confidence_threshold", 0.5)
    sahi.setdefault("slice_size_px", 640)
    sahi.setdefault("overlap_ratio", 0.25)
    sahi.setdefault("tile_batch_size", 6)
    sahi.setdefault("merge_iou_threshold", 0.1)

def load_predict_config(config_path=CONFIG_PATH):
    """Load, migrate, and expand a CelFDrive prediction configuration file.

    Parameters
    ----------
    config_path : path-like
        YAML document using schema version 1.

    Returns
    -------
    dict
        Effective configuration after in-place legacy migration and ``${...}``
        reference expansion.

    Raises
    ------
    ValueError
        If the YAML declares an unsupported schema version.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError("celfdrive_predict.yaml must contain a YAML mapping")
    if raw_config.get("schema_version") != 1:
        raise ValueError("Unsupported celfdrive_predict.yaml schema_version")

    migrate_predict_config(raw_config)
    return _expand_config_value(raw_config, raw_config)

# Runtime and configured model

def configure_prediction_runtime(config, model=None):
    """Set the prediction resources for the current execution context.

    Call this before using the module with an in-memory configuration, such as
    a GUI-edited configuration or a test fixture. The model is optional and is
    otherwise loaded lazily from ``config['model']['weights_path']``.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping")
    _runtime.set(PredictionRuntime(config=config, model=model))

def get_runtime():
    """Return this context's runtime, loading the default config on first use."""
    runtime = _runtime.get()
    if runtime is None:
        runtime = PredictionRuntime(config=load_predict_config())
        _runtime.set(runtime)
    return runtime

def get_config():
    """Return the effective configuration for the current execution context."""
    return get_runtime().config

def get_backend(cfg):
    """Validate and return the configured inference backend identifier."""
    backend = cfg["model"].get("backend", "ultralytics_yolo")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported model backend: {backend}. Expected one of {sorted(SUPPORTED_BACKENDS)}")
    return backend


def get_project_root(cfg=None):
    """Return the configured CelFDrive project root independent of the CWD.

    SlideBook starts its Python driver from its scripts directory, so relative
    project paths must be anchored to the CelFDrive checkout rather than the
    driver's working directory.
    """
    if cfg is None:
        cfg = get_config()
    project_root = Path(cfg["project"]["repo_path"])
    if not project_root.is_absolute():
        project_root = CONFIG_PATH.parent / project_root
    return project_root.resolve()

def get_model():
    """Load and cache the configured Ultralytics model weights.

    Raises ``ImportError`` when Ultralytics is unavailable and ``ValueError``
    for unsupported configured backends.
    """
    runtime = get_runtime()
    if runtime.model is not None:
        return runtime.model

    cfg = get_config()
    backend = get_backend(cfg)
    weights_path = Path(cfg["model"]["weights_path"])
    if not weights_path.is_absolute():
        weights_path = get_project_root(cfg) / weights_path

    if backend != "ultralytics_yolo":
        raise ValueError(f"Unsupported model backend: {backend}")

    from ultralytics import YOLO

    if not weights_path.is_file():
        raise FileNotFoundError(f"Configured model weights do not exist: {weights_path}")
    runtime.model = YOLO(str(weights_path))
    return runtime.model

# Configuration-derived values

def get_class_info(profile_config):
    """Return class names, confidence thresholds, and priorities by class ID.

    Parameters
    ----------
    profile_config : mapping
        Config ``profile`` mapping with a ``classes`` mapping keyed by numeric
        class IDs.

    Returns
    -------
    dict[int, tuple[str, float, int]]
        ``class_id -> (name, confidence_threshold, priority_rank)``. A rank of
        ``-1`` disables a class from capture-target output.
    """
    return {
        int(class_id): (
            class_config["name"],
            float(class_config["confidence_threshold"]),
            int(class_config["priority_rank"]),
        )
        for class_id, class_config in profile_config["classes"].items()
    }

def get_inference_confidence(class_info):
    """Return the lowest class threshold needed to retain filterable detections."""
    # YOLO applies this threshold before class-specific filtering, so it must be
    # the lowest class confidence threshold to keep all later-filterable detections.
    return min(class_config[1] for class_config in class_info.values())

# Input validation

def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite number")
    return float(value)

def _positive_number(value, name):
    numeric_value = _finite_number(value, name)
    if numeric_value <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric_value

def _stage_direction(value, name):
    numeric_value = _finite_number(value, name)
    if numeric_value not in {-1.0, 1.0}:
        raise ValueError(f"{name} must be -1 or 1")
    return int(numeric_value)

def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)

def _non_negative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)

def _capture_positions(stage_x, stage_y, stage_z):
    scalar_inputs = [stage_x, stage_y, stage_z]
    if all(isinstance(value, Real) and not isinstance(value, bool) for value in scalar_inputs):
        return [
            CapturePosition(
                _finite_number(stage_x, "stage_x"),
                _finite_number(stage_y, "stage_y"),
                _finite_number(stage_z, "stage_z"),
            )
        ]

    try:
        axes = [np.asarray(axis, dtype=float) for axis in scalar_inputs]
    except (TypeError, ValueError) as error:
        raise TypeError("stage_x, stage_y, and stage_z must be numeric scalars or one-dimensional sequences") from error
    if any(axis.ndim != 1 for axis in axes):
        raise ValueError("stage_x, stage_y, and stage_z sequences must be one-dimensional")
    if not axes[0].size or len({axis.size for axis in axes}) != 1:
        raise ValueError("stage_x, stage_y, and stage_z must be non-empty sequences of equal length")
    if not all(np.isfinite(axis).all() for axis in axes):
        raise ValueError("stage coordinates must be finite")
    return [CapturePosition(float(x), float(y), float(z)) for x, y, z in zip(*axes)]

def _validate_image_stack(image, position_count):
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray shaped (height, width, position)")
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width, position)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must have non-empty height and width")
    if image.shape[2] != position_count:
        raise ValueError("image position axis must match the number of stage positions")

def _resolve_coordinate_mode(coordinate_mode, coordinate_converter):
    valid_modes = {"pixel", "stage", "callable"}
    if coordinate_mode not in valid_modes:
        raise ValueError(f"coordinate_mode must be one of {sorted(valid_modes)}")
    if coordinate_mode == "callable" and not callable(coordinate_converter):
        raise TypeError("coordinate_converter must be callable when coordinate_mode is 'callable'")
    if coordinate_mode != "callable" and coordinate_converter is not None:
        raise ValueError("coordinate_converter can only be used when coordinate_mode is 'callable'")

# Logging and output allocation

def is_prediction_image_logging_enabled():
    """Return whether annotated prediction images are written to disk.

    Timing reports are intentionally independent: microscopy acquisition can
    report callback timings without creating logging directories or images.
    """
    cfg = get_config()
    logging_cfg = cfg["logging"]
    prediction_images = logging_cfg.get("prediction_images")
    if prediction_images is not None:
        return prediction_images.get("enabled", True)
    # Support in-memory legacy configurations supplied by older callers.
    return logging_cfg.get("enabled", True) and cfg.get("plotting", {}).get("enabled", True)


def is_timing_enabled():
    """Return whether timing is measured and written to the host Python output."""
    return get_config()["logging"].get("timing", {}).get("enabled", True)

def get_logging_directory():
    """Return the configured logging directory, including date subfolder if enabled."""
    cfg = get_config()
    logging_cfg = cfg["logging"]
    logging_directory = Path(logging_cfg["root_dir"])
    if not logging_directory.is_absolute():
        logging_directory = get_project_root(cfg) / logging_directory
    if logging_cfg.get("use_date_subfolder", True):
        today_date = datetime.now().strftime(logging_cfg.get("date_format", "%Y-%m-%d"))
        logging_directory = logging_directory / today_date
    return logging_directory

def create_experiment_folder(base_dir, experiment_config):
    """Create and return the next numbered experiment directory beneath ``base_dir``."""
    prefix = experiment_config.get("prefix", "exp")
    digits = _positive_integer(experiment_config.get("digits", 3), "logging.experiment_folder.digits")

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_numbers = [
        int(item.name[len(prefix):])
        for item in base_dir.iterdir()
        if item.is_dir() and item.name.startswith(prefix) and item.name[len(prefix):].isdigit()
    ]
    next_exp_num = max(existing_numbers, default=0) + 1
    while True:
        experiment_path = base_dir / f"{prefix}{next_exp_num:0{digits}d}"
        try:
            experiment_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            next_exp_num += 1
        else:
            return experiment_path

def next_output_image_path(experiment_path, output_config):
    """Return the next unused annotated-image path in ``experiment_path``."""
    experiment_path = Path(experiment_path)
    prefix = output_config.get("prefix", "tmpimg")
    digits = _positive_integer(output_config.get("digits", 3), "logging.output_image.digits")
    extension = output_config.get("extension", ".png")
    if not isinstance(extension, str) or not extension.startswith(".") or extension == ".":
        raise ValueError("logging.output_image.extension must be a file extension beginning with '.'")

    existing_numbers = [
        int(item.name[len(prefix):-len(extension)])
        for item in experiment_path.iterdir()
        if item.is_file()
        and item.name.startswith(prefix)
        and item.name.endswith(extension)
        and item.name[len(prefix):-len(extension)].isdigit()
    ]
    next_img_num = max(existing_numbers, default=0) + 1

    return experiment_path / f"{prefix}{next_img_num:0{digits}d}{extension}"

def get_output_image_path():
    """Return the next annotated-image path for the active prediction runtime."""
    runtime = get_runtime()
    if runtime.experiment_path is None:
        raise RuntimeError("Experiment folder has not been created")
    return next_output_image_path(
        runtime.experiment_path,
        get_config()["logging"]["output_image"],
    )
# Image preprocessing and inference

def preprocess_image(img):
    """Convert a supported 2-D or RGB image to a normalised uint8 image.

    Parameters
    ----------
    img : numpy.ndarray
        A two-dimensional ``(height, width)`` image or a three-dimensional
        ``(height, width, channel)`` image. X is columns and Y is rows.

    Returns
    -------
    numpy.ndarray
        Normalized ``uint8`` image of shape ``(height, width)``.

    Raises
    ------
    TypeError, ValueError
        If the input type, shape, or configured clipping percentile is invalid.

    Notes
    -----
    Two-dimensional images are used directly. Three-dimensional images must use
    the configured ``first_channel_if_rgb`` mode, in which case channel zero is
    selected. Values above the configured upper percentile are clipped before
    min/max normalisation.
    """
    preprocessing_cfg = get_config()["preprocessing"]

    input_mode = preprocessing_cfg["input_channel"].get("mode", "first_channel_if_rgb")
    if not isinstance(img, np.ndarray):
        raise TypeError("Image preprocessing expects a numpy.ndarray")
    if img.ndim == 3 and input_mode == "first_channel_if_rgb":
        timepoint_data = img[:, :, 0]
    elif img.ndim == 2:
        timepoint_data = img
    else:
        raise ValueError(f"Unsupported image shape for preprocessing: {img.shape}")

    top_clip_percentile = float(preprocessing_cfg.get("top_clip_percentile", 0.01))
    if not 0 <= top_clip_percentile < 100:
        raise ValueError("preprocessing.top_clip_percentile must be in [0, 100)")
    threshold_value = np.percentile(timepoint_data, 100 - top_clip_percentile)
    timepoint_data_normalized = np.where(
        timepoint_data > threshold_value,
        threshold_value,
        timepoint_data,
    )

    if preprocessing_cfg.get("normalize_min_max", True):
        value_range = timepoint_data_normalized.max() - timepoint_data_normalized.min()
        if value_range == 0:
            timepoint_data_normalized = np.zeros_like(timepoint_data_normalized, dtype=float)
        else:
            timepoint_data_normalized = (
                timepoint_data_normalized - timepoint_data_normalized.min()
            ) / value_range

    img = (timepoint_data_normalized * 255).astype(np.uint8)

    return img

def split_image(img):
    """Split a 2-D image into configured tiles and return each tile and offset.

    Parameters
    ----------
    img : numpy.ndarray
        ``(height, width)`` image. Tile offsets are pixel ``(x, y)`` values.

    Returns
    -------
    list[tuple[numpy.ndarray, int, int]]
        Tile arrays with their left-column and top-row offsets.
    """
    tiling_cfg = get_config()["tiling"]
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        raise ValueError("Image tiling expects a two-dimensional numpy.ndarray")
    if not tiling_cfg.get("enabled", True):
        return [(img, 0, 0)]

    height, width = img.shape
    desired_im_size = _positive_integer(tiling_cfg.get("tile_size_px", 640), "tiling.tile_size_px")
    overlap_px = _non_negative_integer(tiling_cfg.get("overlap_px", 0), "tiling.overlap_px")
    if desired_im_size <= 0:
        raise ValueError("tiling.tile_size_px must be a positive integer")
    if overlap_px >= desired_im_size:
        raise ValueError("tiling.overlap_px must be smaller than tiling.tile_size_px")
    if height == 0 or width == 0:
        raise ValueError("Image tiling does not support empty images")
    stride = desired_im_size - overlap_px
    x_offsets = _tile_offsets(width, desired_im_size, stride)
    y_offsets = _tile_offsets(height, desired_im_size, stride)
    split_images = []

    for y1 in y_offsets:
        for x1 in x_offsets:
            x2 = min(x1 + desired_im_size, width)
            y2 = min(y1 + desired_im_size, height)
            split_img = img[y1:y2, x1:x2]
            split_images.append((split_img, x1, y1))

    return split_images

def _tile_offsets(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    last_offset = length - tile_size
    offsets = list(range(0, last_offset + 1, stride))
    if offsets[-1] != last_offset:
        offsets.append(last_offset)
    return offsets

def _detection_from_values(values):
    return Detection(
        class_id=int(values[0]),
        x=float(values[1]),
        y=float(values[2]),
        width=float(values[3]),
        height=float(values[4]),
        confidence=float(values[5]),
    )

def adjust_coordinates(detections, x_offset, y_offset):
    """Convert Ultralytics boxes to source-image pixel records with tile offsets."""
    detections_adjusted = []
    for detection in detections:
        box = []
        box.append(int(detection.cls))
        box.append(float(detection.xyxy[:, 0] + x_offset))
        box.append(float(detection.xyxy[:, 1] + y_offset))
        box.append(float(detection.xywh[:, 2]))
        box.append(float(detection.xywh[:, 3]))
        box.append(float(detection.conf))
        detections_adjusted.append(box)
    return detections_adjusted

def ultralytics_results_to_detections(results, x_offset, y_offset):
    """Convert Ultralytics result objects to pixel detection records."""
    if len(results[0].boxes.xyxy) == 0:
        return []
    return adjust_coordinates(results[0].boxes, x_offset, y_offset)

def run_model_inference(img, conf):
    """Run the configured backend on an RGB image array at a confidence threshold."""
    backend = get_backend(get_config())
    current_model = get_model()

    if backend == "ultralytics_yolo":
        results = current_model(img, conf=conf)
        return ultralytics_results_to_detections(results, 0, 0)
    raise ValueError(f"Unsupported model backend: {backend}")


def _synchronize_cuda():
    """Wait for queued CUDA work when PyTorch has an active CUDA device."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def run_sahi_inference(img, sahi_config, _timings=None):
    """Run batched SAHI slicing and class-aware IOU merging on one image.

    Parameters
    ----------
    img : numpy.ndarray
        Normalized two-dimensional ``uint8`` image.
    sahi_config : mapping
        Validated SAHI inference settings from the prediction configuration.

    Returns
    -------
    list[list]
        ``[class_id, x, y, width, height, confidence]`` boxes in full-image
        pixel coordinates after greedy non-maximum merging.
    """
    from sahi.postprocess.combine import GreedyNMMPostprocess
    from sahi.prediction import ObjectPrediction
    from sahi.slicing import slice_image

    slice_size = _positive_integer(sahi_config["slice_size_px"], "inference.sahi.slice_size_px")
    tile_batch_size = _positive_integer(sahi_config["tile_batch_size"], "inference.sahi.tile_batch_size")
    overlap_ratio = _finite_number(sahi_config["overlap_ratio"], "inference.sahi.overlap_ratio")
    confidence = _finite_number(
        sahi_config["confidence_threshold"], "inference.sahi.confidence_threshold"
    )
    merge_iou = _finite_number(
        sahi_config["merge_iou_threshold"], "inference.sahi.merge_iou_threshold"
    )
    if not 0 <= confidence <= 1:
        raise ValueError("inference.sahi.confidence_threshold must be between 0 and 1")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("inference.sahi.overlap_ratio must be in [0, 1)")
    if not 0 <= merge_iou <= 1:
        raise ValueError("inference.sahi.merge_iou_threshold must be between 0 and 1")

    postprocessing_started = perf_counter()
    rgb_image = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    slices = slice_image(
        rgb_image,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        auto_slice_resolution=False,
    )
    if _timings is not None:
        _timings.postprocessing_s += perf_counter() - postprocessing_started
    current_model = get_model()
    object_predictions = []
    for start in range(0, len(slices), tile_batch_size):
        postprocessing_started = perf_counter()
        batch = slices[start:start + tile_batch_size]
        tile_images = [item["image"][:, :, ::-1] for item in batch]
        if _timings is not None:
            _timings.postprocessing_s += perf_counter() - postprocessing_started
        inference_started = perf_counter()
        if _timings is not None:
            _synchronize_cuda()
        results = current_model.predict(
            tile_images,
            imgsz=slice_size,
            conf=confidence,
            batch=len(batch),
            verbose=False,
        )
        if _timings is not None:
            _synchronize_cuda()
            _timings.inference_s += perf_counter() - inference_started
        postprocessing_started = perf_counter()
        for item, result in zip(batch, results):
            shift_x, shift_y = item["starting_pixel"]
            for box, score, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                full_box = [
                    float(box[0] + shift_x),
                    float(box[1] + shift_y),
                    float(box[2] + shift_x),
                    float(box[3] + shift_y),
                ]
                object_predictions.append(
                    ObjectPrediction(
                        bbox=full_box,
                        category_id=int(class_id),
                        category_name=str(current_model.names[int(class_id)]),
                        score=float(score),
                        shift_amount=[0, 0],
                        full_shape=[img.shape[0], img.shape[1]],
                    )
                )
        if _timings is not None:
            _timings.postprocessing_s += perf_counter() - postprocessing_started

    postprocessing_started = perf_counter()
    postprocess = GreedyNMMPostprocess(
        match_threshold=merge_iou,
        match_metric="IOU",
        class_agnostic=False,
    )
    detections = []
    for prediction in postprocess(object_predictions):
        box = prediction.bbox
        detections.append(
            [
                int(prediction.category.id),
                float(box.minx),
                float(box.miny),
                float(box.maxx - box.minx),
                float(box.maxy - box.miny),
                float(prediction.score.value),
            ]
        )
    if _timings is not None:
        _timings.postprocessing_s += perf_counter() - postprocessing_started
    return detections

def deduplicate_detections(detections, tolerance_px):
    """Keep the highest-confidence nearby detection for each class.

    ``detections`` contains ``[class_id, x, y, width, height, confidence]``
    pixel boxes. Centres within ``tolerance_px`` pixels are duplicates only when
    their class IDs match.
    """
    tolerance_px = _finite_number(tolerance_px, "tiling.deduplication_tolerance_px")
    if tolerance_px < 0:
        raise ValueError("tiling.deduplication_tolerance_px must be non-negative")

    ordered = sorted(
        (_detection_from_values(detection) for detection in detections),
        key=lambda detection: (
            detection.class_id,
            -detection.confidence,
            detection.centre_y,
            detection.centre_x,
        ),
    )
    unique = []
    for detection in ordered:
        is_duplicate = any(
            detection.class_id == accepted.class_id
            and math.hypot(detection.centre_x - accepted.centre_x, detection.centre_y - accepted.centre_y)
            <= tolerance_px
            for accepted in unique
        )
        if not is_duplicate:
            unique.append(detection)
    return [[d.class_id, d.x, d.y, d.width, d.height, d.confidence] for d in unique]

def filter_and_sort_detections(detections, class_info):
    """Filter pixel detections by class threshold and order by capture priority."""
    filtered_detections = [
        det for det in detections
        if det[5] >= class_info[det[0]][1] and class_info[det[0]][2] != -1
    ]

    return sorted(
        filtered_detections,
        key=lambda det: (class_info[det[0]][2], -det[5])
    )

def global_filter_and_sort_detections(all_detections, class_info):
    """Filter converted target records by class threshold and capture priority."""
    filtered_detections = [
        det for det in all_detections
        if det[3] >= class_info[det[4]][1] and class_info[det[4]][2] != -1
    ]

    return sorted(
        filtered_detections,
        key=lambda det: (class_info[det[4]][2], -det[3])
    )

def plot_image_with_results(image, boxes, class_names, class_info, file_path):
    """Save an annotated prediction image to ``file_path``."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plotting_cfg = get_config()["plotting"]
    bbox_cfg = plotting_cfg["bbox"]
    label_cfg = plotting_cfg["label"]

    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap=plotting_cfg.get("cmap", "gray"))
    ax.axis('off')

    filtered_boxes = filter_and_sort_detections(boxes, class_info)

    for box in filtered_boxes:
        class_id, x, y, w, h, confidence = box
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=bbox_cfg.get("line_width", 1),
            edgecolor=bbox_cfg.get("edge_color", "red"),
            facecolor='none',
        )
        ax.add_patch(rect)

        label = f"{class_names[class_id]}:{confidence:.2f}"
        ax.text(
            x,
            y,
            label,
            color=label_cfg.get("text_color", "white"),
            fontsize=label_cfg.get("font_size", 8),
            ha='left',
            va='bottom',
            bbox=dict(
                boxstyle="square,pad=0.1",
                fc=label_cfg.get("background_color", "black"),
                ec="none",
                alpha=label_cfg.get("background_alpha", 0.5),
            ),
        )

    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_image(raw_img, conf=None, save_path=None, class_info=None, plot=False, _timings=None):
    """Preprocess an image, run inference, and return pixel-space detections.

    Parameters
    ----------
    raw_img : numpy.ndarray
        Image of shape ``(height, width)`` or ``(height, width, channel)``.
    conf : float, optional
        Model inference threshold. Standard mode defaults to the lowest enabled
        class limit; SAHI mode defaults to its configured confidence threshold.
    save_path : path-like, optional
        Plot destination when ``plot`` is true.
    class_info : dict[int, tuple[str, float, int]], optional
        Class filtering configuration.
    plot : bool, default=False
        Save an annotated image as a side effect.

    Returns
    -------
    list[list]
        ``[class_id, x, y, width, height, confidence]`` boxes in source-image
        pixels, after tile offsets and same-class de-duplication.
    """
    cfg = get_config()
    if class_info is None:
        class_info = get_class_info(cfg["profile"])

    preprocessing_started = perf_counter()
    processed_img = preprocess_image(raw_img)
    if _timings is not None:
        _timings.preprocessing_s += perf_counter() - preprocessing_started
    inference_cfg = cfg.get("inference", {"mode": "tiling"})
    inference_mode = inference_cfg.get("mode", "tiling")
    if inference_mode == "sahi":
        sahi_config = dict(inference_cfg["sahi"])
        if conf is not None:
            sahi_config["confidence_threshold"] = conf
        if _timings is None:
            results = run_sahi_inference(processed_img, sahi_config)
        else:
            results = run_sahi_inference(processed_img, sahi_config, _timings=_timings)
    elif inference_mode in {"tiling", "full_image"}:
        if conf is None:
            conf = get_inference_confidence(class_info)
        postprocessing_started = perf_counter()
        split_images = (
            split_image(processed_img)
            if inference_mode == "tiling"
            else [(processed_img, 0, 0)]
        )
        results = []

        for img, x_offset, y_offset in split_images:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

            inference_started = perf_counter()
            _synchronize_cuda()
            if cfg["model"].get("suppress_stdout", True):
                with open(os.devnull, 'w') as nullfile:
                    with contextlib.redirect_stdout(nullfile):
                        results_split = run_model_inference(img, conf)
            else:
                results_split = run_model_inference(img, conf)
            _synchronize_cuda()
            if _timings is not None:
                _timings.inference_s += perf_counter() - inference_started

            for detection in results_split:
                detection[1] += x_offset
                detection[2] += y_offset
            results.extend(results_split)

        tiling_cfg = cfg["tiling"]
        results = deduplicate_detections(
            results,
            tiling_cfg.get("deduplication_tolerance_px", 1.0),
        )
        if _timings is not None:
            _timings.postprocessing_s += perf_counter() - postprocessing_started
    else:
        raise ValueError("inference.mode must be 'tiling', 'full_image', or 'sahi'")

    if plot:
        logging_started = perf_counter()
        if save_path is None:
            save_path = get_output_image_path()
        class_names = {key: value[0] for key, value in class_info.items()}
        plot_image_with_results(processed_img, results, class_names, class_info, save_path)
        if _timings is not None:
            _timings.logging_s += perf_counter() - logging_started

    return results

def process_image_from_path(image_path, conf=None, save_path=None, class_info=None, plot=False):
    """Read an image path with OpenCV and delegate to :func:`process_image`."""
    import cv2

    if isinstance(class_info, bool) and plot is False:
        plot = class_info
        class_info = None

    img = cv2.imread(image_path)
    return process_image(img, conf, save_path, class_info, plot)

# Coordinate conversion

def _run_coordinate_converter(coordinate_converter, **kwargs):
    converted = coordinate_converter(**kwargs)
    try:
        target_x, target_y, target_z = converted
    except (TypeError, ValueError) as error:
        raise ValueError("coordinate_converter must return exactly three values: (x, y, z)") from error
    return (
        _finite_number(target_x, "coordinate_converter x output"),
        _finite_number(target_y, "coordinate_converter y output"),
        _finite_number(target_z, "coordinate_converter z output"),
    )

def _convert_detection(
    position,
    detection,
    image_width_px,
    image_height_px,
    *,
    xy_pixel_spacing_um,
    z_offset_um,
    coordinate_mode,
    coordinate_converter,
    x_stage_direction,
    y_stage_direction,
    legacy_llsm_y_inversion,
    class_name,
):
    target_z = position.z + z_offset_um
    if coordinate_mode == "pixel":
        target_x, target_y = detection.centre_x, detection.centre_y
    elif coordinate_mode == "stage":
        # This preserves the legacy 3i stage-coordinate calculation.
        x_offset = detection.x - image_width_px / 2
        y_offset = detection.y - image_height_px / 2
        adjusted_y_direction = y_stage_direction
        if legacy_llsm_y_inversion:
            adjusted_y_direction *= -1
        target_x = position.x + x_offset * xy_pixel_spacing_um * x_stage_direction
        target_y = position.y + y_offset * xy_pixel_spacing_um * adjusted_y_direction
    else:
        target_x, target_y, target_z = _run_coordinate_converter(
            coordinate_converter,
            stage_x=position.x,
            stage_y=position.y,
            stage_z=position.z,
            detection_x_px=detection.centre_x,
            detection_y_px=detection.centre_y,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            xy_pixel_spacing_um=xy_pixel_spacing_um,
            z_offset_um=z_offset_um,
            class_id=detection.class_id,
            confidence=detection.confidence,
            class_name=class_name,
        )
    return [target_x, target_y, target_z, detection.confidence, detection.class_id, class_name]

def image_coordinates_to_physical(x, y, im_x, im_y, w, h, new_z, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, class_id, conf, class_name):
    """Convert one detected image position to a physical capture target.

    ``z_spacing`` and ``z_stage_direction`` remain accepted because this
    helper mirrors the historic microscope callback signature.
    """
    position = CapturePosition(_finite_number(x, "x"), _finite_number(y, "y"), _finite_number(new_z, "new_z"))
    detection = Detection(int(class_id), float(im_x), float(im_y), 0.0, 0.0, float(conf))
    return _convert_detection(
        position,
        detection,
        w,
        h,
        xy_pixel_spacing_um=_positive_number(xy_pixel_spacing, "xy_pixel_spacing"),
        z_offset_um=0.0,
        coordinate_mode="stage",
        coordinate_converter=None,
        x_stage_direction=_stage_direction(x_stage_direction, "x_stage_direction"),
        y_stage_direction=_stage_direction(y_stage_direction, "y_stage_direction"),
        legacy_llsm_y_inversion=bool(LLSM and get_config()["coordinate_conversion"]["llsm"].get("invert_y_stage_direction", True)),
        class_name=class_name,
    )

image_cordinates_to_physical = image_coordinates_to_physical


def merge_close_coordinates(coordinates, tolerance):
    """Merge target records within an XY tolerance in their output coordinate units."""
    unique_coords = []

    for coord in coordinates:
        x, y, z, conf, class_id, class_name = coord
        if not any(np.sqrt((x - uc[0])**2 + (y - uc[1])**2) <= tolerance for uc in unique_coords):
            unique_coords.append(coord)

    unique_coords = np.array(unique_coords, dtype=object)
    return unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2], list(unique_coords[:, 5])

def process_single_location(
    position,
    image,
    class_info,
    *,
    xy_pixel_spacing_um,
    z_offset_um,
    coordinate_mode,
    coordinate_converter,
    x_stage_direction,
    y_stage_direction,
    legacy_llsm_y_inversion,
    timings=None,
):
    """Run one overview image and convert its detections to capture targets.

    ``image`` is a ``(height, width)`` overview plane and ``position`` is in
    micrometres. Returned records are ``[x, y, z, confidence, class_id, name]``
    in the requested coordinate mode.
    """
    height, width = image.shape
    plot_enabled = is_prediction_image_logging_enabled()
    logging_started = perf_counter()
    img_path = get_output_image_path() if plot_enabled else None
    if timings is not None:
        timings.logging_s += perf_counter() - logging_started
    class_names = {key: value[0] for key, value in class_info.items()}
    results = process_image(image, None, img_path, class_info, plot=plot_enabled, _timings=timings)
    postprocessing_started = perf_counter()
    filtered_detections = filter_and_sort_detections(results, class_info)

    converted = [
        _convert_detection(
            position,
            _detection_from_values(detection),
            width,
            height,
            xy_pixel_spacing_um=xy_pixel_spacing_um,
            z_offset_um=z_offset_um,
            coordinate_mode=coordinate_mode,
            coordinate_converter=coordinate_converter,
            x_stage_direction=x_stage_direction,
            y_stage_direction=y_stage_direction,
            legacy_llsm_y_inversion=legacy_llsm_y_inversion,
            class_name=class_names[int(detection[0])],
        )
        for detection in filtered_detections
    ]
    if timings is not None:
        timings.postprocessing_s += perf_counter() - postprocessing_started
    return converted

# Montage and capture results

def process_montage(
    positions,
    image,
    class_info,
    *,
    xy_pixel_spacing_um,
    z_offset_um,
    coordinate_mode,
    coordinate_converter,
    x_stage_direction,
    y_stage_direction,
    legacy_llsm_y_inversion,
    timings=None,
):
    """Process every plane of a montage and return target coordinate arrays.

    Parameters
    ----------
    positions : sequence[CapturePosition]
        One micrometre-valued stage position per montage plane.
    image : numpy.ndarray
        Overview stack with shape ``(height, width, position)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, list[str]]
        Target X, Y, Z arrays and class names. X/Y are pixels in ``pixel`` mode
        and stage coordinates in micrometres otherwise.
    """
    results = []
    for i, position in enumerate(positions):
        tmp = process_single_location(
            position,
            image[:, :, i],
            class_info,
            xy_pixel_spacing_um=xy_pixel_spacing_um,
            z_offset_um=z_offset_um,
            coordinate_mode=coordinate_mode,
            coordinate_converter=coordinate_converter,
            x_stage_direction=x_stage_direction,
            y_stage_direction=y_stage_direction,
            legacy_llsm_y_inversion=legacy_llsm_y_inversion,
            timings=timings,
        )
        if tmp:
            for item in tmp:
                results.append(item)

    postprocessing_started = perf_counter()
    sorted_results = global_filter_and_sort_detections(results, class_info)

    if not sorted_results:
        if timings is not None:
            timings.postprocessing_s += perf_counter() - postprocessing_started
        return np.array([]), np.array([]), np.array([]), []

    if coordinate_mode == "pixel":
        final_result = np.array(sorted_results, dtype=object)
        if timings is not None:
            timings.postprocessing_s += perf_counter() - postprocessing_started
        return final_result[:, 0], final_result[:, 1], final_result[:, 2], list(final_result[:, 5])

    final_result = np.array(sorted_results, dtype=object)
    tolerance = get_config()["coordinate_conversion"]["merge_tolerance_um"]
    new_x, new_y, new_z, class_names = merge_close_coordinates(final_result, tolerance)
    if timings is not None:
        timings.postprocessing_s += perf_counter() - postprocessing_started
    return new_x, new_y, new_z, class_names

def _capture_result(positions, results, profile_config):
    new_x, new_y, new_z, class_list = results
    count = len(new_x)
    if count == 0:
        no_detection = get_config()["no_detection"]
        mode = no_detection.get("mode", "empty_3i_capture_script")
        if mode == "end_workflow":
            return 0, np.array([]), np.array([]), np.array([]), [], [], []
        if mode == "empty_3i_capture_script":
            first_position = positions[0]
            return (
                1,
                np.array([first_position.x]),
                np.array([first_position.y]),
                np.array([first_position.z]),
                [no_detection.get("empty_3i_capture_script", "donothing")],
                ["nothing"],
                ["nothing"],
            )
        raise ValueError(f"Unsupported no_detection mode: {mode}")

    scripts = [profile_config["highres_script"] for _ in range(count)]
    names = [
        profile_config["name_template"].format(
            class_name=class_list[index],
            x=new_x[index],
            y=new_y[index],
            z=new_z[index],
        )
        for index in range(count)
    ]
    comments = [profile_config["highres_comment"] for _ in range(count)]
    return count, new_x, new_y, new_z, scripts, names, comments

# Public prediction APIs

def get_target_locations(
    *,
    stage_x,
    stage_y,
    stage_z,
    image,
    xy_pixel_spacing_um=None,
    x_stage_direction=1,
    y_stage_direction=1,
    z_offset_um=None,
    coordinate_mode=None,
    coordinate_converter=None,
    _legacy_llsm_y_inversion=False,
):
    """Return capture targets using named parameters and explicit coordinate mode.

    Parameters
    ----------
    stage_x, stage_y, stage_z : float or sequence[float]
        One micrometre-valued stage position or equally sized position vectors.
    image : numpy.ndarray
        Overview montage shaped ``(height, width, position)``. Its final axis
        must have one plane per stage position.
    xy_pixel_spacing_um : float, optional
        Positive physical width of one pixel for ``stage`` mode; finite for a
        caller-supplied converter and unused in ``pixel`` mode.
    x_stage_direction, y_stage_direction : {-1, 1}
        Orientation of image X/Y relative to the stage axes.
    z_offset_um : float, optional
        Offset added to each target Z in micrometres.
    coordinate_mode : {'stage', 'pixel', 'callable'}, optional
        Output coordinate convention; defaults to the YAML configuration.
    coordinate_converter : callable, optional
        Callable-mode converter returning exactly three finite ``(x, y, z)``
        values from the documented keyword context.

    Returns
    -------
    tuple[int, numpy.ndarray, numpy.ndarray, numpy.ndarray, list[str], list[str], list[str]]
        Target count; X/Y/Z arrays; capture scripts; display names; and comments.
        Coordinates are micrometres in stage/callable modes and pixels in pixel
        mode.

    Raises
    ------
    TypeError, ValueError
        If positions, stack shape, coordinate mode, directions, spacing, or a
        converter result violates the public contract.
    """
    cfg = get_config()
    timings = PredictionTimings() if is_timing_enabled() else None
    total_started = perf_counter() if timings is not None else None
    positions = _capture_positions(stage_x, stage_y, stage_z)
    _validate_image_stack(image, len(positions))
    mode = coordinate_mode or cfg["coordinate_conversion"].get("mode", "stage")
    _resolve_coordinate_mode(mode, coordinate_converter)
    offset = cfg["coordinate_conversion"].get("default_z_offset_um", 0) if z_offset_um is None else z_offset_um
    offset = _finite_number(offset, "z_offset_um")
    x_direction = _stage_direction(x_stage_direction, "x_stage_direction")
    y_direction = _stage_direction(y_stage_direction, "y_stage_direction")

    if mode == "stage":
        spacing = _positive_number(xy_pixel_spacing_um, "xy_pixel_spacing_um")
    elif mode == "callable":
        spacing = _finite_number(xy_pixel_spacing_um, "xy_pixel_spacing_um")
    else:
        spacing = None

    profile_config = cfg["profile"]
    class_info = get_class_info(profile_config)
    if is_prediction_image_logging_enabled():
        logging_started = perf_counter()
        logging_directory = get_logging_directory()
        logging_directory.mkdir(parents=True, exist_ok=True)
        get_runtime().experiment_path = create_experiment_folder(
            logging_directory,
            cfg["logging"]["experiment_folder"],
        )
        if timings is not None:
            timings.logging_s += perf_counter() - logging_started

    montage_results = process_montage(
        positions,
        image,
        class_info,
        xy_pixel_spacing_um=spacing,
        z_offset_um=offset,
        coordinate_mode=mode,
        coordinate_converter=coordinate_converter,
        x_stage_direction=x_direction,
        y_stage_direction=y_direction,
        legacy_llsm_y_inversion=_legacy_llsm_y_inversion,
        timings=timings,
    )
    postprocessing_started = perf_counter()
    capture_result = _capture_result(positions, montage_results, profile_config)
    if timings is not None:
        timings.postprocessing_s += perf_counter() - postprocessing_started
        timings.total_s = perf_counter() - total_started
    get_runtime().last_timings = timings
    if timings is not None:
        print(timings.format_report())
    return capture_result


def get_last_prediction_timings():
    """Return timing measurements from the most recent target-selection call.

    The returned :class:`PredictionTimings` instance holds seconds accumulated
    over all overview planes in that montage. It is also emitted as one line to
    the acquisition host's Python output after every successful call.
    """
    return get_runtime().last_timings

def get_target_location(X, Y, Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM=False, z_offset=None):
    """Legacy positional wrapper for :func:`get_target_locations`.

    ``z_spacing`` and ``z_stage_direction`` are accepted for compatibility with
    existing microscope scripts; the current two-dimensional target conversion
    does not use them.
    """
    _finite_number(z_spacing, "z_spacing")
    _stage_direction(z_stage_direction, "z_stage_direction")
    llsm_inversion = bool(LLSM and get_config()["coordinate_conversion"]["llsm"].get("invert_y_stage_direction", True))
    return get_target_locations(
        stage_x=X,
        stage_y=Y,
        stage_z=Z,
        image=image,
        xy_pixel_spacing_um=xy_pixel_spacing,
        x_stage_direction=x_stage_direction,
        y_stage_direction=y_stage_direction,
        z_offset_um=z_offset,
        coordinate_mode="stage",
        _legacy_llsm_y_inversion=llsm_inversion,
    )
