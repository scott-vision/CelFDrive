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
from dataclasses import dataclass
from numbers import Real

import numpy as np
import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "celfdrive_predict.yaml"
CONFIG_REF_PATTERN = re.compile(r"\$\{([^}]+)\}")
SUPPORTED_BACKENDS = {"ultralytics_yolo"}

config = None
model = None
experiment_path = None


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

    if raw_config.get("schema_version") != 1:
        raise ValueError("Unsupported celfdrive_predict.yaml schema_version")

    migrate_predict_config(raw_config)
    return _expand_config_value(raw_config, raw_config)


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
    slidebook = raw_config.setdefault("slidebook", {})
    slidebook.setdefault("objective_offset_um", {"x": 0.0, "y": 0.0, "z": 0.0})
    tiling = raw_config.setdefault("tiling", {})
    tiling.setdefault("overlap_px", 0)
    tiling.setdefault("deduplication_tolerance_px", 1.0)


def get_config():
    """Return the process-cached effective prediction configuration mapping."""
    global config
    if config is None:
        config = load_predict_config()
    return config


def get_model():
    """Load and cache the configured Ultralytics model weights.

    Raises ``ImportError`` when Ultralytics is unavailable and ``ValueError``
    for unsupported configured backends.
    """
    global model
    if model is not None:
        return model

    cfg = get_config()
    backend = get_backend(cfg)
    weights_path = cfg["model"]["weights_path"]

    if backend != "ultralytics_yolo":
        raise ValueError(f"Unsupported model backend: {backend}")

    from ultralytics import YOLO

    model = YOLO(weights_path)

    return model


def get_backend(cfg):
    """Validate and return the configured inference backend identifier."""
    backend = cfg["model"].get("backend", "ultralytics_yolo")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported model backend: {backend}. Expected one of {sorted(SUPPORTED_BACKENDS)}")
    return backend


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


def is_logging_enabled():
    """Return whether prediction logging and plot output are enabled."""
    return get_config()["logging"].get("enabled", True)


def get_logging_directory():
    """Return the configured logging directory, including date subfolder if enabled."""
    cfg = get_config()
    logging_cfg = cfg["logging"]
    logging_directory = Path(logging_cfg["root_dir"])
    if not logging_directory.is_absolute():
        logging_directory = Path(cfg["project"]["repo_path"]) / logging_directory
    if logging_cfg.get("use_date_subfolder", True):
        today_date = datetime.now().strftime(logging_cfg.get("date_format", "%Y-%m-%d"))
        logging_directory = logging_directory / today_date
    return logging_directory


def create_exp_folder(base_dir):
    """Create and cache the next numbered experiment directory beneath ``base_dir``."""
    global experiment_path
    logging_cfg = get_config()["logging"]
    exp_cfg = logging_cfg["experiment_folder"]
    prefix = exp_cfg.get("prefix", "exp")
    digits = int(exp_cfg.get("digits", 3))

    base_dir = Path(base_dir)
    items = os.listdir(base_dir)
    exp_folders = [
        item for item in items
        if item.startswith(prefix)
        and item[len(prefix):].isdigit()
        and (base_dir / item).is_dir()
    ]
    exp_folders.sort()

    if exp_folders:
        last_exp_num = int(exp_folders[-1][len(prefix):])
        next_exp_num = last_exp_num + 1
    else:
        next_exp_num = 1

    experiment_path = base_dir / f"{prefix}{next_exp_num:0{digits}d}"
    experiment_path.mkdir(parents=True, exist_ok=True)


def get_outimg_path():
    """Return the next unused annotated-image path in the current experiment."""
    global experiment_path
    if experiment_path is None:
        raise RuntimeError("Experiment folder has not been created")

    output_cfg = get_config()["logging"]["output_image"]
    prefix = output_cfg.get("prefix", "tmpimg")
    digits = int(output_cfg.get("digits", 3))
    extension = output_cfg.get("extension", ".png")

    items = os.listdir(experiment_path)
    tmp_images = [
        item for item in items
        if item.startswith(prefix)
        and item.endswith(extension)
        and item[len(prefix):-len(extension)].isdigit()
        and (experiment_path / item).is_file()
    ]
    tmp_images.sort()

    if tmp_images:
        last_img_num = int(tmp_images[-1][len(prefix):-len(extension)])
        next_img_num = last_img_num + 1
    else:
        next_img_num = 1

    return experiment_path / f"{prefix}{next_img_num:0{digits}d}{extension}"


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


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


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


def run_model_inference(img, conf):
    """Run the configured backend on an RGB image array at a confidence threshold."""
    backend = get_backend(get_config())
    current_model = get_model()

    if backend == "ultralytics_yolo":
        results = current_model(img, conf=conf)
        return ultralytics_results_to_detections(results, 0, 0)
    raise ValueError(f"Unsupported model backend: {backend}")


def ultralytics_results_to_detections(results, x_offset, y_offset):
    """Convert Ultralytics result objects to pixel detection records."""
    if len(results[0].boxes.xyxy) == 0:
        return []
    return adjust_coordinates(results[0].boxes, x_offset, y_offset)


def process_image(raw_img, conf=None, save_path=None, class_info=None, plot=False):
    """Preprocess an image, run inference, and return pixel-space detections.

    Parameters
    ----------
    raw_img : numpy.ndarray
        Image of shape ``(height, width)`` or ``(height, width, channel)``.
    conf : float, optional
        Model inference threshold; defaults to the lowest enabled class limit.
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
    if conf is None:
        conf = get_inference_confidence(class_info)

    processed_img = preprocess_image(raw_img)
    split_images = split_image(processed_img)
    results = []

    for img, x_offset, y_offset in split_images:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        if cfg["model"].get("suppress_stdout", True):
            with open(os.devnull, 'w') as nullfile:
                with contextlib.redirect_stdout(nullfile):
                    results_split = run_model_inference(img, conf)
        else:
            results_split = run_model_inference(img, conf)

        for detection in results_split:
            detection[1] += x_offset
            detection[2] += y_offset
        results.extend(results_split)

    tiling_cfg = cfg["tiling"]
    results = deduplicate_detections(
        results,
        tiling_cfg.get("deduplication_tolerance_px", 1.0),
    )

    if plot:
        if save_path is None:
            save_path = get_outimg_path()
        class_names = {key: value[0] for key, value in class_info.items()}
        plot_image_with_results(processed_img, results, class_names, class_info, save_path)

    return results


def process_image_from_path(image_path, conf=None, save_path=None, class_info=None, plot=False):
    """Read an image path with OpenCV and delegate to :func:`process_image`."""
    import cv2

    if isinstance(class_info, bool) and plot is False:
        plot = class_info
        class_info = None

    img = cv2.imread(image_path)
    return process_image(img, conf, save_path, class_info, plot)


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
):
    """Run one overview image and convert its detections to capture targets.

    ``image`` is a ``(height, width)`` overview plane and ``position`` is in
    micrometres. Returned records are ``[x, y, z, confidence, class_id, name]``
    in the requested coordinate mode.
    """
    height, width = image.shape
    plot_enabled = get_config()["plotting"].get("enabled", True) and is_logging_enabled()
    img_path = get_outimg_path() if plot_enabled else None
    class_names = {key: value[0] for key, value in class_info.items()}
    results = process_image(image, None, img_path, class_info, plot=plot_enabled)
    filtered_detections = filter_and_sort_detections(results, class_info)

    return [
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


def image_cordinates_to_physical(x, y, im_x, im_y, w, h, new_z, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, class_id, conf, class_name):
    """Legacy coordinate-conversion helper retained for existing integrations."""
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


def merge_close_coordinates(coordinates, tolerance):
    """Merge target records within an XY tolerance in their output coordinate units."""
    unique_coords = []

    for coord in coordinates:
        x, y, z, conf, class_id, class_name = coord
        if not any(np.sqrt((x - uc[0])**2 + (y - uc[1])**2) <= tolerance for uc in unique_coords):
            unique_coords.append(coord)

    unique_coords = np.array(unique_coords, dtype=object)
    return unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2], list(unique_coords[:, 5])


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
        )
        if tmp:
            for item in tmp:
                results.append(item)

    sorted_results = global_filter_and_sort_detections(results, class_info)

    if not sorted_results:
        return np.array([]), np.array([]), np.array([]), []

    if coordinate_mode == "pixel":
        final_result = np.array(sorted_results, dtype=object)
        return final_result[:, 0], final_result[:, 1], final_result[:, 2], list(final_result[:, 5])

    final_result = np.array(sorted_results, dtype=object)
    tolerance = get_config()["coordinate_conversion"].get("merge_tolerance_um", 20)
    new_x, new_y, new_z, class_names = merge_close_coordinates(final_result, tolerance)
    return new_x, new_y, new_z, class_names


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
    global experiment_path

    cfg = get_config()
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
    if is_logging_enabled():
        logging_directory = get_logging_directory()
        logging_directory.mkdir(parents=True, exist_ok=True)
        create_exp_folder(logging_directory)

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
    )
    return _capture_result(positions, montage_results, profile_config)


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
