from pathlib import Path
import re
from datetime import datetime
import contextlib
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "celfdrive_predict.yaml"
CONFIG_REF_PATTERN = re.compile(r"\$\{([^}]+)\}")

config = None
model = None
experiment_path = None


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
    with open(config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if raw_config.get("schema_version") != 1:
        raise ValueError("Unsupported celfdrive_predict.yaml schema_version")

    return _expand_config_value(raw_config, raw_config)


def get_config():
    global config
    if config is None:
        config = load_predict_config()
    return config


def get_model():
    global model
    if model is not None:
        return model

    cfg = get_config()
    backend = normalize_backend(cfg["model"].get("backend", "ultralytics_yolo"))
    weights_path = cfg["model"]["weights_path"]

    if backend == "ultralytics_yolo":
        from ultralytics import YOLO

        model = YOLO(weights_path)
    elif backend == "rfdetr":
        from rfdetr import RFDETRBase

        try:
            model = RFDETRBase(pretrain_weights=weights_path)
        except TypeError:
            model = RFDETRBase()
            if hasattr(model, "load"):
                model.load(weights_path)
            else:
                raise TypeError("RFDETRBase does not support loading weights with this adapter")
    elif backend == "torchscript":
        import torch

        model = torch.jit.load(weights_path, map_location="cpu")
        model.eval()
    else:
        raise ValueError(f"Unsupported model backend: {backend}")

    return model


def normalize_backend(backend):
    backend_key = str(backend).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ultrayltics_yolo": "ultralytics_yolo",
        "ultralytics_yolo": "ultralytics_yolo",
        "rfdetr": "rfdetr",
        "rf_detr": "rfdetr",
        "torchscript": "torchscript",
        "torch_script": "torchscript",
    }
    return aliases.get(backend_key, backend_key)


def get_profile_config(LLSM):
    profiles = get_config()["profiles"]
    profile_name = "llsm" if LLSM else "sdc"
    return profiles[profile_name]


def get_class_info(profile_config):
    return {
        int(class_id): (
            class_config["name"],
            float(class_config["confidence_threshold"]),
            int(class_config["priority_rank"]),
        )
        for class_id, class_config in profile_config["classes"].items()
    }


def get_inference_confidence(class_info):
    # YOLO applies this threshold before class-specific filtering, so it must be
    # the lowest class confidence threshold to keep all later-filterable detections.
    return min(class_config[1] for class_config in class_info.values())


def is_logging_enabled():
    return get_config()["logging"].get("enabled", True)


def get_logging_directory():
    cfg = get_config()
    logging_cfg = cfg["logging"]
    logging_directory = Path(logging_cfg["root_dir"])
    if logging_cfg.get("use_date_subfolder", True):
        today_date = datetime.now().strftime(logging_cfg.get("date_format", "%Y-%m-%d"))
        logging_directory = logging_directory / today_date
    return logging_directory


def create_exp_folder(base_dir):
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
    filtered_detections = [
        det for det in detections
        if det[5] >= class_info[det[0]][1] and class_info[det[0]][2] != -1
    ]

    return sorted(
        filtered_detections,
        key=lambda det: (class_info[det[0]][2], -det[5])
    )


def global_filter_and_sort_detections(all_detections, class_info):
    filtered_detections = [
        det for det in all_detections if det[3] >= class_info[det[4]][1]
    ]

    return sorted(
        filtered_detections,
        key=lambda det: (class_info[det[4]][2], -det[3])
    )


def plot_image_with_results(image, boxes, class_names, class_info, file_path):
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
    preprocessing_cfg = get_config()["preprocessing"]

    input_mode = preprocessing_cfg["input_channel"].get("mode", "first_channel_if_rgb")
    if len(img.shape) == 3 and input_mode == "first_channel_if_rgb":
        timepoint_data = img[:, :, 0]
    elif len(img.shape) == 2:
        timepoint_data = img
    else:
        raise ValueError(f"Unsupported image shape for preprocessing: {img.shape}")

    top_clip_percentile = float(preprocessing_cfg.get("top_clip_percentile", 0.01))
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
    tiling_cfg = get_config()["tiling"]
    if not tiling_cfg.get("enabled", True):
        return [(img, 0, 0)]

    height, width = img.shape
    desired_im_size = int(tiling_cfg.get("tile_size_px", 640))
    split_images = []

    for row in range(math.ceil(height / desired_im_size)):
        for col in range(math.ceil(width / desired_im_size)):
            x1 = col * desired_im_size
            y1 = row * desired_im_size
            x2 = min(x1 + desired_im_size, width)
            y2 = min(y1 + desired_im_size, height)
            x1 = x2 - desired_im_size
            y1 = y2 - desired_im_size
            split_img = img[y1:y2, x1:x2]
            split_images.append((split_img, x1, y1))

    return split_images


def adjust_coordinates(detections, x_offset, y_offset):
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
    cfg = get_config()
    backend = normalize_backend(cfg["model"].get("backend", "ultralytics_yolo"))
    current_model = get_model()

    if backend == "ultralytics_yolo":
        results = current_model(img, conf=conf)
        return ultralytics_results_to_detections(results, 0, 0)
    if backend == "rfdetr":
        results = current_model.predict(img, threshold=conf)
        return rfdetr_results_to_detections(results)
    if backend == "torchscript":
        return torchscript_results_to_detections(current_model, img, conf)
    raise ValueError(f"Unsupported model backend: {backend}")


def ultralytics_results_to_detections(results, x_offset, y_offset):
    if len(results[0].boxes.xyxy) == 0:
        return []
    return adjust_coordinates(results[0].boxes, x_offset, y_offset)


def rfdetr_results_to_detections(results):
    xyxy = getattr(results, "xyxy", None)
    confidence = getattr(results, "confidence", None)
    class_id = getattr(results, "class_id", None)

    if xyxy is None and isinstance(results, dict):
        xyxy = results.get("xyxy")
        if xyxy is None:
            xyxy = results.get("boxes")
        confidence = results.get("confidence")
        if confidence is None:
            confidence = results.get("scores")
        class_id = results.get("class_id")
        if class_id is None:
            class_id = results.get("labels")

    if xyxy is None:
        raise ValueError("RF-DETR results must provide xyxy/boxes, confidence/scores, and class_id/labels")

    return xyxy_arrays_to_detections(xyxy, confidence, class_id)


def torchscript_results_to_detections(current_model, img, conf):
    import torch

    input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        outputs = current_model(input_tensor)

    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    if isinstance(outputs, dict):
        return xyxy_arrays_to_detections(outputs["boxes"], outputs["scores"], outputs["labels"], conf)

    output_array = outputs.detach().cpu().numpy()
    if output_array.ndim == 3:
        output_array = output_array[0]
    if output_array.shape[1] < 6:
        raise ValueError("TorchScript tensor output must be Nx6: x1,y1,x2,y2,confidence,class_id")

    return [
        xyxy_to_detection(row[:4], row[4], row[5])
        for row in output_array
        if row[4] >= conf
    ]


def xyxy_arrays_to_detections(xyxy, confidence, class_id, conf_threshold=None):
    xyxy = np.asarray(xyxy)
    confidence = np.asarray(confidence)
    class_id = np.asarray(class_id)

    detections = []
    for box, score, cls in zip(xyxy, confidence, class_id):
        if conf_threshold is not None and score < conf_threshold:
            continue
        detections.append(xyxy_to_detection(box, score, cls))
    return detections


def xyxy_to_detection(box, score, cls):
    x1, y1, x2, y2 = [float(value) for value in box]
    return [int(cls), x1, y1, x2 - x1, y2 - y1, float(score)]


def process_image(raw_img, conf=None, save_path=None, class_info=None, plot=False):
    cfg = get_config()
    if class_info is None:
        class_info = get_class_info(cfg["profiles"]["sdc"])
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

    if plot:
        if save_path is None:
            save_path = get_outimg_path()
        class_names = {key: value[0] for key, value in class_info.items()}
        plot_image_with_results(processed_img, results, class_names, class_info, save_path)

    return results


def process_image_from_path(image_path, conf=None, save_path=None, class_info=None, plot=False):
    import cv2

    if isinstance(class_info, bool) and plot is False:
        plot = class_info
        class_info = None

    img = cv2.imread(image_path)
    return process_image(img, conf, save_path, class_info, plot)


def process_single_location(x, y, z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset, class_info):
    new_z = z + z_offset
    if len(image.shape) == 3:
        image = np.max(image, axis=0)

    height, width = image.shape
    plot_enabled = get_config()["plotting"].get("enabled", True) and is_logging_enabled()
    img_path = get_outimg_path() if plot_enabled else None

    class_names = {key: value[0] for key, value in class_info.items()}
    results = process_image(
        image,
        None,
        img_path,
        class_info,
        plot=plot_enabled,
    )

    filtered_sorted_detections = filter_and_sort_detections(results, class_info)

    converted_results = []
    for result in filtered_sorted_detections:
        class_id, im_x, im_y, w, h, conf = result
        class_name = class_names[class_id]
        converted_results.append(
            image_cordinates_to_physical(
                x,
                y,
                im_x,
                im_y,
                width,
                height,
                new_z,
                xy_pixel_spacing,
                z_spacing,
                x_stage_direction,
                y_stage_direction,
                z_stage_direction,
                LLSM,
                class_id,
                conf,
                class_name,
            )
        )

    return converted_results


def image_cordinates_to_physical(x, y, im_x, im_y, w, h, new_z, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, class_id, conf, class_name):
    x_offset = im_x - w / 2
    y_offset = im_y - h / 2

    adjusted_x = x + x_offset * xy_pixel_spacing * x_stage_direction
    adjusted_y_direction = y_stage_direction
    if LLSM and get_config()["coordinate_conversion"]["llsm"].get("invert_y_stage_direction", True):
        adjusted_y_direction *= -1
    adjusted_y = y + y_offset * xy_pixel_spacing * adjusted_y_direction

    return [adjusted_x, adjusted_y, new_z, conf, class_id, class_name]


def merge_close_coordinates(coordinates, tolerance):
    unique_coords = []

    for coord in coordinates:
        x, y, z, conf, class_id, class_name = coord
        if not any(np.sqrt((x - uc[0])**2 + (y - uc[1])**2) <= tolerance for uc in unique_coords):
            unique_coords.append(coord)

    unique_coords = np.array(unique_coords, dtype=object)
    return unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2], list(unique_coords[:, 5])


def process_montage(X, Y, Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset, class_info):
    results = []
    image_array = np.array(image)
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        tmp = process_single_location(
            x,
            y,
            z,
            image_array[:, :, i],
            xy_pixel_spacing,
            z_spacing,
            x_stage_direction,
            y_stage_direction,
            z_stage_direction,
            LLSM,
            z_offset,
            class_info,
        )
        if tmp:
            for item in tmp:
                results.append(item)

    sorted_results = global_filter_and_sort_detections(results, class_info)

    if len(results) == 0:
        return np.array([]), np.array([]), np.array([]), []

    final_result = np.array(sorted_results, dtype=object)
    tolerance = get_config()["coordinate_conversion"].get("merge_tolerance_um", 20)
    new_X, new_Y, new_Z, new_class_names = merge_close_coordinates(final_result, tolerance)
    return new_X, new_Y, new_Z, new_class_names


def get_target_location(X, Y, Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM=False, z_offset=None):
    global experiment_path

    cfg = get_config()
    if z_offset is None:
        z_offset = cfg["coordinate_conversion"].get("default_z_offset_um", 0)

    if type(X) == float:
        X = [X]
        Y = [Y]
        Z = [Z]

    profile_config = get_profile_config(LLSM)
    class_info = get_class_info(profile_config)

    if is_logging_enabled():
        logging_directory = get_logging_directory()
        logging_directory.mkdir(parents=True, exist_ok=True)
        create_exp_folder(logging_directory)

    new_X, new_Y, new_Z, class_list = process_montage(
        X,
        Y,
        Z,
        image,
        xy_pixel_spacing,
        z_spacing,
        x_stage_direction,
        y_stage_direction,
        z_stage_direction,
        LLSM,
        z_offset,
        class_info,
    )

    N = len(new_X)

    if N == 0:
        no_detection_cfg = cfg["no_detection"]
        N = no_detection_cfg.get("n_returned_locations", 1)
        if no_detection_cfg.get("return_original_first_position", True):
            new_X = np.array([X[0]] * N)
            new_Y = np.array([Y[0]] * N)
            new_Z = np.array([Z[0]] * N)
            script_list = [no_detection_cfg.get("script", "donothing")] * N
            name_list = [no_detection_cfg.get("name", "nothing")] * N
            comment_list = [no_detection_cfg.get("comment", "nothing")] * N
        else:
            N = 0
            new_X = np.array([])
            new_Y = np.array([])
            new_Z = np.array([])
            script_list = []
            name_list = []
            comment_list = []
    else:
        script_list = [profile_config["highres_script"] for i in range(N)]
        name_list = [
            profile_config["name_template"].format(
                class_name=class_list[i],
                x=new_X[i],
                y=new_Y[i],
                z=new_Z[i],
            )
            for i in range(N)
        ]
        comment_list = [profile_config["highres_comment"] for i in range(N)]

    return N, new_X, new_Y, new_Z, script_list, name_list, comment_list
