"""Generate SAM2 segmentation box variants for timepoints in tracking XML.

SAM2 receives pixel-space prompts derived from normalized YOLO tracking boxes.
"""

import logging
import os
import tempfile

from PIL import Image

from .tracking_xml import read_tracking_xml, write_tracking_data


DEFAULT_SAM2_MODEL = "sam2_b.pt"
DEFAULT_SAM2_DEVICE = os.environ.get("CELLCLICKER_SAM2_DEVICE", "cuda:0")
LOGGER = logging.getLogger(__name__)


def _ensure_ultralytics():
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for SAM2 integration. Install `ultralytics` in the active environment."
        ) from exc
    return SAM


def _supports_cuda_capability(device_capability, supported_arches):
    """Return whether a PyTorch wheel has a compatible CUDA cubin architecture.

    NVIDIA cubins are forward-compatible within a compute-capability major
    version: for example, an ``sm_86`` cubin can run on an ``sm_89`` device.
    ``torch.cuda.get_arch_list()`` therefore must not require an exact string
    match for minor architecture versions.
    """
    major, minor = device_capability
    architecture_prefix = f"sm_{major}"
    for architecture in supported_arches:
        if not architecture.startswith(architecture_prefix):
            continue
        architecture_minor = architecture[len(architecture_prefix):]
        if architecture_minor.isdigit() and int(architecture_minor) <= minor:
            return True
    return False


def _ensure_device_available(device):
    import torch

    if device is None:
        return

    device_text = str(device).lower()
    if device_text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"SAM2 was asked to use GPU device `{device}`, but this environment has CPU-only PyTorch. "
                "Install a CUDA-enabled PyTorch build and retry, or set CELLCLICKER_SAM2_DEVICE=cpu."
            )

        device_index = 0
        if ":" in device_text:
            device_index = int(device_text.split(":", 1)[1])

        capability = torch.cuda.get_device_capability(device_index)
        capability_tag = f"sm_{capability[0]}{capability[1]}"
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and not _supports_cuda_capability(capability, supported_arches):
            gpu_name = torch.cuda.get_device_name(device_index)
            supported = ", ".join(sorted(supported_arches))
            raise RuntimeError(
                f"SAM2 cannot run on `{device}` because this PyTorch build has no compatible CUDA architecture for {gpu_name} "
                f"({capability_tag}). This PyTorch build supports: {supported}.\n\n"
                "Install a newer CUDA PyTorch build, for example:\n"
                "python -m pip install --upgrade torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu128\n\n"
                "Temporary CPU workaround:\n"
                "set CELLCLICKER_SAM2_DEVICE=cpu"
            )


def _merge_xyxy_boxes(box_a, box_b):
    return [
        min(float(box_a[0]), float(box_b[0])),
        min(float(box_a[1]), float(box_b[1])),
        max(float(box_a[2]), float(box_b[2])),
        max(float(box_a[3]), float(box_b[3])),
    ]


def _pick_sam2_xyxy(timepoint, xyxy, confs):
    if len(xyxy) == 0:
        return None, None

    best_index = int(confs.argmax()) if confs is not None and len(confs) else 0
    best_xyxy = xyxy[best_index]
    best_conf = float(confs[best_index]) if confs is not None and len(confs) else None

    class_id = int(timepoint["class_id"])
    if class_id in [4, 5] and len(xyxy) > 1 and confs is not None and len(confs) > 1:
        sorted_indices = sorted(range(len(confs)), key=lambda idx: confs[idx], reverse=True)
        first_index = sorted_indices[0]
        second_index = sorted_indices[1]
        first_conf = float(confs[first_index])
        second_conf = float(confs[second_index])
        if second_conf >= 0.5 * first_conf:
            merged_xyxy = _merge_xyxy_boxes(xyxy[first_index], xyxy[second_index])
            LOGGER.info(
                "Applying SAM2 anaphase/telophase merge for image `%s` using top two boxes (conf %.4f, %.4f).",
                timepoint.get("image_path"),
                first_conf,
                second_conf,
            )
            return merged_xyxy, first_conf

    return best_xyxy, best_conf


def _pick_prompt_box(timepoint, prompt_box_type="original"):
    boxes = timepoint.get("boxes", [])
    for box in boxes:
        if box.get("box_type") == prompt_box_type:
            return box

    preferred_box_type = timepoint.get("preferred_box_type")
    for box in boxes:
        if box.get("box_type") == preferred_box_type:
            LOGGER.warning(
                "FALLBACK: requested SAM2 prompt box type `%s` missing for image `%s`; using preferred box type `%s` instead.",
                prompt_box_type,
                timepoint.get("image_path"),
                preferred_box_type,
            )
            return box

    if boxes:
        LOGGER.warning(
            "FALLBACK: requested SAM2 prompt box type `%s` and preferred box type `%s` missing for image `%s`; using first available box type `%s`.",
            prompt_box_type,
            preferred_box_type,
            timepoint.get("image_path"),
            boxes[0].get("box_type"),
        )
        return boxes[0]

    return None


def _yolo_xywh_to_xyxy_pixels(box, image_width, image_height):
    center_x = float(box["x_center"]) * image_width
    center_y = float(box["y_center"]) * image_height
    width = float(box["width"]) * image_width
    height = float(box["height"]) * image_height

    x1 = max(0.0, center_x - width / 2.0)
    y1 = max(0.0, center_y - height / 2.0)
    x2 = min(float(image_width), center_x + width / 2.0)
    y2 = min(float(image_height), center_y + height / 2.0)
    return [x1, y1, x2, y2]


def _xyxy_pixels_to_yolo_xywh(x1, y1, x2, y2, image_width, image_height):
    x1 = max(0.0, min(float(image_width), float(x1)))
    y1 = max(0.0, min(float(image_height), float(y1)))
    x2 = max(0.0, min(float(image_width), float(x2)))
    y2 = max(0.0, min(float(image_height), float(y2)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0

    return {
        "x_center": center_x / image_width,
        "y_center": center_y / image_height,
        "width": width / image_width,
        "height": height / image_height,
    }


def _xyxy_center_point(xyxy):
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _expand_crop_xyxy(xyxy, image_width, image_height, expansion_factor=1.6, min_crop_size=128):
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    crop_width = max(min_crop_size, width * expansion_factor)
    crop_height = max(min_crop_size, height * expansion_factor)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    crop_x1 = max(0.0, center_x - crop_width / 2.0)
    crop_y1 = max(0.0, center_y - crop_height / 2.0)
    crop_x2 = min(float(image_width), center_x + crop_width / 2.0)
    crop_y2 = min(float(image_height), center_y + crop_height / 2.0)
    return [crop_x1, crop_y1, crop_x2, crop_y2]


def _offset_xyxy_boxes(xyxy, crop_x1, crop_y1):
    adjusted = xyxy.copy()
    adjusted[:, 0] += crop_x1
    adjusted[:, 1] += crop_y1
    adjusted[:, 2] += crop_x1
    adjusted[:, 3] += crop_y1
    return adjusted


def _upsert_sam2_box(timepoint, sam2_box, overwrite=True):
    boxes = timepoint.setdefault("boxes", [])
    for index, existing_box in enumerate(boxes):
        if existing_box.get("box_type") == "sam2":
            if overwrite:
                boxes[index] = sam2_box
                return "updated"
            return "skipped_existing"

    boxes.append(sam2_box)
    return "created"


def _has_box_type(timepoint, box_type):
    for box in timepoint.get("boxes", []):
        if box.get("box_type") == box_type:
            return True
    return False


def predict_sam2_merged_box_from_points(
    image_path,
    point_pixels,
    crop_xyxy,
    image_size,
    model_name=DEFAULT_SAM2_MODEL,
    conf=0.0,
    device=DEFAULT_SAM2_DEVICE,
):
    """Predict a normalized tracking box from user-selected pixel points.

    ``point_pixels`` are full-image ``(x, y)`` coordinates and ``crop_xyxy`` is
    a full-image pixel crop. The returned ``tightened`` box is normalized YOLO.
    """
    if not point_pixels:
        raise ValueError("At least one point is required for SAM2 point prompting.")

    image_path = os.path.normpath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"SAM2 failed: image path does not exist: `{image_path}`.")

    SAM = _ensure_ultralytics()
    _ensure_device_available(device)
    model = SAM(model_name)

    image_width, image_height = image_size
    crop_x1, crop_y1, crop_x2, crop_y2 = [int(round(value)) for value in crop_xyxy]
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("SAM2 failed: crop bounds were invalid.")

    with Image.open(image_path) as image:
        crop_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    crop_points = []
    for point_x, point_y in point_pixels:
        crop_points.append([float(point_x) - crop_x1, float(point_y) - crop_y1])

    merged_xyxy = None
    best_conf = None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_handle:
        temp_crop_path = temp_handle.name
    try:
        crop_image.save(temp_crop_path)
        for crop_point in crop_points:
            results = model.predict(
                temp_crop_path,
                points=[crop_point],
                labels=[1],
                conf=conf,
                device=device,
                verbose=False,
            )
            if not results:
                LOGGER.warning(
                    "FALLBACK: Ultralytics SAM2 returned no results for image `%s` at point `(%.1f, %.1f)`; skipping that point.",
                    image_path,
                    crop_point[0],
                    crop_point[1],
                )
                continue

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                LOGGER.warning(
                    "FALLBACK: Ultralytics SAM2 returned no boxes for image `%s` at point `(%.1f, %.1f)`; skipping that point.",
                    image_path,
                    crop_point[0],
                    crop_point[1],
                )
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            point_xyxy, point_conf = _pick_sam2_xyxy({"class_id": 0, "image_path": image_path}, xyxy, confs)
            if point_xyxy is None:
                LOGGER.warning(
                    "FALLBACK: SAM2 point-box selection returned no usable box for image `%s` at point `(%.1f, %.1f)`; skipping that point.",
                    image_path,
                    crop_point[0],
                    crop_point[1],
                )
                continue

            adjusted_xyxy = [
                float(point_xyxy[0]) + crop_x1,
                float(point_xyxy[1]) + crop_y1,
                float(point_xyxy[2]) + crop_x1,
                float(point_xyxy[3]) + crop_y1,
            ]
            merged_xyxy = adjusted_xyxy if merged_xyxy is None else _merge_xyxy_boxes(merged_xyxy, adjusted_xyxy)
            if point_conf is not None:
                best_conf = point_conf if best_conf is None else max(best_conf, point_conf)
    finally:
        if os.path.exists(temp_crop_path):
            os.remove(temp_crop_path)

    if merged_xyxy is None:
        raise RuntimeError("SAM2 did not return a usable box for any selected point.")

    yolo_box = _xyxy_pixels_to_yolo_xywh(
        merged_xyxy[0],
        merged_xyxy[1],
        merged_xyxy[2],
        merged_xyxy[3],
        image_width,
        image_height,
    )
    source = f"ultralytics_sam2:{model_name}:points_to_box"
    if best_conf is not None:
        source = f"{source}:conf={best_conf:.4f}"
    return {
        "box_type": "tightened",
        "format": "yolo_xywh_norm",
        "x_center": yolo_box["x_center"],
        "y_center": yolo_box["y_center"],
        "width": yolo_box["width"],
        "height": yolo_box["height"],
        "source": source,
    }


def run_sam2_on_tracking_xml(
    tracking_xml_path,
    model_name=DEFAULT_SAM2_MODEL,
    prompt_box_type="original",
    overwrite=True,
    conf=0.0,
    progress_callback=None,
    device=DEFAULT_SAM2_DEVICE,
    save_every=1,
):
    """Generate SAM2 box variants and save the tracking XML incrementally.

    Prompts are normalized tracking boxes transformed to pixels for SAM2;
    returned variants are normalized ``sam2`` boxes. Per-image failures are
    logged and counted in the returned statistics.
    """
    tracking_xml_path = os.path.normpath(tracking_xml_path)
    tracking_data = read_tracking_xml(tracking_xml_path)
    tracking_data.setdefault("box_types", {})
    tracking_data["box_types"]["sam2"] = "Box adjusted by SAM2."

    SAM = _ensure_ultralytics()
    _ensure_device_available(device)
    model = SAM(model_name)

    stats = {
        "tracks": len(tracking_data.get("tracks", [])),
        "timepoints": 0,
        "processed": 0,
        "created": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
    }

    total_timepoints = sum(len(track.get("timepoints", [])) for track in tracking_data.get("tracks", []))
    current_timepoint = 0
    pending_saves = 0

    for track in tracking_data.get("tracks", []):
        for timepoint in track.get("timepoints", []):
            current_timepoint += 1
            if progress_callback is not None:
                progress_callback(current_timepoint, total_timepoints, timepoint.get("image_path"))
            stats["timepoints"] += 1

            if not overwrite and _has_box_type(timepoint, "sam2"):
                LOGGER.warning(
                    "FALLBACK: existing SAM2 box preserved for image `%s` because overwrite is disabled.",
                    timepoint.get("image_path"),
                )
                stats["skipped"] += 1
                continue

            prompt_box = _pick_prompt_box(timepoint, prompt_box_type=prompt_box_type)
            if prompt_box is None:
                LOGGER.warning(
                    "FALLBACK: no promptable boxes exist for image `%s`; skipping SAM2 generation for this timepoint.",
                    timepoint.get("image_path"),
                )
                stats["skipped"] += 1
                continue

            image_path = os.path.normpath(timepoint["image_path"])
            if not os.path.exists(image_path):
                LOGGER.error("SAM2 failed: image path does not exist: `%s`.", image_path)
                stats["failed"] += 1
                continue

            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
                    prompt_bbox = _yolo_xywh_to_xyxy_pixels(prompt_box, image_width, image_height)
                    crop_xyxy = _expand_crop_xyxy(prompt_bbox, image_width, image_height)
                    crop_x1, crop_y1, crop_x2, crop_y2 = [int(round(value)) for value in crop_xyxy]
                    crop_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    crop_prompt_bbox = [
                        prompt_bbox[0] - crop_x1,
                        prompt_bbox[1] - crop_y1,
                        prompt_bbox[2] - crop_x1,
                        prompt_bbox[3] - crop_y1,
                    ]
                    crop_prompt_point = _xyxy_center_point(crop_prompt_bbox)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_handle:
                    temp_crop_path = temp_handle.name
                try:
                    crop_image.save(temp_crop_path)
                    class_id = int(timepoint["class_id"])
                    if class_id in [4, 5]:
                        results = model.predict(
                            temp_crop_path,
                            bboxes=[crop_prompt_bbox],
                            conf=conf,
                            device=device,
                            verbose=False,
                        )
                    else:
                        results = model.predict(
                            temp_crop_path,
                            points=[crop_prompt_point],
                            labels=[1],
                            conf=conf,
                            device=device,
                            verbose=False,
                        )
                finally:
                    if os.path.exists(temp_crop_path):
                        os.remove(temp_crop_path)

                if not results:
                    LOGGER.warning(
                        "FALLBACK: Ultralytics SAM2 returned no results for image `%s`; leaving existing boxes unchanged.",
                        image_path,
                    )
                    stats["skipped"] += 1
                    continue

                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    LOGGER.warning(
                        "FALLBACK: Ultralytics SAM2 returned no boxes for image `%s`; leaving existing boxes unchanged.",
                        image_path,
                    )
                    stats["skipped"] += 1
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                xyxy = _offset_xyxy_boxes(xyxy, crop_x1, crop_y1)
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                best_xyxy, best_conf = _pick_sam2_xyxy(timepoint, xyxy, confs)
                if best_xyxy is None:
                    LOGGER.warning(
                        "FALLBACK: SAM2 box selection returned no usable box for image `%s`; leaving existing boxes unchanged.",
                        image_path,
                    )
                    stats["skipped"] += 1
                    continue
                yolo_box = _xyxy_pixels_to_yolo_xywh(
                    best_xyxy[0],
                    best_xyxy[1],
                    best_xyxy[2],
                    best_xyxy[3],
                    image_width,
                    image_height,
                )

                prompt_mode = "bbox" if int(timepoint["class_id"]) in [4, 5] else "point_center"
                source = f"ultralytics_sam2:{model_name}:{prompt_mode}"
                if best_conf is not None:
                    source = f"{source}:conf={best_conf:.4f}"

                sam2_box = {
                    "box_type": "sam2",
                    "format": "yolo_xywh_norm",
                    "x_center": yolo_box["x_center"],
                    "y_center": yolo_box["y_center"],
                    "width": yolo_box["width"],
                    "height": yolo_box["height"],
                    "source": source,
                }
                result = _upsert_sam2_box(timepoint, sam2_box, overwrite=overwrite)
                if result == "created":
                    stats["created"] += 1
                elif result == "updated":
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
                    continue

                stats["processed"] += 1
                pending_saves += 1
                if pending_saves >= max(1, int(save_every)):
                    write_tracking_data(tracking_xml_path, tracking_data)
                    pending_saves = 0
            except Exception:
                LOGGER.exception("SAM2 generation failed for image `%s`.", image_path)
                stats["failed"] += 1

    write_tracking_data(tracking_xml_path, tracking_data)
    return stats
