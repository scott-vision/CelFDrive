"""Export reviewed tracking XML to YOLO, COCO, and cropped miniseries data.

Tracking boxes are normalized YOLO centre boxes; COCO and crop coordinates are
derived in image pixels, with X as columns and Y as rows.
"""

import logging
import os
import json
import hashlib
import shutil
import tempfile

from PIL import Image

from .tracking_xml import read_tracking_xml


LOGGER = logging.getLogger(__name__)


def tracking_xml_digest(tracking_xml_path):
    """Return the content fingerprint used to prove an export is current."""
    with open(tracking_xml_path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def export_manifest_path(output_dir):
    return os.path.join(output_dir, "export_manifest.json")


def write_export_manifest(output_dir, tracking_xml_path, box_type, export_type):
    with open(export_manifest_path(output_dir), "w", encoding="utf-8") as handle:
        json.dump({
            "tracking_xml": os.path.normpath(tracking_xml_path),
            "tracking_digest": tracking_xml_digest(tracking_xml_path),
            "box_type": box_type,
            "export_type": export_type,
        }, handle, indent=2)


def exported_labels_are_current(project_dir):
    """Return whether exported YOLO labels match current tracking XML."""
    tracking_xml = os.path.join(project_dir, "user_selections", "tracking_review.xml")
    manifest = export_manifest_path(os.path.join(project_dir, "user_selections", "exported_labels"))
    try:
        with open(manifest, encoding="utf-8") as handle:
            return json.load(handle).get("tracking_digest") == tracking_xml_digest(tracking_xml)
    except (OSError, ValueError):
        return False


def _normalize_slashes(path):
    return path.replace("\\", "/")


def _resolve_relative_label_path(image_path, dataset_root):
    norm_image = _normalize_slashes(os.path.normpath(image_path))
    if dataset_root:
        norm_root = _normalize_slashes(os.path.normpath(dataset_root))
        prefix = norm_root.rstrip("/") + "/images/"
        if norm_image.startswith(prefix):
            relative = norm_image[len(prefix):]
            return relative.rsplit(".", 1)[0] + ".txt"

    raise ValueError(
        "Tracking metadata must contain a dataset_root whose images directory contains "
        f"the exported image: {image_path!r}."
    )


def _resolve_relative_image_path(image_path, dataset_root):
    norm_image = _normalize_slashes(os.path.normpath(image_path))
    if dataset_root:
        norm_root = _normalize_slashes(os.path.normpath(dataset_root))
        prefix = norm_root.rstrip("/") + "/images/"
        if norm_image.startswith(prefix):
            return norm_image[len(prefix):]

    raise ValueError(
        "Tracking metadata must contain a dataset_root whose images directory contains "
        f"the exported image: {image_path!r}."
    )


def _choose_box(timepoint, box_type):
    boxes = timepoint.get("boxes", [])
    if not boxes:
        return None

    chosen_box = None
    if box_type == "preferred":
        preferred_box_type = timepoint.get("preferred_box_type")
        for box in boxes:
            if box["box_type"] == preferred_box_type:
                chosen_box = box
                break
    else:
        for box in boxes:
            if box["box_type"] == box_type:
                chosen_box = box
                break

    if chosen_box is None:
        requested_box_type = box_type if box_type != "preferred" else timepoint.get("preferred_box_type")
        raise ValueError(
            f"Requested box type {requested_box_type!r} is unavailable for image "
            f"{timepoint.get('image_path')!r}."
        )

    return chosen_box


def _yolo_box_to_coco_bbox(box, image_width, image_height):
    width = float(box["width"]) * image_width
    height = float(box["height"]) * image_height
    x_center = float(box["x_center"]) * image_width
    y_center = float(box["y_center"]) * image_height
    x_min = x_center - width / 2.0
    y_min = y_center - height / 2.0
    return [x_min, y_min, width, height]


def _sanitize_filename_part(text):
    safe = []
    for char in str(text):
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def _yolo_box_to_xyxy_pixels(box, image_width, image_height):
    width = float(box["width"]) * image_width
    height = float(box["height"]) * image_height
    x_center = float(box["x_center"]) * image_width
    y_center = float(box["y_center"]) * image_height
    x_min = x_center - width / 2.0
    y_min = y_center - height / 2.0
    x_max = x_center + width / 2.0
    y_max = y_center + height / 2.0
    return [x_min, y_min, x_max, y_max]


def _expand_xyxy(xyxy, image_width, image_height, padding_ratio=0.10):
    x_min, y_min, x_max, y_max = [float(value) for value in xyxy]
    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    return [
        max(0, int(round(x_min - pad_x))),
        max(0, int(round(y_min - pad_y))),
        min(int(image_width), int(round(x_max + pad_x))),
        min(int(image_height), int(round(y_max + pad_y))),
    ]


def export_tracking_xml_to_yolo(tracking_xml_path, output_dir, box_type="preferred"):
    """Export selected tracking boxes as YOLO label files.

    Parameters
    ----------
    tracking_xml_path : path-like
        Review XML containing ``dataset_root`` metadata.
    output_dir : path-like
        Label directory created beneath which image-relative ``.txt`` files are
        written.
    box_type : str, default="preferred"
        A named variant or ``"preferred"`` for each timepoint's selected box.

    Returns
    -------
    dict[str, list[str]]
        Image-relative label paths and YOLO lines, each containing class ID and
        normalized ``x_center y_center width height`` values.
    """
    tracking_data = read_tracking_xml(tracking_xml_path)
    metadata = tracking_data.get("metadata", {})
    dataset_root = metadata.get("dataset_root")

    labels_by_file = {}
    for track in tracking_data.get("tracks", []):
        for timepoint in track.get("timepoints", []):
            chosen_box = _choose_box(timepoint, box_type=box_type)
            if chosen_box is None:
                continue

            relative_label_path = _resolve_relative_label_path(timepoint["image_path"], dataset_root)
            labels_by_file.setdefault(relative_label_path, []).append(
                f"{timepoint['class_id']} {chosen_box['x_center']} {chosen_box['y_center']} {chosen_box['width']} {chosen_box['height']}"
            )

    output_dir = os.path.normpath(output_dir)
    parent = os.path.dirname(output_dir) or os.curdir
    staging_dir = tempfile.mkdtemp(prefix="celfdrive-export-", dir=parent)
    for relative_label_path, label_lines in labels_by_file.items():
        output_path = os.path.join(staging_dir, relative_label_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(label_lines) + "\n")
    write_export_manifest(staging_dir, tracking_xml_path, box_type, "yolo")
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.replace(staging_dir, output_dir)

    return labels_by_file


def export_tracking_xml_to_coco(tracking_xml_path, output_json_path, box_type="preferred"):
    """Export selected tracking boxes as a COCO annotation JSON document.

    Pixel-space COCO ``bbox`` values are ``[x_min, y_min, width, height]``;
    X is the image column and Y is the image row.
    """
    tracking_data = read_tracking_xml(tracking_xml_path)
    metadata = tracking_data.get("metadata", {})
    dataset_root = metadata.get("dataset_root")

    classes = tracking_data.get("classes", {})
    categories = [
        {"id": int(class_id), "name": class_name}
        for class_id, class_name in sorted(classes.items(), key=lambda item: int(item[0]))
    ]

    images = []
    annotations = []
    image_id_by_path = {}
    annotation_id = 1

    for track in tracking_data.get("tracks", []):
        for timepoint in track.get("timepoints", []):
            chosen_box = _choose_box(timepoint, box_type=box_type)
            if chosen_box is None:
                continue

            image_path = os.path.normpath(timepoint["image_path"])
            if image_path not in image_id_by_path:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size

                image_id = len(image_id_by_path) + 1
                image_id_by_path[image_path] = image_id
                images.append(
                    {
                        "id": image_id,
                        "file_name": _resolve_relative_image_path(image_path, dataset_root),
                        "width": image_width,
                        "height": image_height,
                    }
                )
            else:
                image_id = image_id_by_path[image_path]
                image_record = next(item for item in images if item["id"] == image_id)
                image_width = image_record["width"]
                image_height = image_record["height"]

            coco_bbox = _yolo_box_to_coco_bbox(chosen_box, image_width, image_height)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(timepoint["class_id"]),
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0,
                    "track_id": track["track_id"],
                    "timepoint_index": int(timepoint["timepoint_index"]),
                    "box_type": chosen_box["box_type"],
                }
            )
            annotation_id += 1

    coco_data = {
        "info": {
            "description": "COCO export generated from tracking_review.xml",
            "source_tracking_xml": os.path.normpath(tracking_xml_path),
            "box_type": box_type,
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_json_path = os.path.normpath(output_json_path)
    output_directory = os.path.dirname(output_json_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    temporary_path = output_json_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(coco_data, handle, indent=2)
    os.replace(temporary_path, output_json_path)
    write_export_manifest(output_directory or os.curdir, tracking_xml_path, box_type, "coco")

    return coco_data


def export_tracking_xml_to_miniseries(tracking_xml_path, output_dir, box_type="preferred", padding_ratio=0.10):
    """Write padded image crops for each selected tracking timepoint.

    Parameters
    ----------
    padding_ratio : float, default=0.10
        Fraction of each pixel-space box dimension added on every side.

    Returns
    -------
    list[str]
        Paths of PNG crops grouped into one directory per source track.
    """
    tracking_data = read_tracking_xml(tracking_xml_path)
    classes = tracking_data.get("classes", {})

    output_dir = os.path.normpath(output_dir)
    parent = os.path.dirname(output_dir) or os.curdir
    staging_dir = tempfile.mkdtemp(prefix="celfdrive-miniseries-", dir=parent)

    exported_images = []
    for track_index, track in enumerate(tracking_data.get("tracks", []), start=1):
        series_dir = os.path.join(staging_dir, str(track_index))
        os.makedirs(series_dir, exist_ok=True)

        for timepoint_index, timepoint in enumerate(track.get("timepoints", []), start=1):
            chosen_box = _choose_box(timepoint, box_type=box_type)
            if chosen_box is None:
                continue

            image_path = os.path.normpath(timepoint["image_path"])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Miniseries export failed: image path does not exist: `{image_path}`.")

            class_name = timepoint.get("phase_name") or classes.get(timepoint.get("class_id"), "unknown")
            class_name = _sanitize_filename_part(class_name)
            filename = f"miniseries_{track_index}_{timepoint_index:03d}_{class_name}.png"
            output_path = os.path.join(series_dir, filename)

            with Image.open(image_path) as image:
                image_width, image_height = image.size
                xyxy = _yolo_box_to_xyxy_pixels(chosen_box, image_width, image_height)
                crop_bounds = _expand_xyxy(xyxy, image_width, image_height, padding_ratio=padding_ratio)
                cropped = image.crop(tuple(crop_bounds))
                cropped.save(output_path)

            exported_images.append(os.path.join(output_dir, os.path.relpath(output_path, staging_dir)))

    write_export_manifest(staging_dir, tracking_xml_path, box_type, "miniseries")
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.replace(staging_dir, output_dir)

    return exported_images
