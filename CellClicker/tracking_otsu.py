"""Generate Otsu-threshold box variants for timepoints in tracking XML."""

import logging
import os

from .convert_selections_multiphase import adjust_bbox_via_threshold
from .tracking_xml import read_tracking_xml, write_tracking_data


LOGGER = logging.getLogger(__name__)


def _pick_prompt_box(timepoint, prompt_box_type="original"):
    boxes = timepoint.get("boxes", [])
    for box in boxes:
        if box.get("box_type") == prompt_box_type:
            return box

    preferred_box_type = timepoint.get("preferred_box_type")
    for box in boxes:
        if box.get("box_type") == preferred_box_type:
            LOGGER.warning(
                "FALLBACK: requested Otsu prompt box type `%s` missing for image `%s`; using preferred box type `%s` instead.",
                prompt_box_type,
                timepoint.get("image_path"),
                preferred_box_type,
            )
            return box

    if boxes:
        LOGGER.warning(
            "FALLBACK: requested Otsu prompt box type `%s` and preferred box type `%s` missing for image `%s`; using first available box type `%s`.",
            prompt_box_type,
            preferred_box_type,
            timepoint.get("image_path"),
            boxes[0].get("box_type"),
        )
        return boxes[0]

    return None


def _upsert_otsu_box(timepoint, otsu_box, overwrite=True):
    boxes = timepoint.setdefault("boxes", [])
    for index, existing_box in enumerate(boxes):
        if existing_box.get("box_type") == "otsu":
            if overwrite:
                boxes[index] = otsu_box
                return "updated"
            return "skipped_existing"

    boxes.append(otsu_box)
    return "created"


def run_otsu_on_tracking_xml(tracking_xml_path, prompt_box_type="original", overwrite=True, progress_callback=None):
    """Generate normalized Otsu box variants for every review timepoint.

    Parameters
    ----------
    tracking_xml_path : path-like
        Tracking XML updated in place.
    prompt_box_type : str, default="original"
        Existing normalized YOLO box used to constrain thresholding.
    overwrite : bool, default=True
        Replace existing ``otsu`` variants when present.
    progress_callback : callable, optional
        Called as ``(current, total, image_path)`` from the processing loop.

    Returns
    -------
    dict[str, int]
        Counts for processed, created, updated, skipped, and failed timepoints.
    """
    tracking_xml_path = os.path.normpath(tracking_xml_path)
    tracking_data = read_tracking_xml(tracking_xml_path)
    tracking_data.setdefault("box_types", {})
    tracking_data["box_types"]["otsu"] = "Box adjusted by Otsu thresholding."

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

    for track in tracking_data.get("tracks", []):
        for timepoint in track.get("timepoints", []):
            current_timepoint += 1
            if progress_callback is not None:
                progress_callback(current_timepoint, total_timepoints, timepoint.get("image_path"))
            stats["timepoints"] += 1
            prompt_box = _pick_prompt_box(timepoint, prompt_box_type=prompt_box_type)
            if prompt_box is None:
                LOGGER.warning(
                    "FALLBACK: no promptable boxes exist for image `%s`; skipping Otsu generation for this timepoint.",
                    timepoint.get("image_path"),
                )
                stats["skipped"] += 1
                continue

            image_path = os.path.normpath(timepoint["image_path"])
            if not os.path.exists(image_path):
                LOGGER.error("Otsu generation failed: image path does not exist: `%s`.", image_path)
                stats["failed"] += 1
                continue

            try:
                otsu_label = adjust_bbox_via_threshold(
                    image_path,
                    int(timepoint["class_id"]),
                    prompt_box["x_center"],
                    prompt_box["y_center"],
                    prompt_box["width"],
                    prompt_box["height"],
                )
                otsu_box = {
                    "box_type": "otsu",
                    "format": "yolo_xywh_norm",
                    "x_center": float(otsu_label[1]),
                    "y_center": float(otsu_label[2]),
                    "width": float(otsu_label[3]),
                    "height": float(otsu_label[4]),
                    "source": "otsu_threshold",
                }
                result = _upsert_otsu_box(timepoint, otsu_box, overwrite=overwrite)
                if result == "created":
                    stats["created"] += 1
                elif result == "updated":
                    stats["updated"] += 1
                else:
                    LOGGER.warning(
                        "FALLBACK: existing Otsu box preserved for image `%s` because overwrite is disabled.",
                        image_path,
                    )
                    stats["skipped"] += 1
                    continue

                stats["processed"] += 1
            except Exception:
                LOGGER.exception("Otsu generation failed for image `%s`.", image_path)
                stats["failed"] += 1

    write_tracking_data(tracking_xml_path, tracking_data)
    return stats
