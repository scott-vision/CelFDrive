"""Build reviewable tracking XML from CellClicker and phase-selection XML.

Tracking boxes use YOLO-normalized ``(x_center, y_center, width, height)``
coordinates, while image paths identify individual timepoints.
"""

import logging
import os
import xml.etree.ElementTree as ET

from .clicker_utils import get_relative_image_name
from .convert_selections_multiphase import (
    adjust_bbox_via_threshold,
    parse_xml_for_labels,
    parse_xml_for_phases,
)
from .tracking_xml import DEFAULT_BOX_TYPES, DEFAULT_CLASSES, write_tracking_xml
from .project_reconciliation import reconcile_tracking_records
from .workflow_state import raw_revision_fingerprint, raw_track_revisions


LOGGER = logging.getLogger(__name__)


PHASE_ORDER = [
    "prophase",
    "earlyprometaphase",
    "prometaphase",
    "metaphase",
    "anaphase",
    "telophase",
]


def resolve_image_path(anchor_path, step_back, dataset_root=None):
    """Resolve a series-relative image path, optionally beneath a project root.

    ``step_back`` is the source frame/class index used by legacy CellClicker
    series naming. Raises ``ValueError`` if a requested project root cannot
    contain the resolved image beneath its ``images`` directory.
    """
    image_path = get_relative_image_name(anchor_path, step_back)
    if image_path is None:
        return None

    if dataset_root and "/images/" in image_path.replace("\\", "/"):
        suffix = image_path.replace("\\", "/").split("/images/", 1)[1]
        return os.path.normpath(os.path.join(dataset_root, "images", suffix))

    if dataset_root:
        raise ValueError(
            "The tracking source path could not be resolved beneath the selected "
            f"project's images directory: {anchor_path!r}."
        )
    return os.path.normpath(image_path)


def get_phase_for_index(index_in_time_order, phases):
    """Return the active phase for a zero-based chronological frame index."""
    selected_phase = None
    for phase in PHASE_ORDER:
        start_index = phases.get(phase, -1)
        if start_index != -1 and index_in_time_order >= start_index:
            selected_phase = phase
    return selected_phase


def make_box(box_type, x_center, y_center, width, height, source):
    """Create a normalized YOLO centre-box record for tracking XML.

    Coordinates are fractions of image width/height, not pixel values.
    """
    return {
        "box_type": box_type,
        "format": "yolo_xywh_norm",
        "x_center": float(x_center),
        "y_center": float(y_center),
        "width": float(width),
        "height": float(height),
        "source": source,
    }


def build_track_record(track_index, anchor_path, series_id, labels, phases, dataset_root=None, include_otsu=False, raw_revision=0):
    """Build one chronological review track from legacy label and phase records.

    ``labels`` provides normalized YOLO boxes; ``phases`` maps phase names to
    selected chronological indices. Optional Otsu generation reads each image.
    """
    sorted_labels = sorted(labels, key=lambda item: int(item["class_id"]), reverse=True)
    timepoints = []

    for timepoint_index, label in enumerate(sorted_labels):
        source_class_id = int(label["class_id"])
        image_path = resolve_image_path(anchor_path, source_class_id, dataset_root=dataset_root)
        if image_path is None:
            continue

        phase_name = get_phase_for_index(timepoint_index, phases)
        if phase_name is None:
            continue

        class_id = next(
            class_key for class_key, class_name in DEFAULT_CLASSES.items() if class_name == phase_name
        )

        boxes = [
            make_box(
                "original",
                label["x_center"],
                label["y_center"],
                label["width"],
                label["height"],
                "cell_reigons.xml",
            )
        ]

        if include_otsu:
            otsu_label = adjust_bbox_via_threshold(
                image_path,
                class_id,
                label["x_center"],
                label["y_center"],
                label["width"],
                label["height"],
            )
            boxes.append(
                make_box(
                    "otsu",
                    otsu_label[1],
                    otsu_label[2],
                    otsu_label[3],
                    otsu_label[4],
                    "otsu_threshold",
                )
            )

        timepoints.append(
            {
                "timepoint_index": timepoint_index,
                "frame_index": timepoint_index,
                "image_path": image_path,
                "class_id": class_id,
                "phase_name": phase_name,
                "source_class_id": source_class_id,
                "preferred_box_type": "original",
                "boxes": boxes,
            }
        )

    return {
        "track_id": f"T{track_index:05d}",
        "source_path": anchor_path,
        "series_id": str(series_id),
        "raw_revision": int(raw_revision),
        "review_state": "pending",
        "timepoints": timepoints,
    }


def build_tracking_records(phase_xml, cell_regions_xml, dataset_root=None, include_otsu=False):
    """Build non-empty tracking records from phase and cell-region XML files."""
    phase_data = parse_xml_for_phases(phase_xml)
    label_data = parse_xml_for_labels(cell_regions_xml)
    revisions = raw_track_revisions(cell_regions_xml)

    tracks = []
    track_index = 1

    for (anchor_path, series_id), labels in label_data.items():
        phases = phase_data.get((anchor_path, series_id), {})
        phases = {key: value for key, value in phases.items() if value != -1}
        if not phases:
            continue

        track = build_track_record(
            track_index,
            anchor_path,
            series_id,
            labels,
            phases,
            dataset_root=dataset_root,
            include_otsu=include_otsu,
            raw_revision=revisions.get((anchor_path, str(series_id)), 0),
        )
        if track["timepoints"]:
            tracks.append(track)
            track_index += 1

    return tracks


def build_tracking_xml(phase_xml, cell_regions_xml, output_xml, dataset_root=None, include_otsu=False):
    """Create tracking XML and return its in-memory track records.

    The output document stores source paths, phase-derived class IDs, and
    normalized YOLO variants. ``include_otsu`` performs image thresholding.
    """
    aggregate_root = ET.parse(phase_xml).getroot()
    aggregate_fingerprint = aggregate_root.get("raw_revision_fingerprint")
    current_fingerprint = raw_revision_fingerprint(cell_regions_xml)
    if aggregate_fingerprint is not None and aggregate_fingerprint != current_fingerprint:
        raise ValueError("Aggregated phase selections are stale. Re-run aggregation before building tracking review.")
    if aggregate_fingerprint is None and any(raw_track_revisions(cell_regions_xml).values()):
        raise ValueError("Aggregated phase selections lack raw revision provenance. Re-run aggregation before building tracking review.")

    metadata = {
        "phase_xml": os.path.normpath(phase_xml),
        "cell_regions_xml": os.path.normpath(cell_regions_xml),
    }
    if dataset_root:
        metadata["dataset_root"] = os.path.normpath(dataset_root)

    tracks = build_tracking_records(
        phase_xml,
        cell_regions_xml,
        dataset_root=dataset_root,
        include_otsu=include_otsu,
    )
    tracks, metadata = reconcile_tracking_records(tracks, output_xml, cell_regions_xml, metadata)
    write_tracking_xml(
        output_xml,
        tracks,
        classes=DEFAULT_CLASSES,
        box_types=DEFAULT_BOX_TYPES,
        metadata=metadata,
    )
    return tracks
