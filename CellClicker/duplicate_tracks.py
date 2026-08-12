"""Detect likely duplicate reviewed tracks before dataset export."""

from collections import defaultdict


def _preferred_box(timepoint):
    preferred_type = timepoint.get("preferred_box_type")
    for box in timepoint.get("boxes", []):
        if box.get("box_type") == preferred_type:
            return box
    return None


def _iou(first_box, second_box):
    """Return IoU for normalized YOLO centre boxes."""
    def bounds(box):
        half_width = float(box["width"]) / 2
        half_height = float(box["height"]) / 2
        return (
            float(box["x_center"]) - half_width,
            float(box["y_center"]) - half_height,
            float(box["x_center"]) + half_width,
            float(box["y_center"]) + half_height,
        )

    first_left, first_top, first_right, first_bottom = bounds(first_box)
    second_left, second_top, second_right, second_bottom = bounds(second_box)
    overlap_width = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    overlap_height = max(0.0, min(first_bottom, second_bottom) - max(first_top, second_top))
    overlap = overlap_width * overlap_height
    first_area = max(0.0, first_right - first_left) * max(0.0, first_bottom - first_top)
    second_area = max(0.0, second_right - second_left) * max(0.0, second_bottom - second_top)
    union = first_area + second_area - overlap
    return overlap / union if union else 0.0


def find_near_duplicate_tracks(tracks, minimum_iou=0.9, minimum_shared_frames=2):
    """Return reviewed-track pairs with near-identical preferred labels.

    Candidates need matching class IDs and high-IoU preferred boxes in at least
    two shared images. This deliberately avoids flagging adjacent cells that
    overlap in just one frame.
    """
    frames_by_track = []
    for track in tracks:
        frames = {}
        for point in track.get("timepoints", []):
            box = _preferred_box(point)
            if box is not None:
                frames[point.get("image_path")] = (int(point.get("class_id")), box)
        frames_by_track.append(frames)

    candidates = []
    for first_index, first_track in enumerate(tracks):
        for second_index in range(first_index + 1, len(tracks)):
            second_track = tracks[second_index]
            shared_ious = []
            for image_path in frames_by_track[first_index].keys() & frames_by_track[second_index].keys():
                first_class, first_box = frames_by_track[first_index][image_path]
                second_class, second_box = frames_by_track[second_index][image_path]
                if first_class == second_class:
                    shared_ious.append(_iou(first_box, second_box))
            close_ious = [value for value in shared_ious if value >= minimum_iou]
            if len(close_ious) >= minimum_shared_frames:
                candidates.append({
                    "first": first_track,
                    "second": second_track,
                    "shared_frames": len(close_ious),
                    "mean_iou": sum(close_ious) / len(close_ious),
                })
    return candidates
