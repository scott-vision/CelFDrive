"""Reconcile raw CellClicker changes with project workflow records."""

import os
import tempfile
import xml.etree.ElementTree as ET

from .tracking_xml import read_tracking_xml, write_tracking_data
from .workflow_state import (
    annotator_selection_files,
    raw_revision_fingerprint,
    raw_track_revisions,
    selection_fingerprint,
)
from .project_paths import resolve_cell_regions_xml


def _remove_selection_entry(tree, track_key):
    root = tree.getroot()
    changed = False
    for entry in list(root.findall("DataEntry")):
        key = (entry.findtext("PathName"), str(entry.findtext("SeriesID")))
        if key == track_key:
            root.remove(entry)
            changed = True
    return changed


def _stage_tree(path, tree):
    descriptor, temporary_path = tempfile.mkstemp(prefix="celfdrive-delete-", suffix=".xml", dir=os.path.dirname(path) or None)
    os.close(descriptor)
    tree.write(temporary_path, encoding="utf-8", xml_declaration=True)
    return temporary_path


def _stage_tracking(path, tracking_data):
    descriptor, temporary_path = tempfile.mkstemp(prefix="celfdrive-delete-", suffix=".xml", dir=os.path.dirname(path) or None)
    os.close(descriptor)
    write_tracking_data(temporary_path, tracking_data)
    return temporary_path


def delete_track_from_project(project_dir, anchor_path, series_id):
    """Remove one raw track and its canonical downstream records.

    Export directories are deliberately retained. ``tracking_review.xml`` is
    marked stale so consumers require a fresh export snapshot.
    """
    if not project_dir:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(anchor_path)))
    project_dir = os.path.normpath(project_dir)
    cell_xml = str(resolve_cell_regions_xml(project_dir).path)
    selections_dir = os.path.join(project_dir, "user_selections")
    key = (anchor_path, str(series_id))

    # Build every replacement first, so a parse or write failure changes none.
    documents = annotator_selection_files(selections_dir)
    aggregate_xml = os.path.join(selections_dir, "aggregated_tracking.xml")
    tracking_xml = os.path.join(selections_dir, "tracking_review.xml")
    tracking_data = read_tracking_xml(tracking_xml) if os.path.isfile(tracking_xml) else None
    raw_tree = ET.parse(cell_xml)
    raw_root = raw_tree.getroot()
    removed = False
    for path_elem in list(raw_root.findall("path")):
        if path_elem.findtext("name") != anchor_path:
            continue
        for series in list(path_elem.findall("series")):
            if series.get("id") == str(series_id):
                path_elem.remove(series)
                removed = True
        if not path_elem.findall("series"):
            raw_root.remove(path_elem)
    if not removed:
        raise ValueError(f"CellClicker series {series_id} was not found for deletion.")

    updates = [(cell_xml, _stage_tree(cell_xml, raw_tree))]
    new_raw_fingerprint = raw_revision_fingerprint(updates[0][1])
    try:
        for document in documents:
            tree = ET.parse(document)
            if _remove_selection_entry(tree, key):
                updates.append((document, _stage_tree(document, tree)))
        if os.path.isfile(aggregate_xml):
            tree = ET.parse(aggregate_xml)
            if _remove_selection_entry(tree, key):
                tree.getroot().set("raw_revision_fingerprint", new_raw_fingerprint)
                updates.append((aggregate_xml, _stage_tree(aggregate_xml, tree)))
        if tracking_data is not None:
            tracking_data["tracks"] = [
                track for track in tracking_data["tracks"]
                if (track.get("source_path"), str(track.get("series_id"))) != key
            ]
            tracking_data.setdefault("metadata", {})["exports_stale"] = "true"
            # Calculate from a temporary raw snapshot by reading the staged file.
            tracking_data["metadata"]["raw_revision_fingerprint"] = new_raw_fingerprint
            updates.append((tracking_xml, _stage_tracking(tracking_xml, tracking_data)))
    except Exception:
        for _, temporary_path in updates:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        raise

    for destination, temporary_path in updates:
        os.replace(temporary_path, destination)


def reconcile_tracking_records(candidate_tracks, output_xml, cell_regions_xml, metadata):
    """Merge rebuilt source tracks with retained review-box decisions."""
    existing = read_tracking_xml(output_xml) if os.path.isfile(output_xml) else None
    existing_by_key = {}
    if existing:
        existing_by_key = {
            (track.get("source_path"), str(track.get("series_id"))): track
            for track in existing.get("tracks", [])
        }

    reconciled = []
    for candidate in candidate_tracks:
        key = (candidate["source_path"], str(candidate["series_id"]))
        prior = existing_by_key.get(key)
        if prior is None:
            candidate["review_state"] = "pending"
            reconciled.append(candidate)
            continue

        prior_points = {point["source_class_id"]: point for point in prior.get("timepoints", [])}
        revised = int(prior.get("raw_revision", 0)) != int(candidate.get("raw_revision", 0))
        for point in candidate["timepoints"]:
            previous = prior_points.get(point["source_class_id"])
            if previous:
                point["boxes"] = previous.get("boxes", point["boxes"])
                point["preferred_box_type"] = previous.get("preferred_box_type", point["preferred_box_type"])
        candidate["track_id"] = prior.get("track_id", candidate["track_id"])
        candidate["review_state"] = "pending" if revised else prior.get("review_state", "reviewed")
        reconciled.append(candidate)

    metadata = dict(metadata)
    metadata["raw_revision_fingerprint"] = raw_revision_fingerprint(cell_regions_xml)
    metadata["exports_stale"] = "true" if existing else "false"
    return reconciled, metadata


def tracking_is_current(tracking_xml, cell_regions_xml):
    """Return whether tracking XML was reconciled against current raw revisions."""
    if not os.path.isfile(tracking_xml):
        return False
    fingerprint = read_tracking_xml(tracking_xml).get("metadata", {}).get("raw_revision_fingerprint")
    selection_digest = read_tracking_xml(tracking_xml).get("metadata", {}).get("selection_fingerprint")
    if selection_digest is not None:
        selections_dir = os.path.dirname(os.path.abspath(tracking_xml))
        if selection_digest != selection_fingerprint(annotator_selection_files(selections_dir)):
            return False
    if fingerprint is None:
        # Projects built before revision provenance are safe until a raw series
        # is actually extended: all legacy raw tracks have revision zero.
        return not any(raw_track_revisions(cell_regions_xml).values())
    return fingerprint == raw_revision_fingerprint(cell_regions_xml)
