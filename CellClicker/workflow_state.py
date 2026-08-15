"""Revision and provenance helpers for the CellClicker project workflow."""

import hashlib
import os
import xml.etree.ElementTree as ET


PHASES = (
    "prophase", "earlyprometaphase", "prometaphase",
    "metaphase", "anaphase", "telophase",
)
RESERVED_SELECTION_FILENAMES = {
    "aggregated_tracking.xml", "tracking_review.xml", "polled.xml",
}


def raw_track_revisions(cell_regions_xml):
    """Return ``(anchor_path, series_id) -> raw revision`` from region XML."""
    root = ET.parse(cell_regions_xml).getroot()
    revisions = {}
    for path_elem in root.findall("path"):
        anchor = path_elem.findtext("name")
        for series in path_elem.findall("series"):
            try:
                revision = int(series.get("revision", "0"))
            except ValueError as exc:
                raise ValueError(f"Invalid revision on CellClicker series {series.get('id')!r}.") from exc
            revisions[(anchor, str(series.get("id")))] = revision
    return revisions


def raw_revision_fingerprint(cell_regions_xml):
    """Return a deterministic fingerprint of current raw track revisions."""
    values = raw_track_revisions(cell_regions_xml)
    source = "\n".join(f"{path}\t{series}\t{revision}" for (path, series), revision in sorted(values.items()))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def is_annotator_selection_xml(xml_path):
    """Whether ``xml_path`` is a user phase-selection document, not an output."""
    if os.path.basename(xml_path).lower() in RESERVED_SELECTION_FILENAMES:
        return False
    try:
        return ET.parse(xml_path).getroot().tag == "Data"
    except (ET.ParseError, OSError):
        return False


def annotator_selection_files(selections_dir):
    """Return direct phase-selection XML files in deterministic order."""
    if not os.path.isdir(selections_dir):
        return []
    paths = [
        os.path.join(selections_dir, name)
        for name in os.listdir(selections_dir)
        if name.lower().endswith(".xml")
    ]
    return sorted(path for path in paths if is_annotator_selection_xml(path))


def selection_fingerprint(selection_paths):
    """Return a stable digest of the current annotator phase-selection documents."""
    digest = hashlib.sha256()
    for path in sorted(os.path.normcase(os.path.abspath(path)) for path in selection_paths):
        digest.update(path.encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def entry_is_current(entry, revision, phases=PHASES, phase_signature=None):
    """Return whether one annotator entry is complete for a raw revision."""
    if entry is None:
        return False
    try:
        selected_revision = int(entry.get("selection_revision", "0"))
    except ValueError:
        return False
    if selected_revision != revision:
        return False
    if phase_signature is not None and entry.get("phase_signature") != phase_signature:
        return False
    for phase in phases:
        value = entry.findtext(phase)
        if value is None:
            return False
        try:
            int(value)
        except ValueError:
            return False
    return True


def selection_entries_by_track(xml_path):
    """Return phase selection entries keyed by their raw source track."""
    root = ET.parse(xml_path).getroot()
    entries = {}
    for entry in root.findall("DataEntry"):
        path = entry.findtext("PathName")
        series = entry.findtext("SeriesID")
        if path is not None and series is not None:
            entries[(path, str(series))] = entry
    return entries


def stale_selection_report(cell_regions_xml, annotator_xmls, phases=PHASES, phase_signature=None):
    """Return ``[(xml_path, track_key), ...]`` for missing/stale selections."""
    revisions = raw_track_revisions(cell_regions_xml)
    stale = []
    for xml_path in annotator_xmls:
        entries = selection_entries_by_track(xml_path)
        for key, revision in revisions.items():
            if not entry_is_current(entries.get(key), revision, phases, phase_signature=phase_signature):
                stale.append((xml_path, key))
    return stale


def format_stale_selection_report(stale):
    """Format stale selection records into an actionable error message."""
    grouped = {}
    for xml_path, (anchor, series) in stale:
        grouped.setdefault(xml_path, []).append(f"series {series} ({os.path.basename(anchor)})")
    lines = ["Phase selections are incomplete or stale:"]
    for xml_path, tracks in grouped.items():
        lines.append(f"- {os.path.basename(xml_path)}: {', '.join(tracks)}")
    return "\n".join(lines)
