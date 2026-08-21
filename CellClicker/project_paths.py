"""Resolve canonical paths in a CellClicker project directory."""

from dataclasses import dataclass
from pathlib import Path


CELL_REGIONS_FILENAME = "cell_regions.xml"
LEGACY_CELL_REGIONS_FILENAME = "cell_reigons.xml"


@dataclass(frozen=True)
class CellRegionsXMLResolution:
    """Canonical raw-annotation XML path and any compatibility action taken."""

    path: Path
    migrated_legacy_file: bool = False
    both_files_present: bool = False


def resolve_cell_regions_xml(project_dir):
    """Return a project's canonical raw-annotation XML path.

    A sole legacy ``cell_reigons.xml`` is renamed to ``cell_regions.xml``.
    When both files exist, the canonical file remains authoritative and neither
    file is changed; callers can surface ``both_files_present`` to the user.
    """
    images_dir = Path(project_dir).expanduser() / "images"
    canonical_path = images_dir / CELL_REGIONS_FILENAME
    legacy_path = images_dir / LEGACY_CELL_REGIONS_FILENAME
    canonical_exists = canonical_path.is_file()
    legacy_exists = legacy_path.is_file()

    if canonical_exists and legacy_exists:
        return CellRegionsXMLResolution(canonical_path, both_files_present=True)
    if legacy_exists:
        legacy_path.replace(canonical_path)
        return CellRegionsXMLResolution(canonical_path, migrated_legacy_file=True)
    return CellRegionsXMLResolution(canonical_path)
