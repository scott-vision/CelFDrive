"""Resolve canonical paths in a CellClicker project directory."""

from dataclasses import dataclass
from pathlib import Path


CELL_REGIONS_FILENAME = "cell_regions.xml"
LEGACY_CELL_REGIONS_FILENAME = "cell_reigons.xml"


@dataclass(frozen=True)
class CellRegionsXMLResolution:
    """Resolved raw-annotation XML path and compatibility state."""

    path: Path
    using_legacy_file: bool = False
    migrated_legacy_file: bool = False
    both_files_present: bool = False


def resolve_cell_regions_xml(project_dir):
    """Return the raw-annotation XML path to use for a project.

    The canonical ``cell_regions.xml`` is preferred. A project containing only
    the legacy ``cell_reigons.xml`` remains usable without changing its files.
    Call :func:`migrate_legacy_cell_regions_xml` when an explicit, user-visible
    rename is wanted.
    """
    images_dir = Path(project_dir).expanduser() / "images"
    canonical_path = images_dir / CELL_REGIONS_FILENAME
    legacy_path = images_dir / LEGACY_CELL_REGIONS_FILENAME
    canonical_exists = canonical_path.is_file()
    legacy_exists = legacy_path.is_file()

    if canonical_exists and legacy_exists:
        return CellRegionsXMLResolution(canonical_path, both_files_present=True)
    if legacy_exists:
        return CellRegionsXMLResolution(legacy_path, using_legacy_file=True)
    return CellRegionsXMLResolution(canonical_path)


def migrate_legacy_cell_regions_xml(project_dir):
    """Explicitly rename a sole legacy raw-annotation XML file to the canonical name.

    This operation changes project files and is intentionally separate from
    :func:`resolve_cell_regions_xml`. When both names are already present, the
    canonical file remains authoritative and neither file is changed.
    """
    resolution = resolve_cell_regions_xml(project_dir)
    if not resolution.using_legacy_file:
        return resolution

    legacy_path = resolution.path
    canonical_path = legacy_path.with_name(CELL_REGIONS_FILENAME)
    if canonical_path.exists():
        return resolve_cell_regions_xml(project_dir)

    legacy_path.rename(canonical_path)
    return CellRegionsXMLResolution(canonical_path, migrated_legacy_file=True)
