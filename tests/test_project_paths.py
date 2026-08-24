import xml.etree.ElementTree as ET

from CellClicker.manageXML import append_cell_regions_xml
from CellClicker.project_paths import (
    CELL_REGIONS_FILENAME,
    LEGACY_CELL_REGIONS_FILENAME,
    migrate_legacy_cell_regions_xml,
    resolve_cell_regions_xml,
)


def test_legacy_raw_xml_is_resolved_without_modifying_the_project(tmp_path):
    images = tmp_path / "project" / "images"
    images.mkdir(parents=True)
    legacy_path = images / LEGACY_CELL_REGIONS_FILENAME
    legacy_path.write_text("<annotations><path /></annotations>", encoding="utf-8")

    resolution = resolve_cell_regions_xml(images.parent)

    assert resolution.path == legacy_path
    assert resolution.using_legacy_file
    assert not resolution.migrated_legacy_file
    assert not resolution.both_files_present
    assert resolution.path.read_text(encoding="utf-8") == "<annotations><path /></annotations>"
    assert legacy_path.exists()
    assert not (images / CELL_REGIONS_FILENAME).exists()


def test_explicit_legacy_raw_xml_migration_renames_to_the_canonical_filename(tmp_path):
    images = tmp_path / "project" / "images"
    images.mkdir(parents=True)
    legacy_path = images / LEGACY_CELL_REGIONS_FILENAME
    legacy_path.write_text("<annotations><path /></annotations>", encoding="utf-8")

    resolution = migrate_legacy_cell_regions_xml(images.parent)

    assert resolution.path == images / CELL_REGIONS_FILENAME
    assert not resolution.using_legacy_file
    assert resolution.migrated_legacy_file
    assert not resolution.both_files_present
    assert resolution.path.read_text(encoding="utf-8") == "<annotations><path /></annotations>"
    assert not legacy_path.exists()


def test_canonical_raw_xml_is_left_unchanged(tmp_path):
    images = tmp_path / "project" / "images"
    images.mkdir(parents=True)
    canonical_path = images / CELL_REGIONS_FILENAME
    canonical_path.write_text("<annotations />", encoding="utf-8")

    resolution = resolve_cell_regions_xml(images.parent)

    assert resolution.path == canonical_path
    assert not resolution.using_legacy_file
    assert not resolution.migrated_legacy_file
    assert not resolution.both_files_present
    assert canonical_path.read_text(encoding="utf-8") == "<annotations />"


def test_duplicate_raw_xml_files_prefer_canonical_without_modifying_legacy(tmp_path):
    images = tmp_path / "project" / "images"
    images.mkdir(parents=True)
    canonical_path = images / CELL_REGIONS_FILENAME
    legacy_path = images / LEGACY_CELL_REGIONS_FILENAME
    canonical_path.write_text("<annotations canonical=\"true\" />", encoding="utf-8")
    legacy_path.write_text("<annotations legacy=\"true\" />", encoding="utf-8")

    resolution = resolve_cell_regions_xml(images.parent)

    assert resolution.path == canonical_path
    assert not resolution.using_legacy_file
    assert resolution.both_files_present
    assert not resolution.migrated_legacy_file
    assert legacy_path.read_text(encoding="utf-8") == "<annotations legacy=\"true\" />"


def test_explicit_migration_leaves_duplicate_raw_xml_files_unchanged(tmp_path):
    images = tmp_path / "project" / "images"
    images.mkdir(parents=True)
    canonical_path = images / CELL_REGIONS_FILENAME
    legacy_path = images / LEGACY_CELL_REGIONS_FILENAME
    canonical_path.write_text("<annotations canonical=\"true\" />", encoding="utf-8")
    legacy_path.write_text("<annotations legacy=\"true\" />", encoding="utf-8")

    resolution = migrate_legacy_cell_regions_xml(images.parent)

    assert resolution.path == canonical_path
    assert resolution.both_files_present
    assert not resolution.migrated_legacy_file
    assert legacy_path.read_text(encoding="utf-8") == "<annotations legacy=\"true\" />"


def test_first_annotation_uses_the_canonical_filename(tmp_path):
    project = tmp_path / "project"
    (project / "images").mkdir(parents=True)
    xml_path = resolve_cell_regions_xml(project).path

    append_cell_regions_xml(
        str(xml_path), "frame_001.png", 0, 0.5, 0.5, 10, 10, 100, 100, 1
    )

    assert xml_path.is_file()
    assert not (project / "images" / LEGACY_CELL_REGIONS_FILENAME).exists()
    assert ET.parse(xml_path).getroot().tag == "annotations"
