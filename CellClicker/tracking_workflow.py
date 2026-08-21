"""Convenience entry points for building and optionally reviewing tracking XML."""

import os
from tkinter import filedialog

from .build_tracking_xml import build_tracking_xml
from .tracking_review_ui import launch_tracking_review_ui
from .project_paths import resolve_cell_regions_xml

def build_tracking_xml_from_dataset(
    dataset_dir,
    phase_xml=None,
    output_xml=None,
    include_otsu=False,
    launch_ui=False,
    phases=None,
):
    """Build tracking XML from the standard CellClicker project layout.

    ``dataset_dir`` must contain ``images/cell_regions.xml`` and phase
    selections under ``user_selections``. Returns the output XML path and can
    optionally open the review interface.
    """
    dataset_dir = os.path.normpath(dataset_dir)
    selections_dir = os.path.join(dataset_dir, "user_selections")
    cell_regions_xml = str(resolve_cell_regions_xml(dataset_dir).path)

    if phase_xml is None:
        phase_xml = os.path.join(selections_dir, "aggregated_tracking.xml")

    if output_xml is None:
        output_xml = os.path.join(selections_dir, "tracking_review.xml")

    build_tracking_xml(
        phase_xml=phase_xml,
        cell_regions_xml=cell_regions_xml,
        output_xml=output_xml,
        dataset_root=dataset_dir,
        include_otsu=include_otsu,
        phases=phases or None,
    )
    if launch_ui:
        launch_tracking_review_ui(output_xml)

    return output_xml


def choose_dataset_and_build_tracking_xml(
    phase_xml=None,
    output_xml=None,
    include_otsu=False,
    launch_ui=True,
):
    """Ask for a project directory, build tracking XML, and return its path.

    Returns ``None`` when the file dialog is cancelled.
    """
    dataset_dir = filedialog.askdirectory(title="Select Dataset Directory")
    if not dataset_dir:
        return None

    return build_tracking_xml_from_dataset(
        dataset_dir=dataset_dir,
        phase_xml=phase_xml,
        output_xml=output_xml,
        include_otsu=include_otsu,
        launch_ui=launch_ui,
    )
