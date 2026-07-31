"""SlideBook Python hierarchical-capture bridge for raw montage images.

SlideBook supplies a raw montage as ``(position, height, width)``.  CelFDrive's
public prediction API uses ``(height, width, position)``, so this module is the
small adapter installed alongside ``CelFDrive.sbs``.
"""

import numpy as np

import predict


def _to_celfdrive_montage(raw_image):
    """Return a SlideBook raw montage in CelFDrive's image-axis convention."""
    image = np.asarray(raw_image)
    if image.ndim != 3:
        raise ValueError(
            "SlideBook raw montage must have shape (position, height, width); "
            f"received shape {image.shape}"
        )
    return np.moveaxis(image, 0, -1)


def _apply_objective_offset(targets):
    """Apply the configured high-resolution-objective stage calibration."""
    count, target_x, target_y, target_z, scripts, names, comments = targets
    offset = predict.get_config()["slidebook"]["objective_offset_um"]
    return (
        count,
        np.asarray(target_x, dtype=float) + float(offset["x"]),
        np.asarray(target_y, dtype=float) + float(offset["y"]),
        np.asarray(target_z, dtype=float) + float(offset["z"]),
        scripts,
        names,
        comments,
    )


def find_locations_of_interest_montage(
    image,
    stage_x,
    stage_y,
    stage_z,
    xy_pixel_spacing_um,
    z_spacing_um,
    x_stage_direction,
    y_stage_direction,
    z_stage_direction,
    dims,
    channels,
    name,
    comments,
):
    """Return CelFDrive targets for a raw SlideBook montage callback.

    ``z_spacing_um``, ``z_stage_direction``, ``dims``, ``channels``, ``name``,
    and ``comments`` are supplied by SlideBook but are not used by the current
    two-dimensional detection and coordinate conversion workflow.
    """
    del z_spacing_um, z_stage_direction, dims, channels, name, comments
    targets = predict.get_target_locations(
        stage_x=stage_x,
        stage_y=stage_y,
        stage_z=stage_z,
        image=_to_celfdrive_montage(image),
        xy_pixel_spacing_um=xy_pixel_spacing_um,
        x_stage_direction=x_stage_direction,
        y_stage_direction=y_stage_direction,
    )
    return _apply_objective_offset(targets)
