"""Import metadata-labelled TIFF time series into CellClicker projects."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import tifffile
from PIL import Image

from benchmarks.core import preprocess_image
from .project_paths import CELL_REGIONS_FILENAME


AUTO_AXIS = "auto"
Z_REDUCTIONS = frozenset({"max", "first"})


def find_tiff_files(source_directory):
    """Return TIFF files directly contained in ``source_directory`` in name order."""
    source_directory = Path(source_directory)
    if not source_directory.is_dir():
        raise NotADirectoryError(f"TIFF source directory does not exist: {source_directory}")
    files = sorted(path for path in source_directory.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"})
    if not files:
        raise FileNotFoundError(f"No .tif or .tiff files found in {source_directory}")
    return files


def _read_series_metadata(path):
    """Read the first TIFF series' axes and shape with actionable validation."""
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("contains no image series")
            series = tif.series[0]
            axes, shape = series.axes, series.shape
    except (OSError, tifffile.TiffFileError) as exc:
        raise ValueError(f"Could not read TIFF `{path}`: {exc}") from exc
    if len(axes) != len(shape) or len(shape) < 2:
        raise ValueError(f"TIFF `{path}` must contain at least two spatial dimensions; found axes `{axes}`.")
    return axes, shape


def _spatial_axis_indices(axes):
    """Return Y/X axes, falling back to the conventional trailing two dimensions."""
    if axes.count("Y") == 1 and axes.count("X") == 1:
        return axes.index("Y"), axes.index("X")
    return len(axes) - 2, len(axes) - 1


def _suggest_axis_mapping(axes):
    """Suggest time and channel dimensions from metadata or array order."""
    y_axis, x_axis = _spatial_axis_indices(axes)
    non_spatial = [index for index in range(len(axes)) if index not in {y_axis, x_axis}]
    time_axis = axes.index("T") if "T" in axes else (non_spatial[0] if non_spatial else None)
    channel_axis = axes.index("C") if "C" in axes and axes.index("C") != time_axis else None
    return time_axis, channel_axis


def describe_tiff_axes(source_directory):
    """Describe the first TIFF's dimensions and metadata-derived import defaults.

    TIFF readers sometimes infer generic dimension labels when metadata is absent.
    The returned indices remain usable in that case, allowing callers to map
    dimensions by their displayed position and size.
    """
    path = find_tiff_files(source_directory)[0]
    axes, shape = _read_series_metadata(path)
    time_axis, channel_axis = _suggest_axis_mapping(axes)
    return {
        "path": str(path), "axes": axes, "shape": tuple(shape),
        "time_axis": time_axis, "channel_axis": channel_axis,
        "spatial_axes": _spatial_axis_indices(axes),
    }


def _resolve_axis_mapping(path, time_axis, channel_axis, z_reduction):
    """Validate one TIFF's axis-role mapping and return resolved indices."""
    axes, shape = _read_series_metadata(path)
    y_axis, x_axis = _spatial_axis_indices(axes)
    suggested_time, suggested_channel = _suggest_axis_mapping(axes)
    time_axis = suggested_time if time_axis == AUTO_AXIS else time_axis
    channel_axis = suggested_channel if channel_axis == AUTO_AXIS else channel_axis
    for role, index in (("Time", time_axis), ("Channel", channel_axis)):
        if index is not None and (not isinstance(index, int) or not 0 <= index < len(axes)):
            raise ValueError(f"{role} axis {index!r} is not available in TIFF `{path}` with axes `{axes}`.")
        if index in {y_axis, x_axis}:
            raise ValueError(f"{role} axis cannot use spatial axis {index} in TIFF `{path}`.")
    if time_axis is not None and time_axis == channel_axis:
        raise ValueError(f"Time and channel axes both use dimension {time_axis} in TIFF `{path}`.")
    if z_reduction not in Z_REDUCTIONS:
        raise ValueError(f"Z reduction must be one of {sorted(Z_REDUCTIONS)}; got `{z_reduction}`.")
    assigned = {y_axis, x_axis, time_axis, channel_axis}
    z_axis = axes.index("Z") if axes.count("Z") == 1 else None
    for index, length in enumerate(shape):
        if index not in assigned and index != z_axis and length != 1:
            raise ValueError(
                f"TIFF `{path}` has unassigned dimension {index} ({axes[index]}, size {length}). "
                "Choose it as the time or channel axis, or use a TIFF with a singleton dimension there."
            )
    return {
        "axes": axes, "shape": shape, "time_axis": time_axis,
        "channel_axis": channel_axis, "y_axis": y_axis, "x_axis": x_axis,
        "z_axis": z_axis,
    }


def available_channel_indices(source_directory, channel_axis=AUTO_AXIS):
    """Return channel indices available in every TIFF for the selected axis."""
    counts = []
    for path in find_tiff_files(source_directory):
        axes, shape = _read_series_metadata(path)
        _, suggested_channel = _suggest_axis_mapping(axes)
        selected_axis = suggested_channel if channel_axis == AUTO_AXIS else channel_axis
        if selected_axis is None:
            counts.append(1)
        elif not isinstance(selected_axis, int) or not 0 <= selected_axis < len(axes):
            raise ValueError(f"Channel axis {selected_axis!r} is not available in TIFF `{path}` with axes `{axes}`.")
        else:
            counts.append(shape[selected_axis])
    return list(range(min(counts)))


def _timepoints_from_tiff(path, channel_index, time_axis, channel_axis, z_reduction):
    """Yield 2-D frames after reducing dimensions according to explicit roles."""
    mapping = _resolve_axis_mapping(path, time_axis, channel_axis, z_reduction)
    try:
        with tifffile.TiffFile(path) as tif:
            image = tif.series[0].asarray()
    except (OSError, tifffile.TiffFileError) as exc:
        raise ValueError(f"Could not read TIFF pixels from `{path}`: {exc}") from exc
    image = np.asarray(image)
    if mapping["channel_axis"] is None and channel_index != 0:
        raise ValueError(f"TIFF `{path}` has no selected channel axis; only channel 0 can be selected.")
    active_axes = list(range(image.ndim))
    kept_axes = {mapping["time_axis"], mapping["y_axis"], mapping["x_axis"]}
    for original_axis in reversed(range(image.ndim)):
        if original_axis in kept_axes:
            continue
        current_axis = active_axes.index(original_axis)
        if original_axis == mapping["channel_axis"]:
            if not 0 <= channel_index < image.shape[current_axis]:
                raise ValueError(
                    f"Channel {channel_index} is unavailable in TIFF `{path}` "
                    f"(dimension {original_axis} has {image.shape[current_axis]} channels)."
                )
            image = np.take(image, channel_index, axis=current_axis)
        elif original_axis == mapping["z_axis"] and z_reduction == "max":
            image = np.max(image, axis=current_axis)
        else:
            image = np.take(image, 0, axis=current_axis)
        active_axes.pop(current_axis)
    output_axes = [mapping["y_axis"], mapping["x_axis"]]
    if mapping["time_axis"] is not None:
        output_axes.insert(0, mapping["time_axis"])
    image = np.transpose(image, [active_axes.index(axis) for axis in output_axes])
    if mapping["time_axis"] is None:
        image = image[np.newaxis, ...]
    yield from image


def _timepoint_count(path, time_axis, channel_axis, z_reduction):
    """Return the number of timepoints a TIFF contributes without reading pixels."""
    mapping = _resolve_axis_mapping(path, time_axis, channel_axis, z_reduction)
    return mapping["shape"][mapping["time_axis"]] if mapping["time_axis"] is not None else 1


def series_name_for_tiff(path, used_names):
    """Create a deterministic, filesystem-safe, unique series name for a TIFF.

    The name is used as the filename prefix of every frame the TIFF produces,
    so it preserves the source stem and is disambiguated when two stems
    sanitize identically.
    """
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).stem).strip("._") or "series"
    name, suffix = base, 2
    while name.casefold() in used_names:
        name, suffix = f"{base}_{suffix}", suffix + 1
    used_names.add(name.casefold())
    return name


def assign_series_names(tiff_paths):
    """Return ``{tiff path: series name}`` for one import, resolving collisions."""
    used_names = set()
    return {path: series_name_for_tiff(path, used_names) for path in tiff_paths}


def frame_filename(series_name, timepoint, frame_count=0):
    """Return the flat project filename for one timepoint of one series.

    Frame numbers are zero padded to at least three digits so that they sort
    chronologically as text and match CellClicker's ``t<frame>`` convention.
    """
    return f"{series_name}_t{timepoint:0{max(3, len(str(frame_count)))}d}.png"


def _write_project(tiff_paths, project_directory, channel_index, time_axis, channel_axis, z_reduction, series_names, progress_callback=None):
    """Write a complete flat project into an already-created temporary directory."""
    images_directory = project_directory / "images"
    images_directory.mkdir()
    ElementTree.ElementTree(ElementTree.Element("annotations")).write(images_directory / CELL_REGIONS_FILENAME, encoding="utf-8", xml_declaration=True)
    frame_counts = {
        path: _timepoint_count(path, time_axis, channel_axis, z_reduction)
        for path in tiff_paths
    }
    total_frames = sum(frame_counts.values())
    completed = 0
    for path in tiff_paths:
        series_name, frame_count = series_names[path], frame_counts[path]
        for timepoint, frame in enumerate(
            _timepoints_from_tiff(path, channel_index, time_axis, channel_axis, z_reduction), start=1
        ):
            Image.fromarray(preprocess_image(frame)).save(images_directory / frame_filename(series_name, timepoint, frame_count))
            completed += 1
            if progress_callback:
                progress_callback(completed, total_frames, str(path))
    return {"series": len(tiff_paths), "frames": completed, "project_directory": str(project_directory)}


def create_projects_from_tiff_folder(
    source_directory,
    output_directory,
    channel_index=0,
    separate_projects=False,
    progress_callback=None,
    time_axis=AUTO_AXIS,
    channel_axis=AUTO_AXIS,
    z_reduction="max",
):
    """Create atomic CellClicker project(s) from every TIFF in one directory.

    Every generated PNG is written directly into the project's flat ``images/``
    directory as ``<series>_t<frame>.png``; series identity lives in the
    filename rather than in a directory hierarchy. ``time_axis`` and
    ``channel_axis`` are dimension indices, ``None``, or ``"auto"``; automatic
    mode uses TIFF metadata and falls back to the first non-spatial dimension
    for time. ``z_reduction`` is ``"max"`` or ``"first"`` when Z is not assigned
    as time or channel.
    """
    tiff_paths = find_tiff_files(source_directory)
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_directory}")
    if channel_index not in available_channel_indices(source_directory, channel_axis):
        raise ValueError(f"Channel {channel_index} is not available in every TIFF in {source_directory}.")
    series_names = assign_series_names(tiff_paths)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="celfdrive-tiff-import-", dir=output_directory.parent))
    try:
        if separate_projects:
            summaries = []
            for path in tiff_paths:
                project = staging / series_names[path]
                project.mkdir()
                summaries.append(
                    _write_project(
                        [path], project, channel_index, time_axis, channel_axis,
                        z_reduction, series_names, progress_callback,
                    )
                )
            result = {"projects": summaries, "series": len(tiff_paths), "frames": sum(item["frames"] for item in summaries)}
        else:
            result = _write_project(
                tiff_paths, staging, channel_index, time_axis, channel_axis,
                z_reduction, series_names, progress_callback,
            )
            result["projects"] = [result.copy()]
        os.replace(staging, output_directory)
        for project in result["projects"]:
            project["project_directory"] = project["project_directory"].replace(str(staging), str(output_directory), 1)
        result["output_directory"] = str(output_directory)
        if not separate_projects:
            result["project_directory"] = str(output_directory)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
