"""Batch extraction, per-movie plots, aggregate plots, and CSV export.

For every base TIFF in INPUT_DIR this script:
  1. Loads the raw movie, cell mask, and spot mask.
  2. Extracts mean NDC80 and NUP intensity from spot ∩ cell at every frame.
  3. Saves the existing per-movie dual-axis PDF with bounded 4-parameter tanh fits.
  4. Writes every per-frame measurement from every movie to one long-format CSV.
  5. Interpolates each movie onto a common normalised-time grid (0-1).
  6. Writes aggregate mean/SD/SEM/n values to a second CSV.
  7. Saves an aggregate mean ± SEM PDF, optionally with individual traces and
     tanh fits to the aggregate means.

The raw CSV always contains unscaled intensity values. AGGREGATE_INTENSITY_MODE
only controls the aggregate statistics and aggregate plot.
"""

from __future__ import annotations

import csv
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import xml.etree.ElementTree as ET

# -----------------------------------------------------------------------------
# Non-interactive backend BEFORE pyplot import
# -----------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg", force=True)  # type: ignore
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402
from scipy.optimize import OptimizeWarning, curve_fit  # noqa: E402

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_DIR = Path(
    "../../Segmentation/NUP_data/Deconvolved and deskewed/"
).expanduser().resolve()
SEG_MASK_DIR = INPUT_DIR / "segmentation_outputs_batch"
SPOT_MASK_DIR = INPUT_DIR / "filtered_spotmask_outputs_batch"
PLOT_DIR = INPUT_DIR / "intensity_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_NDC80 = 0
CHANNEL_NUP = 1
TIME_INTERVAL = 8  # seconds per frame

COLOR_NDC80 = "tab:blue"
COLOR_NUP = "tab:red"

# Number of positions in the common 0-1 time grid used for aggregation.
COMMON_TIME_POINTS = 201

# "raw" keeps the original fluorescence units in the aggregate.
# "minmax" scales each movie independently to 0-1 before aggregation.
# The all_raw_intensities.csv output is raw in either case.
AGGREGATE_INTENSITY_MODE = "raw"  # choose "raw" or "minmax"

DRAW_INDIVIDUAL_TRACES_ON_AGGREGATE = True
FIT_AGGREGATE_MEANS = True

RAW_CSV_PATH = PLOT_DIR / "all_raw_intensities.csv"
AGGREGATE_CSV_PATH = PLOT_DIR / "aggregate_intensities.csv"
AGGREGATE_PDF_PATH = PLOT_DIR / "aggregate_intensity_plot.pdf"


@dataclass
class MovieResult:
    """All extracted per-frame values for one movie."""

    movie: str
    times_s: np.ndarray
    time_norm: np.ndarray
    ndc80: np.ndarray
    nup: np.ndarray
    roi_voxels: np.ndarray
    spot_counts: np.ndarray


# -----------------------------------------------------------------------------
# Locate and load data
# -----------------------------------------------------------------------------


def discover_movies(directory: Path) -> List[Path]:
    """Return sorted base movies, excluding derivative TIFF stacks."""
    excluded_suffixes = ("_segmented.tif", "_spotmasks.tif")
    return sorted(
        p
        for p in directory.glob("*.tif")
        if not p.name.endswith(excluded_suffixes)
    )


def load_triplet(
    movie_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load the raw movie and its matching cell and spot masks."""
    seg_path = SEG_MASK_DIR / f"{movie_path.stem}_segmented.tif"
    spot_path = SPOT_MASK_DIR / f"{movie_path.stem}_spotmasks.tif"

    missing = [path.name for path in (seg_path, spot_path) if not path.exists()]
    if missing:
        logging.warning(
            "%s - missing %s; skipping", movie_path.name, ", ".join(missing)
        )
        return None

    movie = tiff.imread(str(movie_path))
    seg_mask = tiff.imread(str(seg_path))
    spot_mask = tiff.imread(str(spot_path))
    return movie, seg_mask, spot_mask



def load_filtered_spot_counts(movie_stem: str, n_frames: int) -> np.ndarray:
    """
    Return the exact number of retained/used spots at each timepoint.

    Counts are read from <movie>_filtered_coordinates.xml produced by the
    filtered spot-detection script. This is preferable to connected-component
    counting on the binary TIFF because neighbouring spot spheres can overlap
    and merge into one connected component.
    """
    xml_path = SPOT_MASK_DIR / f"{movie_stem}_filtered_coordinates.xml"

    counts = np.full(n_frames, -1, dtype=np.int64)

    if not xml_path.exists():
        logging.warning(
            "%s - filtered coordinate XML not found; n_spots_used will be -1",
            xml_path.name,
        )
        return counts

    root = ET.parse(str(xml_path)).getroot()

    for frame_elem in root.findall("frame"):
        try:
            frame_idx = int(frame_elem.get("number", "-1"))
        except ValueError:
            continue

        if 0 <= frame_idx < n_frames:
            counts[frame_idx] = len(frame_elem.findall("spot"))

    missing = int(np.count_nonzero(counts < 0))
    if missing:
        logging.warning(
            "%s - %d frame(s) missing from filtered coordinate XML",
            movie_stem,
            missing,
        )

    return counts


# -----------------------------------------------------------------------------
# Intensity extraction
# -----------------------------------------------------------------------------


def normalise_time(times_s: np.ndarray) -> np.ndarray:
    """Map an acquisition's full time range to 0-1."""
    times_s = np.asarray(times_s, dtype=float)
    if times_s.size < 2:
        return np.zeros_like(times_s, dtype=float)

    duration = float(times_s[-1] - times_s[0])
    if duration <= 0:
        return np.zeros_like(times_s, dtype=float)

    return (times_s - times_s[0]) / duration


def extract_means(
    movie: np.ndarray,
    seg: np.ndarray,
    spot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return time, NDC80, NUP, and ROI voxel-count arrays."""
    if movie.ndim != 5:
        raise ValueError(f"Expected 5-D movie (t,z,c,y,x); got {movie.shape}")

    t_frames, z_slices, n_channels, height, width = movie.shape
    if max(CHANNEL_NDC80, CHANNEL_NUP) >= n_channels:
        raise IndexError(
            f"Requested channels {CHANNEL_NDC80}/{CHANNEL_NUP}, "
            f"but movie has {n_channels} channel(s)"
        )

    expected_mask_shape = (t_frames, z_slices, height, width)
    if seg.shape != expected_mask_shape:
        raise ValueError(
            f"Cell mask shape {seg.shape} does not match expected "
            f"{expected_mask_shape}"
        )
    if spot.shape != expected_mask_shape:
        raise ValueError(
            f"Spot mask shape {spot.shape} does not match expected "
            f"{expected_mask_shape}"
        )

    roi_mask = (seg > 0) & (spot > 0)

    ndc80_vals = np.full(t_frames, np.nan, dtype=float)
    nup_vals = np.full(t_frames, np.nan, dtype=float)
    roi_voxels = np.zeros(t_frames, dtype=np.int64)

    for frame in range(t_frames):
        frame_mask = roi_mask[frame]
        count = int(np.count_nonzero(frame_mask))
        roi_voxels[frame] = count

        if count == 0:
            continue

        ndc80_vals[frame] = float(
            np.mean(movie[frame, :, CHANNEL_NDC80][frame_mask])
        )
        nup_vals[frame] = float(
            np.mean(movie[frame, :, CHANNEL_NUP][frame_mask])
        )

    times_s = np.arange(t_frames, dtype=float) * TIME_INTERVAL
    return times_s, ndc80_vals, nup_vals, roi_voxels


# -----------------------------------------------------------------------------
# Four-parameter tanh fitting
# -----------------------------------------------------------------------------


def tanh_func(
    x: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> np.ndarray:
    """Model: d + a * tanh(b * (x - c))."""
    return d + a * np.tanh(b * (x - c))


def fit_tanh_bounded(
    times: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit on normalised time and return x in original units, y-fit, parameters.

    The best successful fit across all initial seeds is retained rather than the
    first fit that happens to converge.
    """
    mask = np.isfinite(times) & np.isfinite(values)
    x_raw = np.asarray(times[mask], dtype=float)
    y_raw = np.asarray(values[mask], dtype=float)

    if x_raw.size < 4:
        raise RuntimeError("Too few finite data points for fitting")

    x_span = float(np.ptp(x_raw))
    if x_span > 0:
        x_norm = (x_raw - x_raw.min()) / x_span
    else:
        x_norm = np.zeros_like(x_raw)

    y_min = float(np.min(y_raw))
    y_max = float(np.max(y_raw))
    y_span = y_max - y_min

    if y_span <= 0:
        raise RuntimeError("No intensity variation to fit")

    amp_guess = y_span / 2.0
    d_guess = y_min

    slope_seeds = (0.5, 1.0, 2.0, 5.0, 10.0)
    centre_seeds = (0.20, 0.35, 0.50, 0.65, 0.80)

    bounds_lower = (0.0, 0.0, 0.0, y_min)
    bounds_upper = (np.inf, np.inf, 1.0, y_max)

    best_popt: np.ndarray | None = None
    best_sse = np.inf

    for b0 in slope_seeds:
        for c0 in centre_seeds:
            p0 = (amp_guess, b0, c0, d_guess)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=OptimizeWarning)
                    popt, _ = curve_fit(
                        tanh_func,
                        x_norm,
                        y_raw,
                        p0=p0,
                        bounds=(bounds_lower, bounds_upper),
                        maxfev=50000,
                    )

                residuals = y_raw - tanh_func(x_norm, *popt)
                sse = float(np.sum(residuals**2))
                if np.isfinite(sse) and sse < best_sse:
                    best_sse = sse
                    best_popt = popt
            except (RuntimeError, ValueError, FloatingPointError):
                continue

    if best_popt is None:
        raise RuntimeError("tanh fit failed for all initial guesses")

    x_fine_norm = np.linspace(0.0, 1.0, 300)
    y_fit = tanh_func(x_fine_norm, *best_popt)
    x_fine = x_raw.min() + x_fine_norm * x_span
    return x_fine, y_fit, best_popt


# -----------------------------------------------------------------------------
# Per-movie plotting
# -----------------------------------------------------------------------------


def plot_dual_axis(
    times: np.ndarray,
    ndc80: np.ndarray,
    nup: np.ndarray,
    out_path: Path,
) -> None:
    """Save the existing per-movie dual-axis raw-data and fit plot."""
    fig, ax_left = plt.subplots(figsize=(8, 6))

    ax_left.plot(times, ndc80, "x", color=COLOR_NDC80, label="NDC80")
    ax_left.set_xlabel("Time (s)")
    ax_left.set_ylabel("NDC80 Mean Intensity", color=COLOR_NDC80)
    ax_left.tick_params(axis="y", labelcolor=COLOR_NDC80)

    ax_right = ax_left.twinx()
    ax_right.plot(times, nup, "o", color=COLOR_NUP, label="NUP")
    ax_right.set_ylabel("NUP Mean Intensity", color=COLOR_NUP)
    ax_right.tick_params(axis="y", labelcolor=COLOR_NUP)

    try:
        x_fit, y_fit, _ = fit_tanh_bounded(times, ndc80)
        ax_left.plot(
            x_fit,
            y_fit,
            "--",
            color=COLOR_NDC80,
            label="NDC80 fit",
        )
    except RuntimeError as err:
        logging.warning("NDC80 fit failed: %s", err)

    try:
        x_fit, y_fit, _ = fit_tanh_bounded(times, nup)
        ax_right.plot(
            x_fit,
            y_fit,
            "--",
            color=COLOR_NUP,
            label="NUP fit",
        )
    except RuntimeError as err:
        logging.warning("NUP fit failed: %s", err)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        ncol=2,
    )

    ax_left.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CSV export
# -----------------------------------------------------------------------------


def csv_number(value: float) -> str | float:
    """Write missing numerical values as blank CSV fields."""
    return "" if not np.isfinite(value) else float(value)


def write_raw_csv(results: Sequence[MovieResult], out_path: Path) -> None:
    """Write every movie and frame to one long-format CSV."""
    fieldnames = (
        "movie",
        "frame",
        "time_s",
        "time_norm",
        "roi_voxels",
        "n_spots_used",
        "ndc80_mean_raw",
        "nup_mean_raw",
    )

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            for frame in range(result.times_s.size):
                writer.writerow(
                    {
                        "movie": result.movie,
                        "frame": frame,
                        "time_s": float(result.times_s[frame]),
                        "time_norm": float(result.time_norm[frame]),
                        "roi_voxels": int(result.roi_voxels[frame]),
                        "n_spots_used": int(result.spot_counts[frame]),
                        "ndc80_mean_raw": csv_number(result.ndc80[frame]),
                        "nup_mean_raw": csv_number(result.nup[frame]),
                    }
                )


def write_aggregate_csv(
    common_time: np.ndarray,
    ndc80_stats: dict[str, np.ndarray],
    nup_stats: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Write common-grid aggregate mean, SD, SEM, and sample size."""
    fieldnames = (
        "time_norm",
        "ndc80_mean",
        "ndc80_sd",
        "ndc80_sem",
        "ndc80_n",
        "nup_mean",
        "nup_sd",
        "nup_sem",
        "nup_n",
        "aggregate_intensity_mode",
    )

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for idx, time_norm in enumerate(common_time):
            writer.writerow(
                {
                    "time_norm": float(time_norm),
                    "ndc80_mean": csv_number(ndc80_stats["mean"][idx]),
                    "ndc80_sd": csv_number(ndc80_stats["sd"][idx]),
                    "ndc80_sem": csv_number(ndc80_stats["sem"][idx]),
                    "ndc80_n": int(ndc80_stats["n"][idx]),
                    "nup_mean": csv_number(nup_stats["mean"][idx]),
                    "nup_sd": csv_number(nup_stats["sd"][idx]),
                    "nup_sem": csv_number(nup_stats["sem"][idx]),
                    "nup_n": int(nup_stats["n"][idx]),
                    "aggregate_intensity_mode": AGGREGATE_INTENSITY_MODE,
                }
            )


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def prepare_intensity_for_aggregate(values: np.ndarray) -> np.ndarray:
    """Apply the configured per-movie scaling for aggregate calculations."""
    values = np.asarray(values, dtype=float).copy()

    if AGGREGATE_INTENSITY_MODE == "raw":
        return values

    if AGGREGATE_INTENSITY_MODE == "minmax":
        finite = np.isfinite(values)
        if not np.any(finite):
            return values

        low = float(np.min(values[finite]))
        high = float(np.max(values[finite]))
        span = high - low
        if span <= 0:
            values[finite] = 0.0
        else:
            values[finite] = (values[finite] - low) / span
        return values

    raise ValueError(
        "AGGREGATE_INTENSITY_MODE must be either 'raw' or 'minmax'; "
        f"got {AGGREGATE_INTENSITY_MODE!r}"
    )


def interpolate_trace(
    time_norm: np.ndarray,
    values: np.ndarray,
    common_time: np.ndarray,
) -> np.ndarray:
    """Interpolate one finite trace onto the common 0-1 time grid."""
    finite = np.isfinite(time_norm) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return np.full(common_time.shape, np.nan, dtype=float)

    x = np.asarray(time_norm[finite], dtype=float)
    y = np.asarray(values[finite], dtype=float)

    # Protect np.interp from duplicate x positions.
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    if unique_x.size < 2:
        return np.full(common_time.shape, np.nan, dtype=float)

    return np.interp(
        common_time,
        unique_x,
        unique_y,
        left=np.nan,
        right=np.nan,
    )


def calculate_stats(stack: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate column-wise mean, sample SD, SEM, and finite n."""
    finite = np.isfinite(stack)
    n = np.sum(finite, axis=0).astype(int)

    sums = np.nansum(stack, axis=0)
    mean = np.full(stack.shape[1], np.nan, dtype=float)
    np.divide(sums, n, out=mean, where=n > 0)

    centred = np.where(finite, stack - mean[np.newaxis, :], np.nan)
    sum_squares = np.nansum(centred**2, axis=0)

    sd = np.full_like(mean, np.nan)
    valid_sd = n > 1
    sd[valid_sd] = np.sqrt(sum_squares[valid_sd] / (n[valid_sd] - 1))

    sem = np.full_like(mean, np.nan)
    sem[valid_sd] = sd[valid_sd] / np.sqrt(n[valid_sd])

    return {"mean": mean, "sd": sd, "sem": sem, "n": n}


def aggregate_results(
    results: Sequence[MovieResult],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Return common time, channel stacks, and channel summary statistics."""
    common_time = np.linspace(0.0, 1.0, COMMON_TIME_POINTS)

    ndc80_rows = []
    nup_rows = []
    for result in results:
        ndc80_rows.append(
            interpolate_trace(
                result.time_norm,
                prepare_intensity_for_aggregate(result.ndc80),
                common_time,
            )
        )
        nup_rows.append(
            interpolate_trace(
                result.time_norm,
                prepare_intensity_for_aggregate(result.nup),
                common_time,
            )
        )

    ndc80_stack = np.vstack(ndc80_rows)
    nup_stack = np.vstack(nup_rows)

    return (
        common_time,
        ndc80_stack,
        nup_stack,
        calculate_stats(ndc80_stack),
        calculate_stats(nup_stack),
    )


# -----------------------------------------------------------------------------
# Aggregate plotting
# -----------------------------------------------------------------------------


def aggregate_axis_label(channel: str) -> str:
    if AGGREGATE_INTENSITY_MODE == "raw":
        return f"{channel} Mean Intensity"
    return f"{channel} Scaled Intensity (0-1 per movie)"


def plot_aggregate(
    common_time: np.ndarray,
    ndc80_stack: np.ndarray,
    nup_stack: np.ndarray,
    ndc80_stats: dict[str, np.ndarray],
    nup_stats: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Save aggregate mean ± SEM curves on normalised acquisition time."""
    fig, ax_left = plt.subplots(figsize=(8, 6))
    ax_right = ax_left.twinx()

    if DRAW_INDIVIDUAL_TRACES_ON_AGGREGATE:
        for trace in ndc80_stack:
            ax_left.plot(
                common_time,
                trace,
                color=COLOR_NDC80,
                alpha=0.15,
                linewidth=0.8,
            )
        for trace in nup_stack:
            ax_right.plot(
                common_time,
                trace,
                color=COLOR_NUP,
                alpha=0.15,
                linewidth=0.8,
            )

    ndc80_mean = ndc80_stats["mean"]
    ndc80_sem = ndc80_stats["sem"]
    nup_mean = nup_stats["mean"]
    nup_sem = nup_stats["sem"]

    ax_left.plot(
        common_time,
        ndc80_mean,
        color=COLOR_NDC80,
        linewidth=2.5,
        label="NDC80 mean",
    )
    ax_left.fill_between(
        common_time,
        ndc80_mean - ndc80_sem,
        ndc80_mean + ndc80_sem,
        color=COLOR_NDC80,
        alpha=0.20,
        label="NDC80 SEM",
    )

    ax_right.plot(
        common_time,
        nup_mean,
        color=COLOR_NUP,
        linewidth=2.5,
        label="NUP mean",
    )
    ax_right.fill_between(
        common_time,
        nup_mean - nup_sem,
        nup_mean + nup_sem,
        color=COLOR_NUP,
        alpha=0.20,
        label="NUP SEM",
    )

    if FIT_AGGREGATE_MEANS:
        try:
            x_fit, y_fit, _ = fit_tanh_bounded(common_time, ndc80_mean)
            ax_left.plot(
                x_fit,
                y_fit,
                "--",
                color=COLOR_NDC80,
                linewidth=2.0,
                label="NDC80 aggregate fit",
            )
        except RuntimeError as err:
            logging.warning("Aggregate NDC80 fit failed: %s", err)

        try:
            x_fit, y_fit, _ = fit_tanh_bounded(common_time, nup_mean)
            ax_right.plot(
                x_fit,
                y_fit,
                "--",
                color=COLOR_NUP,
                linewidth=2.0,
                label="NUP aggregate fit",
            )
        except RuntimeError as err:
            logging.warning("Aggregate NUP fit failed: %s", err)

    ax_left.set_xlabel("Normalised acquisition time (0-1)")
    ax_left.set_ylabel(aggregate_axis_label("NDC80"), color=COLOR_NDC80)
    ax_left.tick_params(axis="y", labelcolor=COLOR_NDC80)

    ax_right.set_ylabel(aggregate_axis_label("NUP"), color=COLOR_NUP)
    ax_right.tick_params(axis="y", labelcolor=COLOR_NUP)

    ax_left.set_xlim(0.0, 1.0)
    ax_left.grid(True)
    ax_left.set_title(
        f"Aggregate fluorescence kinetics (n={ndc80_stack.shape[0]} movies)"
    )

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        ncol=2,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    movies = discover_movies(INPUT_DIR)
    if not movies:
        logging.error("No base movies found in %s", INPUT_DIR)
        return 1

    logging.info("Found %d movie(s)", len(movies))
    results: list[MovieResult] = []

    for movie_path in movies:
        triplet = load_triplet(movie_path)
        if triplet is None:
            continue

        movie, seg_mask, spot_mask = triplet
        try:
            times_s, ndc80, nup, roi_voxels = extract_means(
                movie,
                seg_mask,
                spot_mask,
            )

            spot_counts = load_filtered_spot_counts(
                movie_path.stem,
                times_s.size,
            )

            result = MovieResult(
                movie=movie_path.stem,
                times_s=times_s,
                time_norm=normalise_time(times_s),
                ndc80=ndc80,
                nup=nup,
                roi_voxels=roi_voxels,
                spot_counts=spot_counts,
            )
            results.append(result)

            pdf_path = PLOT_DIR / f"{movie_path.stem}_intensity_plot.pdf"
            plot_dual_axis(times_s, ndc80, nup, pdf_path)
            logging.info("Saved %s", pdf_path.name)
        except Exception:
            logging.exception("Failed on %s", movie_path.name)

    if not results:
        logging.error("No movies were processed successfully")
        return 1

    write_raw_csv(results, RAW_CSV_PATH)
    logging.info("Saved raw measurements: %s", RAW_CSV_PATH.name)

    (
        common_time,
        ndc80_stack,
        nup_stack,
        ndc80_stats,
        nup_stats,
    ) = aggregate_results(results)

    write_aggregate_csv(
        common_time,
        ndc80_stats,
        nup_stats,
        AGGREGATE_CSV_PATH,
    )
    logging.info("Saved aggregate values: %s", AGGREGATE_CSV_PATH.name)

    plot_aggregate(
        common_time,
        ndc80_stack,
        nup_stack,
        ndc80_stats,
        nup_stats,
        AGGREGATE_PDF_PATH,
    )
    logging.info("Saved aggregate plot: %s", AGGREGATE_PDF_PATH.name)

    logging.info("All done - outputs stored in %s", PLOT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
