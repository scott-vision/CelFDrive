"""
Batch NDC80 spot detection, cell filtering, and top-92 selection
================================================================

For every 5-D movie stack (t,z,c,y,x):

1. Extract NDC80 fluorescence channel.
2. Gaussian smooth each 3-D timepoint.
3. Detect candidate local maxima.
4. Remove candidates whose centre is outside the segmented cell.
5. Rank remaining candidates by smoothed NDC80 peak intensity.
6. Retain at most MAX_SPOTS_PER_CELL brightest candidates.
7. Build the FINAL spot-mask TIFF only from those retained spots.
8. Save:
      <movie>_spotmasks.tif
      <movie>_coordinates.xml
      <movie>_filtered_coordinates.xml
      <movie>_spot_counts.csv
9. Save run parameters to options.yaml.

Important:
    MAX_SPOTS_PER_CELL = 92 is a CAP, not a target.

So if 54 spots are detected inside the cell, 54 are retained.
If 130 are detected inside the cell, only the brightest 92 are retained.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import numpy as np
from skimage.feature import peak_local_max
import tifffile as tiff
import yaml


# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path(
    "../../Segmentation/NUP_data/Deconvolved and deskewed/"
).expanduser().resolve()

SEGMENTATION_DIR = INPUT_DIR / "segmentation_outputs_batch"

OUTPUT_DIR = INPUT_DIR / "filtered_spotmask_outputs_batch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Channel containing NDC80
CHANNEL_INDEX = 0

# Number of timepoints processed simultaneously
BATCH_SIZE = 4

# Gaussian smoothing in (z, y, x)
SIGMA = (1, 1, 1)

# Candidate detection threshold
THRESH_PCNT = 99.7

# Minimum distance between detected local maxima
MIN_DISTANCE = 2

# Radius of sphere drawn around each retained peak
SPOT_RADIUS = 2

# Biological maximum
MAX_SPOTS_PER_CELL = 92

# Pixel size for ImageJ metadata
PIXEL_SIZE = 0.1625

# Time between frames
TIME_INTERVAL = 8


# =============================================================================
# Types
# =============================================================================

# Candidate:
# z, y, x, smoothed intensity
SpotCandidate = Tuple[int, int, int, float]


# =============================================================================
# Sphere creation
# =============================================================================

def create_sphere(
    center: Tuple[int, int, int],
    radius: int,
    shape: Tuple[int, int, int],
) -> cp.ndarray:
    """
    Return a boolean CuPy array containing a filled sphere.
    """

    z0, y0, x0 = center

    z, y, x = cp.indices(shape)

    dist_sq = (
        (z - z0) ** 2
        + (y - y0) ** 2
        + (x - x0) ** 2
    )

    return dist_sq <= radius ** 2


# =============================================================================
# Candidate detection
# =============================================================================

def detect_candidates_batch(
    images_3d: np.ndarray,
    *,
    sigma: Tuple[int, int, int] = SIGMA,
    threshold_percentile: float = THRESH_PCNT,
    min_distance: int = MIN_DISTANCE,
) -> Tuple[
    List[List[SpotCandidate]],
    List[float],
]:
    """
    Detect candidate NDC80 peaks in a batch of 3-D images.

    Returns
    -------
    candidates_per_frame
        For each timepoint:
            [(z, y, x, smoothed_peak_intensity), ...]

    thresholds
        Absolute intensity threshold used for each timepoint.
    """

    images_gpu = cp.asarray(images_3d)

    candidates_per_frame: List[List[SpotCandidate]] = []
    thresholds: List[float] = []

    for image_gpu in images_gpu:

        # -------------------------------------------------------------
        # Gaussian smoothing
        # -------------------------------------------------------------

        smoothed_gpu = gaussian_filter(
            image_gpu,
            sigma=sigma,
        )

        # -------------------------------------------------------------
        # Determine percentile threshold
        # -------------------------------------------------------------

        threshold = float(
            cp.percentile(
                smoothed_gpu,
                threshold_percentile,
            ).item()
        )

        thresholds.append(threshold)

        # peak_local_max operates on NumPy arrays
        smoothed_np = cp.asnumpy(smoothed_gpu)

        # -------------------------------------------------------------
        # Candidate local maxima
        # -------------------------------------------------------------

        coords = peak_local_max(
            smoothed_np,
            min_distance=min_distance,
            threshold_abs=threshold,
            exclude_border=False,
        )

        candidates: List[SpotCandidate] = []

        for z, y, x in coords:

            intensity = float(
                smoothed_np[z, y, x]
            )

            candidates.append(
                (
                    int(z),
                    int(y),
                    int(x),
                    intensity,
                )
            )

        candidates_per_frame.append(candidates)

    return candidates_per_frame, thresholds


# =============================================================================
# Cell filtering and ranking
# =============================================================================

def candidates_inside_cell(
    candidates: List[SpotCandidate],
    segmentation_mask: np.ndarray,
) -> List[SpotCandidate]:
    """
    Retain candidate peaks whose CENTRE lies inside the segmented cell.
    """

    inside: List[SpotCandidate] = []

    z_max, y_max, x_max = segmentation_mask.shape

    for z, y, x, intensity in candidates:

        # Defensive bounds check
        if not (
            0 <= z < z_max
            and 0 <= y < y_max
            and 0 <= x < x_max
        ):
            continue

        if segmentation_mask[z, y, x] > 0:

            inside.append(
                (
                    z,
                    y,
                    x,
                    intensity,
                )
            )

    return inside


def select_brightest_spots(
    candidates: List[SpotCandidate],
    max_spots: int = MAX_SPOTS_PER_CELL,
) -> List[SpotCandidate]:
    """
    Sort candidates by smoothed NDC80 intensity and retain at most max_spots.
    """

    sorted_candidates = sorted(
        candidates,
        key=lambda spot: spot[3],
        reverse=True,
    )

    return sorted_candidates[:max_spots]


# =============================================================================
# Final mask generation
# =============================================================================

def build_mask_from_spots(
    spots: List[SpotCandidate],
    shape: Tuple[int, int, int],
    *,
    spot_radius: int = SPOT_RADIUS,
) -> np.ndarray:
    """
    Construct a binary 3-D mask using ONLY the retained spots.
    """

    mask_gpu = cp.zeros(
        shape,
        dtype=cp.uint8,
    )

    for z, y, x, _intensity in spots:

        sphere = create_sphere(
            center=(z, y, x),
            radius=spot_radius,
            shape=shape,
        )

        mask_gpu[sphere] = 1

    return cp.asnumpy(mask_gpu)


# =============================================================================
# TIFF writing
# =============================================================================

def save_hyperstack_tiff(
    mask_stack: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save (t,z,y,x) uint8 mask as an ImageJ hyperstack.
    """

    tiff.imwrite(
        str(out_path),
        mask_stack,
        imagej=True,
        metadata={
            "axes": "TZYX",
        },
        resolution=(
            1 / PIXEL_SIZE,
            1 / PIXEL_SIZE,
        ),
        compression="zlib",
    )


# =============================================================================
# XML output
# =============================================================================

def save_coordinates_to_xml(
    coordinates: List[List[SpotCandidate]],
    out_path: Path,
) -> None:
    """
    Save spot coordinates and their smoothed NDC80 intensity to XML.
    """

    root = ET.Element("spots")

    for t, frame_coords in enumerate(coordinates):

        frame_elem = ET.SubElement(
            root,
            "frame",
            number=str(t),
        )

        for z, y, x, intensity in frame_coords:

            ET.SubElement(
                frame_elem,
                "spot",
                x=str(x),
                y=str(y),
                z=str(z),
                intensity=str(intensity),
            )

    tree = ET.ElementTree(root)

    tree.write(
        out_path,
        encoding="utf-8",
        xml_declaration=True,
    )


# =============================================================================
# CSV output
# =============================================================================

def save_counts_csv(
    rows: List[dict],
    out_path: Path,
) -> None:
    """
    Save candidate/filtered/retained spot counts for every timepoint.
    """

    fieldnames = [
        "movie",
        "frame",
        "time_s",
        "threshold",
        "n_candidates_total",
        "n_candidates_inside_cell",
        "n_spots_retained",
        "max_spots_allowed",
        "weakest_retained_intensity",
        "strongest_retained_intensity",
    ]

    with out_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Process one movie
# =============================================================================

def process_file(
    tif_path: Path,
) -> None:

    logging.info(
        "Processing %s",
        tif_path.name,
    )

    # -------------------------------------------------------------------------
    # Load movie
    # -------------------------------------------------------------------------

    movie = tiff.imread(
        str(tif_path)
    )

    if movie.ndim != 5:

        logging.error(
            "Expected 5-D stack (t,z,c,y,x); "
            "got %s - skipping",
            movie.shape,
        )

        return

    (
        t_frames,
        z_slices,
        n_channels,
        height,
        width,
    ) = movie.shape

    if CHANNEL_INDEX >= n_channels:

        logging.error(
            "Channel index %d out of range "
            "(movie has %d channel(s))",
            CHANNEL_INDEX,
            n_channels,
        )

        return

    # -------------------------------------------------------------------------
    # Load segmentation
    # -------------------------------------------------------------------------

    segmentation_path = (
        SEGMENTATION_DIR
        / f"{tif_path.stem}_segmented.tif"
    )

    if not segmentation_path.exists():

        logging.error(
            "Segmentation file %s not found - skipping",
            segmentation_path.name,
        )

        return

    segmentation = tiff.imread(
        str(segmentation_path)
    )

    expected_shape = (
        t_frames,
        z_slices,
        height,
        width,
    )

    if segmentation.shape != expected_shape:

        logging.error(
            "Segmentation shape %s does not match "
            "expected movie shape %s - skipping",
            segmentation.shape,
            expected_shape,
        )

        return

    # -------------------------------------------------------------------------
    # Allocate final output mask
    # -------------------------------------------------------------------------

    masks_out = np.zeros(
        expected_shape,
        dtype=np.uint8,
    )

    # -------------------------------------------------------------------------
    # Coordinate stores
    # -------------------------------------------------------------------------

    all_candidate_coordinates: List[
        List[SpotCandidate]
    ] = []

    all_inside_cell_coordinates: List[
        List[SpotCandidate]
    ] = []

    all_retained_coordinates: List[
        List[SpotCandidate]
    ] = []

    count_rows: List[dict] = []

    # -------------------------------------------------------------------------
    # Batch loop
    # -------------------------------------------------------------------------

    for t0 in range(
        0,
        t_frames,
        BATCH_SIZE,
    ):

        t1 = min(
            t0 + BATCH_SIZE,
            t_frames,
        )

        logging.info(
            "  Timepoints %d-%d",
            t0,
            t1 - 1,
        )

        batch_imgs = movie[
            t0:t1,
            :,
            CHANNEL_INDEX,
            :,
            :
        ]

        (
            batch_candidates,
            batch_thresholds,
        ) = detect_candidates_batch(
            batch_imgs,
            sigma=SIGMA,
            threshold_percentile=THRESH_PCNT,
            min_distance=MIN_DISTANCE,
        )

        # ---------------------------------------------------------------------
        # Process each frame
        # ---------------------------------------------------------------------

        for i, candidates in enumerate(
            batch_candidates
        ):

            frame = t0 + i

            segmentation_frame = segmentation[
                frame
            ]

            # -------------------------------------------------------------
            # Filter candidates by cell mask
            # -------------------------------------------------------------

            inside = candidates_inside_cell(
                candidates,
                segmentation_frame,
            )

            # -------------------------------------------------------------
            # Rank by smoothed NDC80 peak intensity
            # -------------------------------------------------------------

            retained = select_brightest_spots(
                inside,
                max_spots=MAX_SPOTS_PER_CELL,
            )

            # -------------------------------------------------------------
            # Build FINAL mask using retained spots only
            # -------------------------------------------------------------

            final_mask = build_mask_from_spots(
                retained,
                shape=(
                    z_slices,
                    height,
                    width,
                ),
                spot_radius=SPOT_RADIUS,
            )

            masks_out[frame] = final_mask

            # -------------------------------------------------------------
            # Store coordinates
            # -------------------------------------------------------------

            all_candidate_coordinates.append(
                candidates
            )

            all_inside_cell_coordinates.append(
                inside
            )

            all_retained_coordinates.append(
                retained
            )

            # -------------------------------------------------------------
            # Intensity QC
            # -------------------------------------------------------------

            if retained:

                strongest = retained[0][3]
                weakest = retained[-1][3]

            else:

                strongest = np.nan
                weakest = np.nan

            # -------------------------------------------------------------
            # CSV row
            # -------------------------------------------------------------

            count_rows.append(
                {
                    "movie": tif_path.stem,
                    "frame": frame,
                    "time_s": (
                        frame
                        * TIME_INTERVAL
                    ),
                    "threshold": (
                        batch_thresholds[i]
                    ),
                    "n_candidates_total": (
                        len(candidates)
                    ),
                    "n_candidates_inside_cell": (
                        len(inside)
                    ),
                    "n_spots_retained": (
                        len(retained)
                    ),
                    "max_spots_allowed": (
                        MAX_SPOTS_PER_CELL
                    ),
                    "weakest_retained_intensity": (
                        weakest
                    ),
                    "strongest_retained_intensity": (
                        strongest
                    ),
                }
            )

            logging.info(
                "    frame %d: "
                "%d candidates -> "
                "%d inside cell -> "
                "%d retained",
                frame,
                len(candidates),
                len(inside),
                len(retained),
            )

    # =========================================================================
    # Save outputs
    # =========================================================================

    # -------------------------------------------------------------------------
    # Final filtered mask
    # -------------------------------------------------------------------------

    out_mask = (
        OUTPUT_DIR
        / f"{tif_path.stem}_spotmasks.tif"
    )

    save_hyperstack_tiff(
        masks_out,
        out_mask,
    )

    logging.info(
        "  Saved final spot mask: %s",
        out_mask.name,
    )

    # -------------------------------------------------------------------------
    # All detector candidates
    # -------------------------------------------------------------------------

    out_candidates_xml = (
        OUTPUT_DIR
        / f"{tif_path.stem}_coordinates.xml"
    )

    save_coordinates_to_xml(
        all_candidate_coordinates,
        out_candidates_xml,
    )

    logging.info(
        "  Saved all candidates: %s",
        out_candidates_xml.name,
    )

    # -------------------------------------------------------------------------
    # Candidates inside cell before 92 cap
    # -------------------------------------------------------------------------

    out_inside_xml = (
        OUTPUT_DIR
        / f"{tif_path.stem}_inside_cell_coordinates.xml"
    )

    save_coordinates_to_xml(
        all_inside_cell_coordinates,
        out_inside_xml,
    )

    logging.info(
        "  Saved inside-cell candidates: %s",
        out_inside_xml.name,
    )

    # -------------------------------------------------------------------------
    # Final retained coordinates
    # -------------------------------------------------------------------------

    out_filtered_xml = (
        OUTPUT_DIR
        / f"{tif_path.stem}_filtered_coordinates.xml"
    )

    save_coordinates_to_xml(
        all_retained_coordinates,
        out_filtered_xml,
    )

    logging.info(
        "  Saved final retained spots: %s",
        out_filtered_xml.name,
    )

    # -------------------------------------------------------------------------
    # QC / count CSV
    # -------------------------------------------------------------------------

    out_counts_csv = (
        OUTPUT_DIR
        / f"{tif_path.stem}_spot_counts.csv"
    )

    save_counts_csv(
        count_rows,
        out_counts_csv,
    )

    logging.info(
        "  Saved spot-count QC: %s",
        out_counts_csv.name,
    )


# =============================================================================
# Movie discovery
# =============================================================================

def discover_movies(
    directory: Path,
) -> List[Path]:

    """
    Return base TIFF movies only.
    """

    excluded_suffixes = (
        "_segmented.tif",
        "_spotmasks.tif",
    )

    return sorted(
        p
        for p in directory.glob("*.tif")
        if not p.name.endswith(
            excluded_suffixes
        )
    )


# =============================================================================
# Save options
# =============================================================================

def save_options_yaml() -> None:

    options = {
        "channel_index": (
            CHANNEL_INDEX
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "sigma": list(
            SIGMA
        ),
        "threshold_percentile": (
            THRESH_PCNT
        ),
        "min_distance": (
            MIN_DISTANCE
        ),
        "spot_radius": (
            SPOT_RADIUS
        ),
        "max_spots_per_cell": (
            MAX_SPOTS_PER_CELL
        ),
        "pixel_size": (
            PIXEL_SIZE
        ),
        "time_interval_seconds": (
            TIME_INTERVAL
        ),
        "ranking_method": (
            "Gaussian-smoothed "
            "NDC80 peak intensity"
        ),
        "cell_filtering_method": (
            "peak centre must lie "
            "inside segmentation"
        ),
    }

    out_path = (
        OUTPUT_DIR
        / "options.yaml"
    )

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        yaml.safe_dump(
            options,
            handle,
            sort_keys=False,
        )

    logging.info(
        "Saved hyperparameters: %s",
        out_path,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "[%(asctime)s] "
            "%(levelname)s - "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    save_options_yaml()

    logging.info(
        "Searching for 5-D movies in %s",
        INPUT_DIR,
    )

    movies = discover_movies(
        INPUT_DIR
    )

    if not movies:

        logging.error(
            "No suitable .tif files found"
        )

        return 1

    logging.info(
        "Found %d movie(s)",
        len(movies),
    )

    for movie_path in movies:

        try:

            process_file(
                movie_path
            )

        except Exception:

            logging.exception(
                "Failed to process %s",
                movie_path.name,
            )

    logging.info(
        "All done - filtered spot masks "
        "stored in %s",
        OUTPUT_DIR,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
