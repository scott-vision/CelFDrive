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

# Output directory is tagged with the ROI setting, so a physical-ROI run never
# overwrites the legacy outputs and runs at different diameters can be
# compared side by side.  RUN_TAG is built after the ROI config below.
OUTPUT_DIR = None  # set by _configure_output_dir() once the ROI is known


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

# ---------------------------------------------------------------------------
# Measurement region around each retained peak
#
# The original script drew the sphere in VOXEL space:
#     (dz**2 + dy**2 + dx**2) <= SPOT_RADIUS**2,  SPOT_RADIUS = 2 voxels
# Because the deskewed voxel is anisotropic (0.271 um in z, 0.104 um in x/y),
# that region is not a sphere in physical space: it reaches +/-0.21 um
# laterally but +/-0.54 um axially, i.e. ~0.42 x 0.42 x 1.08 um.
#
# Setting USE_PHYSICAL_ROI = True instead applies the distance test in
# micrometres, so the region is a true sphere of SPOT_DIAMETER_UM diameter.
#
# Note the axial sampling limit: with 0.271 um z-spacing, any diameter below
# ~0.542 um resolves to a single z-plane.  0.3 um therefore gives a 9-voxel
# disc in one plane; 0.6 um gives three planes.  Both are printed at startup.
# ---------------------------------------------------------------------------

USE_PHYSICAL_ROI = True

# Voxel size of the deconvolved, deskewed stacks, (z, y, x) in micrometres
VOXEL_SIZE_UM = (0.271, 0.104, 0.104)

# Diameter of the measurement region in micrometres (USE_PHYSICAL_ROI = True)
SPOT_DIAMETER_UM = 0.3

# Radius in voxels, used only when USE_PHYSICAL_ROI = False (legacy behaviour)
SPOT_RADIUS = 2

# Biological maximum
MAX_SPOTS_PER_CELL = 92

# Pixel size for ImageJ metadata, um/pixel in x and y
PIXEL_SIZE = VOXEL_SIZE_UM[1]

# Time between frames.  NOTE: this is the nominal interval.  The movie
# metadata records the achieved interval (ImageJ "finterval"), which may
# differ by a few per cent and scales all derived recruitment times.
TIME_INTERVAL = 8


def _configure_output_dir() -> Path:
    """Tag the output directory with the ROI actually used."""

    if USE_PHYSICAL_ROI:
        tag = f"physical_{SPOT_DIAMETER_UM:g}um".replace(".", "p")
    else:
        tag = f"legacy_r{SPOT_RADIUS}vox"

    out = INPUT_DIR / f"filtered_spotmask_outputs_batch_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    return out


OUTPUT_DIR = _configure_output_dir()


# ---------------------------------------------------------------------------
# Optional sweep: run several ROI settings back to back in one invocation.
#
# Each entry overrides the constants above for one run and writes to its own
# tagged output directory, so the results can be compared directly.  Set to []
# to run only the single configuration defined above.
#
# The default sweep is the comparison needed to show that the recruitment
# delay does not depend on ROI geometry:
#     0.3 um  biologically sized kinetochore region, one z-plane
#     0.6 um  smallest true sphere spanning three z-planes
#     legacy  reproduces the original radius-2-voxel region exactly
# ---------------------------------------------------------------------------

ROI_SWEEP = [
    {"USE_PHYSICAL_ROI": True, "SPOT_DIAMETER_UM": 0.3},
    {"USE_PHYSICAL_ROI": True, "SPOT_DIAMETER_UM": 0.6},
    {"USE_PHYSICAL_ROI": False, "SPOT_RADIUS": 2},
]


def _apply_roi_config(config: dict) -> None:
    """Override the ROI constants for one sweep entry."""

    global USE_PHYSICAL_ROI
    global SPOT_DIAMETER_UM
    global SPOT_RADIUS
    global OUTPUT_DIR

    USE_PHYSICAL_ROI = config.get(
        "USE_PHYSICAL_ROI", USE_PHYSICAL_ROI
    )
    SPOT_DIAMETER_UM = config.get(
        "SPOT_DIAMETER_UM", SPOT_DIAMETER_UM
    )
    SPOT_RADIUS = config.get(
        "SPOT_RADIUS", SPOT_RADIUS
    )

    OUTPUT_DIR = _configure_output_dir()


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
    radius: float,
    shape: Tuple[int, int, int],
    voxel_size: Tuple[float, float, float] = None,
) -> cp.ndarray:
    """
    Return a boolean CuPy array containing a filled sphere.

    voxel_size is None
        radius is in VOXELS and the distance test is applied in index space,
        reproducing the original behaviour.  On anisotropic data the result
        is an ellipsoid in physical space, not a sphere.

    voxel_size given as (dz, dy, dx) in micrometres
        radius is in MICROMETRES and the distance test is applied in physical
        space, so the region is a true sphere regardless of anisotropy.
    """

    z0, y0, x0 = center

    z, y, x = cp.indices(shape)

    if voxel_size is None:
        dist_sq = (
            (z - z0) ** 2
            + (y - y0) ** 2
            + (x - x0) ** 2
        )
    else:
        dz, dy, dx = voxel_size
        dist_sq = (
            ((z - z0) * dz) ** 2
            + ((y - y0) * dy) ** 2
            + ((x - x0) * dx) ** 2
        )

    return dist_sq <= radius ** 2


def roi_geometry() -> dict:
    """
    Describe the measurement region without needing a GPU.

    Returns the number of voxels included, how many z-planes they span, and
    the physical extent, so the region can be checked before a run and
    recorded in options.yaml afterwards.
    """

    import itertools

    dz, dy, dx = VOXEL_SIZE_UM

    if USE_PHYSICAL_ROI:

        radius = SPOT_DIAMETER_UM / 2.0

        reach = (
            int(radius // dz) + 1,
            int(radius // dy) + 1,
            int(radius // dx) + 1,
        )

        offsets = [
            o
            for o in itertools.product(
                range(-reach[0], reach[0] + 1),
                range(-reach[1], reach[1] + 1),
                range(-reach[2], reach[2] + 1),
            )
            if (o[0] * dz) ** 2
            + (o[1] * dy) ** 2
            + (o[2] * dx) ** 2
            <= radius ** 2
        ]

    else:

        radius = SPOT_RADIUS

        offsets = [
            o
            for o in itertools.product(
                range(-radius, radius + 1),
                repeat=3,
            )
            if o[0] ** 2 + o[1] ** 2 + o[2] ** 2 <= radius ** 2
        ]

    span = [
        max(abs(o[i]) for o in offsets)
        for i in range(3)
    ]

    return {
        "n_voxels": len(offsets),
        "z_planes": 2 * span[0] + 1,
        "extent_um_zyx": [
            round(2 * span[0] * dz, 4),
            round(2 * span[1] * dy, 4),
            round(2 * span[2] * dx, 4),
        ],
    }


def describe_roi() -> str:
    """Human-readable summary of the measurement region."""

    geom = roi_geometry()

    if USE_PHYSICAL_ROI:
        mode = (
            f"physical sphere, "
            f"{SPOT_DIAMETER_UM:g} um diameter"
        )
    else:
        mode = (
            f"legacy voxel sphere, "
            f"radius {SPOT_RADIUS} voxels"
        )

    ez, ey, ex = geom["extent_um_zyx"]

    return (
        f"Measurement region: {mode}\n"
        f"    voxel size (z,y,x) : {VOXEL_SIZE_UM} um\n"
        f"    voxels per spot    : {geom['n_voxels']}\n"
        f"    z-planes spanned   : {geom['z_planes']}\n"
        f"    extent (z,y,x)     : {ez} x {ey} x {ex} um"
    )



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
    spot_radius: float = None,
    voxel_size: Tuple[float, float, float] = None,
) -> np.ndarray:
    """
    Construct a binary 3-D mask using ONLY the retained spots.

    With USE_PHYSICAL_ROI the radius is in micrometres and voxel_size is
    supplied, so each region is a true sphere in physical space.  Otherwise
    the radius is in voxels and the original index-space test is used.
    """

    if spot_radius is None:
        spot_radius = (
            SPOT_DIAMETER_UM / 2.0
            if USE_PHYSICAL_ROI
            else SPOT_RADIUS
        )

    if voxel_size is None and USE_PHYSICAL_ROI:
        voxel_size = VOXEL_SIZE_UM

    mask_gpu = cp.zeros(
        shape,
        dtype=cp.uint8,
    )

    for z, y, x, _intensity in spots:

        sphere = create_sphere(
            center=(z, y, x),
            radius=spot_radius,
            shape=shape,
            voxel_size=voxel_size,
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
        "use_physical_roi": (
            USE_PHYSICAL_ROI
        ),
        "voxel_size_um_zyx": list(
            VOXEL_SIZE_UM
        ),
        "spot_diameter_um": (
            SPOT_DIAMETER_UM
            if USE_PHYSICAL_ROI
            else None
        ),
        "spot_radius_voxels": (
            None
            if USE_PHYSICAL_ROI
            else SPOT_RADIUS
        ),
        "roi_voxel_count": (
            roi_geometry()["n_voxels"]
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

def run_one() -> int:
    """Process every movie once, using the ROI currently configured."""

    for line in describe_roi().split("\n"):
        logging.info(line)

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

    configs = ROI_SWEEP or [None]

    status = 0

    for index, config in enumerate(configs, start=1):

        if config is not None:
            _apply_roi_config(config)

        if len(configs) > 1:
            logging.info(
                "===== ROI run %d of %d =====",
                index,
                len(configs),
            )

        status |= run_one()

    if len(configs) > 1:
        logging.info(
            "Sweep complete: %d ROI setting(s) processed",
            len(configs),
        )

    return status


if __name__ == "__main__":
    sys.exit(main())
