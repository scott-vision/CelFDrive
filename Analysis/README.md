# NDC80 / NUP107 recruitment analysis

These four scripts are a **standalone analysis pipeline**, not part of the
CelFDrive acquisition workflow. They quantify NDC80 and NUP107 recruitment in
deconvolved, deskewed lattice light-sheet movies and produce the per-cell
timing statistics and aggregate intensity figures reported in the paper. They
are included here so that reviewers can inspect and re-run the analysis.

Nothing in `predict.py`, `CellClicker`, `benchmarks`, or `tools` imports this
folder, and nothing here imports CelFDrive. It has its own environment
(`environment-analysis.yml`), its own inputs, and its own outputs, and it is
**not** covered by the repository test suite.

## Requirements

An NVIDIA GPU with a CUDA 12 driver is required for step 1, which uses CuPy.
Steps 2-4 are CPU-only.

```powershell
conda env create --file Analysis/environment-analysis.yml
conda activate celfdrive-analysis
```

## Input data

The pipeline starts from one directory (`INPUT_DIR`) containing:

| Item | Location | Produced by |
| --- | --- | --- |
| 5-D movie stacks `(t, z, c, y, x)` as `<movie>.tif` | `INPUT_DIR/` | deconvolution and deskewing, outside this repository |
| Per-movie cell masks `<movie>_segmented.tif` | `INPUT_DIR/segmentation_outputs_batch/` | cell segmentation, outside this repository |

Channel 0 is NDC80 and channel 1 is NUP107. The analysed movies were acquired
at 0.1625 µm/pixel with an 8 s frame interval.

## Configuration

Steps 1 and 2 are batch scripts configured by editing the constants in the
`Configuration` block at the top of each file, not by a YAML file or command
line flags. This is deliberate for these one-off analysis scripts and is the
one place in the repository where that pattern is used; the CelFDrive
prediction workflow is configured through `celfdrive_predict.yaml` and must not
be configured this way.

Before running, set `INPUT_DIR` in both `find_filtered_spots_batch.py` and
`batch_intensity_aggregate_with_spot_counts.py` to your data directory. The
remaining constants are the values used for the published analysis:

| Constant | Value | Meaning |
| --- | --- | --- |
| `CHANNEL_INDEX` / `CHANNEL_NDC80` | `0` | NDC80 channel |
| `CHANNEL_NUP` | `1` | NUP107 channel |
| `SIGMA` | `(1, 1, 1)` | Gaussian smoothing in (z, y, x) |
| `THRESH_PCNT` | `99.7` | candidate detection percentile |
| `MIN_DISTANCE` | `2` | minimum separation between local maxima, px |
| `SPOT_RADIUS` | `2` | radius of the sphere drawn around each retained peak, px |
| `MAX_SPOTS_PER_CELL` | `92` | **cap**, not a target: fewer detected spots are all retained |
| `PIXEL_SIZE` | `0.1625` | µm/pixel, written to ImageJ metadata |
| `TIME_INTERVAL` | `8` | seconds per frame |
| `AGGREGATE_INTENSITY_MODE` | `raw` | `raw` or `minmax`; affects aggregate statistics only |

Both scripts create their output directories at import time, so importing them
has side effects. Run them, do not import them.

## Pipeline

Run the four steps in order.

### 1. Detect and filter NDC80 spots (GPU)

```powershell
python Analysis/find_filtered_spots_batch.py
```

Smooths each 3-D timepoint, detects candidate local maxima, discards
candidates whose centre falls outside the segmented cell, ranks the remainder
by smoothed NDC80 intensity, and keeps at most `MAX_SPOTS_PER_CELL`.

Writes to `INPUT_DIR/filtered_spotmask_outputs_batch/`:
`<movie>_spotmasks.tif`, `<movie>_coordinates.xml`,
`<movie>_filtered_coordinates.xml`, `<movie>_spot_counts.csv`, and
`options.yaml` recording the parameters actually used for the run.

### 2. Extract per-frame intensities and aggregate

```powershell
python Analysis/batch_intensity_aggregate_with_spot_counts.py
```

For every movie, measures mean NDC80 and NUP107 intensity in the intersection
of the spot mask and the cell mask at each frame, fits bounded 4-parameter
tanh curves, and interpolates each movie onto a common normalised 0-1 time
grid.

Writes to `INPUT_DIR/intensity_plots/`: `all_raw_intensities.csv` (long-format,
always unscaled), `aggregate_intensities.csv` (mean/SD/SEM/n),
`aggregate_intensity_plot.pdf`, and one dual-axis PDF per movie.

### 3. Per-cell recruitment timing

```powershell
python Analysis/analyse_recruitment_timing.py <INPUT_DIR>/intensity_plots/all_raw_intensities.csv
```

Optional: `--output-dir <directory>` (defaults to the input CSV's directory).

Fits each channel per cell, derives t50 recruitment times, and runs paired
tests (paired t-test and Wilcoxon) on the NUP107-minus-NDC80 difference.

Writes `per_cell_recruitment_times.csv`, `recruitment_time_summary.csv`,
`recruitment_t50_paired_plot.{png,pdf}`, and
`recruitment_delta_t50_plot.{png,pdf}`.

### 4. Aggregate plots aligned on the NDC80 peak

```powershell
python Analysis/plot_ndc80_raw_peak_dual_axis.py <INPUT_DIR>/intensity_plots/all_raw_intensities.csv
```

Optional: `--output-dir <directory>`.

Re-aligns every movie on its own raw NDC80 peak instead of on normalised time.

Writes `aligned_raw_intensities_ndc80_raw_peak.csv`,
`aggregate_intensities_ndc80_raw_peak.csv`, and the dual-axis aggregate figure
with and without a legend, each as PDF and PNG.

Steps 3 and 4 both require the columns `movie`, `time_s`, `ndc80_mean_raw`, and
`nup_mean_raw`, and fail with an explicit error if any are missing.

## Scope

This analysis is specific to the NDC80/NUP107 recruitment experiment. It is not
a general-purpose intensity pipeline, it is not validated on other markers or
acquisition geometries, and its parameters were chosen for the movies described
above.
