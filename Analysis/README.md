# NDC80 / NUP107 recruitment analysis

These scripts are a **standalone analysis pipeline**, not part of the CelFDrive
acquisition workflow. They quantify NDC80 and NUP107 recruitment in
deconvolved, deskewed lattice light-sheet movies and produce the per-cell
timing statistics and aggregate intensity figures reported in the paper. They
are included here so that reviewers can inspect and re-run the analysis.

Nothing in `predict.py`, `CellClicker`, `benchmarks`, or `tools` imports this
folder, and nothing here imports CelFDrive. It has its own environment
(`environment-analysis.yml`), its own inputs, and its own outputs, and it is
**not** covered by the repository test suite.

## Requirements

An NVIDIA GPU with a CUDA 12 driver is required for step 1, which uses CuPy.
Steps 2–4 are CPU-only. Step 3 uses the two-argument form of
`Axes.set_xticks`, so matplotlib ≥ 3.5 is required; the pinned environment
satisfies this.

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

Channel 0 is NDC80 and channel 1 is NUP107. The analysed movies have a voxel
size of 0.271 µm axially and 0.104 µm laterally, and an 8 s frame interval.

## Measurement region

Intensities are measured in a region centred on each detected NDC80 spot. The
region is defined **in physical units**, so its size does not depend on the
anisotropic voxel spacing:

| Constant | Value | Meaning |
| --- | --- | --- |
| `USE_PHYSICAL_ROI` | `True` | define the region in µm rather than in voxels |
| `VOXEL_SIZE_UM` | `(0.271, 0.104, 0.104)` | voxel size in (z, y, x), µm |
| `SPOT_DIAMETER_UM` | `0.3` | diameter of the measurement sphere, µm |

At 0.271 µm axial sampling, a 0.3 µm sphere resolves to **9 voxels within a
single z-plane**. `find_filtered_spots_batch.py` prints this geometry
before doing any GPU work and records it in `options.yaml`, so the region
actually used for a run is always recoverable.

Setting `USE_PHYSICAL_ROI = False` reverts to the original behaviour, a sphere
of `SPOT_RADIUS` **voxels**. On anisotropic data that region is an ellipsoid in
physical space (33 voxels spanning ≈0.42 × 0.42 × 1.08 µm at radius 2), which
is why the physical definition is now the default. The legacy path is retained
so that the two can be compared directly.

## Other constants

Set `INPUT_DIR` in both `find_filtered_spots_batch.py` and
`batch_intensity_aggregate_with_spot_counts.py` before running them
individually, or use `run_roi_sweep.py --input-dir`, which sets it for you.
The remaining constants are the values used for the published analysis:

| Constant | Value | Meaning |
| --- | --- | --- |
| `CHANNEL_INDEX` / `CHANNEL_NDC80` | `0` | NDC80 channel |
| `CHANNEL_NUP` | `1` | NUP107 channel |
| `SIGMA` | `(1, 1, 1)` | Gaussian smoothing in (z, y, x) |
| `THRESH_PCNT` | `99.7` | candidate detection percentile |
| `MIN_DISTANCE` | `2` | minimum separation between local maxima, px |
| `MAX_SPOTS_PER_CELL` | `92` | **cap**, not a target: fewer detected spots are all retained |
| `PIXEL_SIZE` | `0.104` | µm/pixel, written to ImageJ metadata only |
| `TIME_INTERVAL` | `8` | seconds per frame |
| `AGGREGATE_INTENSITY_MODE` | `raw` | `raw` or `minmax`; affects aggregate statistics only |

`PIXEL_SIZE` is used only to tag output TIFFs; it takes no part in any
measurement.

Steps 1 and 2 create their output directories at import time, so importing
them has side effects. Run them, do not import them. (`recruitment_fitting.py`
is the exception and is meant to be imported.)

## Running everything at once

`run_roi_sweep.py` runs all four steps for each measurement region in turn and
prints each result as soon as it is available, so the first complete answer
appears before the GPU starts the next condition.

```powershell
python Analysis/run_roi_sweep.py --dry-run --input-dir <INPUT_DIR>
python Analysis/run_roi_sweep.py --input-dir <INPUT_DIR>
```

| Flag | Effect |
| --- | --- |
| `--dry-run` | print the plan, write nothing |
| `--only NAME` | run one condition; repeatable |
| `--resume` | skip steps whose outputs already exist |
| `--time-step SEC` | acquisition interval passed to steps 3 and 4 (default 8) |
| `--input-dir DIR` | overrides `INPUT_DIR` in this script and the steps it runs |

Conditions are listed in `CONDITIONS` at the top of the file. The default
sweep is 0.3 µm, 0.6 µm and the legacy voxel region. Outputs are written to
per-condition directories, so nothing is overwritten:

```
INPUT_DIR/filtered_spotmask_outputs_batch_<condition>/
INPUT_DIR/intensity_plots_<condition>/
```

The steps are configured by constants rather than command-line flags, so
`run_roi_sweep.py` writes a patched **copy** of steps 1 and 2 into
`Analysis/_sweep_work/<condition>/` and runs the copy. The originals are never
modified and never imported, and the patched copies remain afterwards as a
record of exactly what ran, alongside `step1.log`–`step4.log`.

## Running the steps individually

Run the four steps in order.

### 1. Detect and filter NDC80 spots (GPU)

```powershell
python Analysis/find_filtered_spots_batch.py
```

Smooths each 3-D timepoint, detects candidate local maxima, discards
candidates whose centre falls outside the segmented cell, ranks the remainder
by smoothed NDC80 intensity, and keeps at most `MAX_SPOTS_PER_CELL`.

Writes to `INPUT_DIR/filtered_spotmask_outputs_batch_<tag>/`:
`<movie>_spotmasks.tif`, `<movie>_coordinates.xml`,
`<movie>_filtered_coordinates.xml`, `<movie>_spot_counts.csv`, and
`options.yaml` recording the parameters actually used, including the
measurement-region geometry.

`ROI_SWEEP` at the top of this script can run several regions back to back
without the rest of the pipeline. `run_roi_sweep.py` sets it to `[]` and drives
one condition per invocation so that steps 2–4 can be interleaved.

### 2. Extract per-frame intensities and aggregate

```powershell
python Analysis/batch_intensity_aggregate_with_spot_counts.py
```

For every movie, measures mean NDC80 and NUP107 intensity in the intersection
of the spot mask and the cell mask at each frame, fits bounded 4-parameter
tanh curves, and interpolates each movie onto a common normalised 0–1 time
grid.

Writes to `INPUT_DIR/intensity_plots/`: `all_raw_intensities.csv` (long-format,
always unscaled), `aggregate_intensities.csv` (mean/SD/SEM/n),
`aggregate_intensity_plot.pdf`, and one dual-axis PDF per movie.

### 3. Per-cell recruitment timing

```powershell
python Analysis/analyse_recruitment_timing.py <INPUT_DIR>/intensity_plots/all_raw_intensities.csv
```

Optional: `--output-dir <directory>` (defaults to the input CSV's directory),
`--n-boot <N>` and `--seed <N>` for the bootstrap confidence intervals.

Fits each channel per cell, derives t50 recruitment times, and reports the
within-cell difference `delta_t50 = t50(NUP107) − t50(NDC80)` with bootstrap
confidence intervals, a paired t-test and a Wilcoxon signed-rank test.

Writes `per_cell_recruitment_times.csv`, `recruitment_time_summary.csv`,
`recruitment_t50_paired_plot.{png,pdf}`, and
`recruitment_delta_t50_plot.{png,pdf}`.

### 4. Aggregate plots aligned on the fitted NDC80 t50

```powershell
python Analysis/plot_ndc80_midpoint_dual_axis.py <INPUT_DIR>/intensity_plots/all_raw_intensities.csv
```

Optional: `--output-dir <directory>`, `--time-step <seconds>` (default 8; must
match the acquisition interval).

Fits the tanh model to each movie's NDC80 trace, shifts **both** channels of
that movie by the same fitted t50, interpolates onto a common grid, and plots
individual traces, mean ± SEM and aggregate fits on dual axes.

Writes `aligned_raw_intensities_ndc80_t50.csv`,
`aggregate_intensities_ndc80_t50.csv`, `ndc80_t50_alignment_qc.csv`, and
`aggregate_intensity_plot_ndc80_t50_{no_legend,legend_topleft}.{pdf,png}`.

## Shared fitting code

`recruitment_fitting.py` holds the single implementation of the bounded
four-parameter tanh model used by steps 2, 3 and 4. It has no import side
effects and writes nothing, so unlike the pipeline steps it is safe to import.

Each step wraps it in a thin adapter that preserves that step's own return
signature. The module was extracted from three previously duplicated copies
and reproduces all of them exactly: across the 18 traces of the published
dataset (9 movies x 2 channels) every fitted parameter, recruitment time and
plotting curve was bit-identical, and the resulting CSVs and figures were
unchanged.

Because that comparison cannot be repeated once the duplicated copies are
gone, the behaviour it established is pinned by a test that ships with the
release:

```powershell
python Analysis/test_recruitment_fitting.py
```

It checks accuracy against a trace built from known parameters, determinism,
recovery of inflections at either end of an acquisition, agreement between the
three adapter signatures, every documented failure mode, and stored golden
values for the fitted parameters. It needs only numpy and scipy, prints a pass
or fail line per check, and exits non-zero on failure.
