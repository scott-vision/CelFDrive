# CelFDrive

CelFDrive is a deep-learning-assisted workflow for automated microscopy. It detects configured cell states in overview images and returns stage coordinates and capture metadata for a high-resolution workflow.

This repository currently supports local development on Windows and macOS with Python 3.11. The bundled sample is an installation smoke test; it is not a biological benchmark or evidence of model generalisation.

## Install

Create the platform environment from the repository root:

```powershell
conda env create -f Environments/environment-gpu-windows.yml
conda activate celfdrive-windows
```

On macOS, use `Environments/environment-gpu-mac.yml` and activate `celfdrive-macos` instead. Both environment files use `conda-forge` as their only Conda channel. PyTorch and model packages are installed with pip according to their vendor distribution.

The Windows environment installs the default PyTorch package. For CUDA acceleration, replace it with the PyTorch wheel command matching your GPU driver from the [official PyTorch installer](https://pytorch.org/get-started/locally/), then rerun the verification commands below. CUDA operation is not verified by this repository's local smoke test.

## Quick start and local verification

From the repository root, run:

```powershell
python -m pytest -q
python examples/run_sample_workflow.py
```

The test suite requires no microscope hardware or downloads. The sample command loads `sample_data/synthetic_blank_image.csv`, runs the bundled model, prints JSON detections, and exits successfully only when they match `sample_data/expected_detections.json`.

The blank sample is deliberately synthetic because an approved redistributable microscopy subset is not yet included. See [sample_data/README.md](sample_data/README.md) before treating it as anything other than an installation check.

YOLO phase-model training is available from the unified CellClicker GUI or
from the same versioned configuration on the command line:

```powershell
python train.py --config path\to\training.yaml
```

The GUI can load and save these files; see
[`examples/yolo_training.example.yaml`](examples/yolo_training.example.yaml)
and the [CellClicker interface guide](docs/cellclicker-interface-guide.md#7-train-a-phase-model-yolo11).

For a frozen, paper-oriented evaluation on a separately held-out internal data
set, see [the benchmark workflow](docs/benchmarking.md). It writes outputs to
`D:\CelFDriveBenchmark\runs` by default and does not modify source labels or
paper files.

## Prediction configuration

`celfdrive_predict.yaml` is the default configuration. It defines the model weights, preprocessing, tiling, coordinate conversion, logging, capture profile, and class-specific confidence thresholds. `run_config_gui.py` provides an editor for these settings:

```powershell
python run_config_gui.py
```

The default configuration assumes commands are run from the repository root and uses the bundled 99.99-percentile YOLO11x weights at `Models/yolo11x_p99p99_bg05/weights/best.pt`. This is a simple relative path, not a value users normally need to edit. In the editor, set **CelFDrive project root** to the folder cloned from GitHub; the default `Logging` directory is created beneath it. Configure capture-script names and class settings for the local experiment. Do not edit Python globals to configure a workflow.

Standard tiling remains the default inference mode. The Image tab also exposes an optional SAHI mode with validated confidence, slice-size, overlap, batch-size, and class-aware IOU merge settings. See [`examples/max_projection_sahi`](examples/max_projection_sahi) for an executed four-position, three-Z-plane maximum-projection example using 0.315 µm/pixel, SAHI confidence 0.5, and merge IOU 0.1.

In the **High Resolution Imaging** tab, the SlideBook postscan script must have the same name as `profile.highres_script`. Detection classes use a per-class minimum confidence; capture priority `0` runs first and `-1` disables a class without deleting it.

### Prediction input and output

`predict.get_target_locations` is the preferred named-argument API:

```python
from predict import get_target_locations

targets = get_target_locations(
    stage_x=[100.0],
    stage_y=[200.0],
    stage_z=[5.0],
    image=montage_stack,  # shape: (height, width, position)
    xy_pixel_spacing_um=0.325,
    x_stage_direction=1,
    y_stage_direction=1,
    coordinate_mode="stage",
)
```

Programs that construct a configuration in memory should initialise that
workflow explicitly with `predict.configure_prediction_runtime(config)`. This
keeps its model cache and logging run directory separate from other prediction
workflows in the same Python process.

`coordinate_mode` is one of:

- `stage` (default): converts detections to the current CelFDrive physical-stage convention. `xy_pixel_spacing_um` must be positive.
- `pixel`: returns detection-centre X/Y values in image pixels. Use this only when the calling integration performs its own coordinate conversion.
- `callable`: invokes a caller-supplied `coordinate_converter` and uses its `(x, y, z)` output. Callables cannot be stored in YAML.

For callable mode, provide a function accepting these keyword arguments:

```python
def coordinate_converter(
    *, stage_x, stage_y, stage_z,
    detection_x_px, detection_y_px,
    image_width_px, image_height_px,
    xy_pixel_spacing_um, z_offset_um,
    class_id, confidence, class_name,
):
    return converted_x, converted_y, converted_z
```

The converter must return exactly three finite numeric values. The input montage stack must be a NumPy array with shape `(height, width, position)`; its final axis must match the number of stage positions. Stage directions must be `1` or `-1`.

`predict.get_target_location` remains available for existing microscope integrations:

```python
get_target_location(
    X, Y, Z, image,
    xy_pixel_spacing, z_spacing,
    x_stage_direction, y_stage_direction, z_stage_direction,
    LLSM=False, z_offset=None,
)
```

`X`, `Y`, and `Z` are stage positions; `image` must be a montage stack of shape `(height, width, position)`. Pixel spacing and offsets are in micrometres; stage directions are `1` or `-1`. The wrapper retains `z_spacing` and `z_stage_direction` for compatibility, although two-dimensional target conversion does not use them. Both APIs return `(count, X, Y, Z, scripts, names, comments)` for use by the capture workflow.

The current stage conversion is specific to the existing 3i conventions. The legacy `LLSM` flag still inverts Y direction when configured. In standard mode, `tiling.overlap_px` controls tile overlap and same-class detections whose centres are within `tiling.deduplication_tolerance_px` are de-duplicated before coordinate conversion. SAHI mode instead performs class-aware greedy NMM in full-image coordinates using its configured IOU threshold. Validate coordinates on the target microscope before acquiring data.

## Annotation and training tools

For new annotation projects, use the unified GUI:

```powershell
python run_clicker_gui.py
```

It centralises project availability checks and guides the complete workflow:
track annotation, phase selection, aggregation, track review, optional Otsu or
SAM2 box generation, YOLO/COCO/miniseries export, and optional YOLO training.
See the [CellClicker user guide](docs/cellclicker-interface-guide.md) for the
required project layout, tracking XML semantics, and hardware requirements.

The scripts below are retained for legacy workflows:

CellClicker and CellSelector create YOLO-compatible labels from time-series data:

```powershell
python -m CellClicker.legacy.run_clicker
python -m CellClicker.legacy.run_selector
python -m CellClicker.legacy.run_conversion
```

Set the phase names in `CellClicker/legacy/run_selector.py` and the dataset/user values in `CellClicker/legacy/run_conversion.py` for the experiment. CellSelector opens an interactive GUI, so its selection workflow must be checked manually. The entry point now forwards the configured phase list correctly.

`train.py` is the configuration-driven YOLO training entry point. Pass a schema-versioned YAML file with `--config`, using `examples/yolo_training.example.yaml` as the starting template; `--name` can override only the run name for one invocation. The command-line and GUI entry points share the same validation, dataset preparation, training and held-out-test evaluation workflow.

Benchmark, evaluation, and report scripts are organised as modules in
`benchmarks`; use `python -m benchmarks.run_benchmark --help` to inspect the
frozen benchmark commands. Dataset-preparation utilities are likewise in
`tools` and are invoked with `python -m tools.<module>`.

## Microscope integration and limitations

CelFDrive's current 3i-oriented SlideBook workflow invokes Python directly through SlideBook's Python hierarchical-capture support; it does not require MATLAB. The supplied bridge accepts a raw montage, converts SlideBook's image-axis order, and returns capture targets. Follow the [SlideBook direct-Python setup guide](docs/slidebook-capture-script.md), including registration of the Conda environment in SlideBook and microscope-specific objective-offset validation. Adaptations to other systems must provide their own coordinate convention, capture command integration, and hardware safety checks.

Only the bundled Ultralytics YOLO workflow is supported and covered by the smoke test.

## Developer checklist

Before sharing a change:

1. Create the documented Conda environment on the target platform.
2. Run `python -m pytest -q`.
3. Run `python examples/run_sample_workflow.py`.
4. Confirm configuration examples, expected output, and README commands still agree with the changed behavior.
