# CelFDrive

CelFDrive is a deep-learning-assisted workflow for automated microscopy. It detects configured cell states in overview images and returns stage coordinates and capture metadata for a high-resolution workflow.

This repository currently supports local development on Windows and macOS with Python 3.11. The bundled synthetic fixture is an installation smoke test; it is not a biological benchmark or evidence of model generalisation.

## Getting the code and data

The code is small. The large files - the detector weights, the tutorial TIFFs,
and the example SlideBook experiment - are stored with Git LFS and are
downloaded only when you ask for them. Install Git LFS once per machine with
`git lfs install` before cloning.

**Code only** (~11 MB, a few seconds). Use this if you want to read the source,
review it, or work on it:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = 1
git clone https://github.com/scott-vision/CelFDrive.git
Remove-Item Env:\GIT_LFS_SKIP_SMUDGE
```

The large files are present as small text pointer files. Everything else - the
tests, the configuration editor, the annotation GUIs - works normally.

**Then add only what you need**, from inside the clone:

| Command | Downloads | Needed for |
| --- | --- | --- |
| `git lfs pull --include="Models/yolo11x_p99p99_bg05/**"` | 114 MB | running prediction, `examples/run_smoke_test.py`, the SlideBook workflow |
| `git lfs pull --include="benchmarks/data/cellclicker/cellcognition_h2b_P0037/**"` | 162 MB | reproducing the CellCognition P0037 exported-label benchmark |
| `git lfs pull --include="examples/**"` | 108 MB | the max-projection SAHI notebook |
| `git lfs pull --include="SlideBook/**"` | 29 MB | opening the example SlideBook experiment |
| `git lfs pull --include="Models/yolo11x_p99p99_bg05_noaug_v1/**"` | 229 MB | the no-augmentation ablation checkpoint only |
| `git lfs pull` | 480 MB | everything |

**Everything at once**: a plain `git clone` with no environment variable
downloads the code and all 480 MB of large files.

**Without git**: download the [current Zenodo release archive](https://doi.org/10.5281/zenodo.22115209).
It is a single zip containing the code and every large file, and needs neither
git nor Git LFS. This is the recommended route for reviewers and for citing a
specific version.

If Git LFS is not installed, or a `git lfs pull` has not been run, the detector
weights stay as a pointer file. `python -m pytest -q` and
`python examples/run_smoke_test.py` both report this explicitly rather
than failing obscurely.

## Install and use CelFDrive

On Windows, create the GPU environment from the repository root:

```powershell
.\tools\create_windows_conda_env.ps1 -Device gpu
```

For CPU-only Windows, macOS, environment verification, and the optional pip
virtual environment, see the [installation and verification guide](docs/installation.md).

Then choose the workflow you need:

- **Configure prediction:** run `python run_config_gui.py`, then follow the
  [prediction configuration](#prediction-configuration) section.
- **Connect a 3i SlideBook microscope:** follow the [SlideBook direct-Python
  setup guide](docs/slidebook-capture-script.md).
- **Annotate images or train a phase model:** start `python run_clicker_gui.py`
  and use the [CellClicker interface guide](docs/cellclicker-interface-guide.md).
- **Try the SlideBook-independent inference workflow:** run the
  [max-projection SAHI notebook example](examples/max_projection_sahi/README.md).
- **Reproduce the exported-label benchmark:** use the
  [benchmark workflow](docs/benchmarking.md).

## Prediction configuration

`celfdrive_predict.yaml` is the default configuration. It defines the model weights, preprocessing, tiling, coordinate conversion, logging, capture profile, and class-specific confidence thresholds. `run_config_gui.py` provides an editor for these settings:

```powershell
python run_config_gui.py
```

The default configuration uses the bundled 99.99-percentile YOLO11x weights at `Models/yolo11x_p99p99_bg05/weights/best.pt`. Relative model and logging paths are resolved from **CelFDrive project root**, including when SlideBook starts Python from its scripts directory. In the editor, set **CelFDrive project root** to the folder cloned from GitHub; the default `Logging` directory is created beneath it. Configure capture-script names and class settings for the local experiment. Do not edit Python globals to configure a workflow.

Tiling is the default inference mode. The Image tab provides explicit Tiling, Full image, and SAHI modes, showing only the settings for the selected mode. SAHI uses validated confidence, slice-size, overlap, batch-size, and class-aware IOU merging. See [`examples/max_projection_sahi`](examples/max_projection_sahi) for an executed four-position, three-Z-plane maximum-projection example using 0.315 µm/pixel, SAHI confidence 0.5, and merge IOU 0.1.

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

`train.py` is the configuration-driven YOLO training entry point. Pass a schema-versioned YAML file with `--config`, using `examples/yolo_training.example.yaml` as the starting template; `--name` can override only the run name for one invocation. The command-line and GUI entry points share the same validation, dataset preparation, training and held-out-test evaluation workflow.

Benchmark, evaluation, and report scripts are organised as modules in
`benchmarks`; use `python -m benchmarks.run_benchmark --help` to inspect the
frozen benchmark commands. Dataset-preparation utilities are likewise in
`tools` and are invoked with `python -m tools.<module>`.

## Recruitment analysis scripts

`Analysis` holds a standalone NDC80/NUP107 recruitment pipeline used for the
paper figures. It is not part of the acquisition workflow, nothing in CelFDrive
imports it, it needs its own CuPy environment, and it is not covered by the test
suite. See [`Analysis/README.md`](Analysis/README.md) for its inputs, parameters,
and the order in which to run the four steps.

## Microscope integration and limitations

CelFDrive's current 3i-oriented SlideBook workflow invokes Python directly through SlideBook's Python hierarchical-capture support; it does not require MATLAB. The supplied bridge accepts a raw montage, converts SlideBook's image-axis order, and returns capture targets. Follow the [SlideBook direct-Python setup guide](docs/slidebook-capture-script.md), including registration of the Conda environment in SlideBook and microscope-specific objective-offset validation. Adaptations to other systems must provide their own coordinate convention, capture command integration, and hardware safety checks.

Only the bundled Ultralytics YOLO workflow is supported and covered by the smoke test.

## Licence

CelFDrive is distributed under the terms in [`LICENSE.md`](LICENSE.md): a
bespoke University of Warwick licence for non-commercial use.

[`THIRD_PARTY_LICENCES.md`](THIRD_PARTY_LICENCES.md) records the licences of the
software CelFDrive depends on and of the bundled detector checkpoints, which
were fine-tuned with Ultralytics training code and carry Ultralytics'
AGPL-3.0 terms. Read it before redistributing CelFDrive or any model trained
with it.

## Citing CelFDrive

Please cite the [CelFDrive Zenodo concept DOI](https://doi.org/10.5281/zenodo.22115209).
The exact-version DOI is shown on each Zenodo release record. The repository's
[`CITATION.cff`](CITATION.cff) supplies machine-readable citation metadata and
[`.zenodo.json`](.zenodo.json) supplies the release metadata.

## Developer checklist

Before sharing a change:

1. Create the documented Conda environment on the target platform.
2. Run `python -m pytest -q`.
3. Run `python examples/run_smoke_test.py`.
4. Confirm configuration examples, expected output, and README commands still agree with the changed behavior.
