# Annotation and training workflow

CelFDrive's preferred annotation workflow is the unified project GUI. Start it
from the repository root with:

```powershell
python run_gui.py
```

Select a project directory containing `images/`. The GUI reports whether each
workflow input is available and guides users through CellClicker annotation,
multiphase selection, aggregation, track review, export, and optional training.

## Project layout

The project directory is not part of this repository and must contain:

```text
project/
  images/
    cell_reigons.xml
    ... microscopy image files ...
  user_selections/
```

`cell_reigons.xml` is intentionally spelled this way for compatibility with
existing projects. The GUI creates `user_selections/` when phase annotations
are first saved.

Projects may be moved as a complete folder. When `tracking_review.xml` still
contains paths below its former `dataset_root/images/` directory, CelFDrive
resolves those image-relative paths below the moved project's current
`images/` directory. The repaired paths and dataset root are persisted the
next time the tracking XML is saved.

## Tracking review

After aggregating phase selections, **Build Tracking XML** creates
`user_selections/tracking_review.xml`. It is the canonical downstream record:
tracks retain their identity and chronological timepoints, each with an image
path, phase/class, and one or more normalized YOLO-format box variants.

The review editor lets users select the preferred variant and edit boxes. Otsu,
SAM2, and the trained YOLO11 mitotic tightener create additional variants
(`otsu`, `sam2`, and `yolo11_tightened`) rather than replacing the original
annotation; manual refinement uses `tightened`.

## Export and training

Choose a box variant and export YOLO labels, COCO annotations, or cropped
miniseries. Export requires every referenced image to be below the selected
project's `images/` directory; CelFDrive raises an actionable error if the
tracking XML metadata does not match the project.

The YOLO training window is optional. It expects each selected project to have
`tracking_review.xml` and exported YOLO labels. It creates `labels/` inside
each selected project and a dataset YAML at the location chosen by the user.
SAM2 and training require the optional `ultralytics` dependency; SAM2 also
requires a compatible CPU or CUDA PyTorch installation. Model weights and
datasets are never bundled with this repository.

## Mitotic tightener

**Train Mitotic Tightener** accepts explicit project-folder lists for train,
validation, and held-out test splits. It creates class-agnostic, review-style
crops around every original box and labels each crop with its preferred review
box. Datasets are written beneath `Models/tightener_datasets/`; the input size
is selected from the largest crop, rounded to the YOLO stride and capped at
320 pixels. The window reports per-epoch validation output and evaluates the
best checkpoint on the test split after training.

Each generated dataset includes `training_manifest.csv`, which records every
included or skipped timepoint, its source project and image, selected preferred
box, review-crop bounds, output crop/label paths, and any skip reason.

To use a trained model in a project, select **Configure Tightener** in the
project GUI and choose its `best.pt`. The path is saved in that project's
`tracking_review.xml`, together with the model's training image size, so
inference uses the same Ultralytics input resolution. **Run YOLO11 Tightener** then writes a
`yolo11_tightened` alternative for each timepoint. If it cannot detect a box,
it records an original-box fallback so every timepoint still has a selectable
variant. When several boxes are detected, CelFDrive selects the one with the
highest confidence that contains the original prompt-box centre by default. If
none contain that centre, it uses greatest overlap with the original prompt
box. **Configure Tightener** can persistently select pure overlap or confidence
selection for a project.

## Legacy commands

`run_clicker.py`, `run_selector.py`, and `run_conversion.py` remain available
for existing workflows. New projects should use the unified GUI so tracking
identity and review decisions are preserved before export.
