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

## Tracking review

After aggregating phase selections, **Build Tracking XML** creates
`user_selections/tracking_review.xml`. It is the canonical downstream record:
tracks retain their identity and chronological timepoints, each with an image
path, phase/class, and one or more normalized YOLO-format box variants.

The review editor lets users select the preferred variant and edit boxes. Otsu
and SAM2 create additional variants (`otsu` and `sam2`) rather than replacing
the original annotation; manual refinement uses `tightened`.

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

## Legacy commands

`run_clicker.py`, `run_selector.py`, and `run_conversion.py` remain available
for existing workflows. New projects should use the unified GUI so tracking
identity and review decisions are preserved before export.
