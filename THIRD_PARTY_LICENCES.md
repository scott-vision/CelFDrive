# Third-party licences

CelFDrive's own source code is licensed under the terms in
[`LICENSE.md`](LICENSE.md). This file records the third-party software and
model weights that CelFDrive redistributes or depends on, so that anyone using
or reviewing the repository can see what other terms apply and to which parts.

It is an informational summary. The authoritative terms are those published by
each project, and licences can change between versions. Commercial enquiries
about CelFDrive go to Warwick Innovations (ventures@warwick.ac.uk, reference
"Warwick CelFDrive"); enquiries about a third-party component go to that
project.

## 1. Files redistributed in this repository

Only one category of file in this repository is not original CelFDrive work.

| Files | Origin | Terms |
| --- | --- | --- |
| `Models/yolo11x_p99p99_bg05/weights/best.pt`<br>`Models/yolo11x_p99p99_bg05_noaug_v1/weights/best.pt`<br>`Models/yolo11x_p99p99_bg05_noaug_v1/weights/last.pt` | Fine-tuned from the Ultralytics YOLO11x architecture and pretrained weights, using Ultralytics training code | AGPL-3.0, per Ultralytics (see section 4) |

Everything else in the repository - the Python source, configuration files,
documentation, screenshots, the sample holder CAD files, the SlideBook scripts
and example experiment, the synthetic smoke-test fixture, the example
microscopy TIFFs and the CellClicker sample labelling data - is original work
of the authors and is covered by [`LICENSE.md`](LICENSE.md).

No third-party source code is vendored into this repository. Every dependency
below is installed by the user into their own environment from conda-forge or
PyPI, and is not redistributed by CelFDrive.

## 2. Runtime dependencies

Installed from the environment files in [`Environments`](Environments), or from
[`Analysis/environment-analysis.yml`](Analysis/environment-analysis.yml) for the
recruitment analysis scripts.

| Package | Licence | Used for |
| --- | --- | --- |
| Ultralytics | **AGPL-3.0-or-later** | YOLO11 detection and training; SAM2 access. See section 4. |
| SAHI | MIT | Sliced inference in SAHI mode |
| PyTorch (`torch`) | BSD-3-Clause (with bundled Apache-2.0, BSD-2-Clause, BSL-1.0 and MIT components) | Model execution |
| torchvision | BSD-3-Clause | Image transforms required by Ultralytics |
| torchaudio | BSD-2-Clause | Installed as part of the PyTorch stack |
| OpenCV (`opencv`, `opencv-python`) | Apache-2.0 | Image preprocessing, annotation overlays |
| NumPy | BSD-3-Clause | Array handling throughout |
| SciPy | BSD-3-Clause | Curve fitting in the recruitment analysis |
| pandas | BSD-3-Clause | Label tables, benchmark outputs |
| Matplotlib | Matplotlib licence (PSF-based, BSD-compatible) | Figures and prediction overlays |
| scikit-image | BSD-3-Clause | Peak detection in spot analysis |
| Pillow | MIT-CMU (HPND) | Image loading in the Tk interfaces |
| seaborn | BSD-3-Clause | Benchmark figures |
| tqdm | MPL-2.0 and MIT | Progress reporting |
| PyYAML | MIT | Configuration files |
| h5py | BSD-3-Clause | HDF5 input |
| tifffile | BSD-3-Clause | TIFF reading and ImageJ-compatible writing |
| imageio | BSD-2-Clause | Image IO used by scikit-image |
| NetworkX | BSD-3-Clause | Transitive dependency |
| dill | BSD-3-Clause | Transitive dependency |
| fsspec | BSD-3-Clause | Transitive dependency |
| openpyxl | MIT | Spreadsheet output |
| PyArrow | Apache-2.0 | Parquet benchmark tables |
| Shapely | BSD-3-Clause | Required by SAHI |
| click | BSD-3-Clause | Required by SAHI |
| Fire | Apache-2.0 | Required by SAHI |
| Requests | Apache-2.0 | Required by SAHI |
| terminaltables | MIT | Required by SAHI |
| CuPy | MIT | GPU spot detection in `Analysis` only |
| pytest | MIT | Test suite |
| Jupyter | BSD-3-Clause | The max-projection SAHI example notebook |

## 3. Downloaded at run time, not redistributed

These files are fetched by Ultralytics into the working directory the first
time a feature needs them. They are excluded by `.gitignore` and are not part
of this repository or of any release archive.

| File | Origin | Feature |
| --- | --- | --- |
| `sam2_b.pt` | Ultralytics SAM2 assets | SAM2 box generation in `CellClicker/tracking_sam2.py` |
| `yolo11n.pt` and other pretrained YOLO11 checkpoints | Ultralytics assets | Starting weights for training in `CellClicker/yolo_training.py` and `train.py` |

Both are obtained under Ultralytics' terms, directly by the user's own
installation of Ultralytics.

## 4. Note on Ultralytics

Ultralytics is published under AGPL-3.0-or-later, with a separate commercial
Enterprise licence available from Ultralytics. Ultralytics states that its
AGPL-3.0 licence "covers the training code and the models produced by that
training code", and that an Enterprise licence is required for commercial use
of custom-trained models.

CelFDrive does not vendor or redistribute any Ultralytics source code. It
imports Ultralytics at run time, from the user's own environment, in
`predict.py`, `CellClicker/yolo_training.py`, `CellClicker/mitotic_tightener.py`
and `CellClicker/tracking_sam2.py`, each behind an explicit import guard. It
does redistribute the fine-tuned checkpoints listed in section 1, which were
produced with Ultralytics training code.

Anyone intending to use CelFDrive, its bundled checkpoints, or any model
trained with it outside non-commercial academic research should review both
[`LICENSE.md`](LICENSE.md) and Ultralytics' licensing terms at
https://www.ultralytics.com/license, and contact Warwick Innovations and
Ultralytics as appropriate.

## 5. Keeping this list current

This list reflects the packages named in the environment files. To check what a
built environment actually contains:

```powershell
conda list --name celfdrive-windows --json | python -c "import json,sys; [print(p['name'], p['version']) for p in json.load(sys.stdin)]"
```

Licence text for each installed package is in its `site-packages` distribution
metadata within the created environment.
