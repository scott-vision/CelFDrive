# CellCognition H2B P0037 reproduction fixture

This directory contains the complete reproducibility inputs for the
CellClicker exported-label benchmark:

- `images/` contains 206 normalized PNG frames for position P0037.
- `user_selections/exported_labels/` contains the 206 exported YOLO label
  files and `export_manifest.json` used by the benchmark.

The source project was generated from the P0037 H2B-mCherry image sequence.
Each frame was clipped at its 99.99th percentile and min-max normalized to an
8-bit PNG for CellClicker display. These PNGs, rather than the original TIFFs,
are the benchmark inference inputs. Their 1392 x 1040 pixel coordinates match
the exported label coordinates.

The PNG files are stored with Git LFS. For a source-only clone, retrieve them
with:

```powershell
git lfs pull --include="benchmarks/data/cellclicker/cellcognition_h2b_P0037/**"
```

Run the documented command in `docs/benchmarking.md` to reproduce the
exported-label evaluation. This fixture is a fixed CellClicker project and not
an independent biological test set.
