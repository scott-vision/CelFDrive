# Max-projection SAHI example

This example contains four illustrative microscope positions with three 16-bit
Z planes per position. The source pages were split in position-major order:
`p1z1.tif`, `p1z2.tif`, `p1z3.tif`, then position 2, through `p4z3.tif`.

`CelFDrive_max_projection_SAHI_example.ipynb` validates the complete grid,
maximum-projects each position, builds CelFDrive's `(height, width, position)`
montage, and runs the bundled 99.99-percentile detector. The notebook selects
SAHI only for the example: confidence 0.5, 640 px slices, 25% overlap, batches
of six, and class-aware greedy NMM with an IOU threshold of 0.1. It uses a
pixel spacing of 0.315 µm and seeded illustrative stage coordinates.

The `max_projections` TIFFs are reproducible derived files. Notebook overlays
are written below `outputs/`, which is intentionally ignored by Git. The data
and outputs demonstrate the software workflow only and are not biological
validation or calibrated acquisition coordinates.

The TIFF fixtures and trained detector are local data assets and are not stored
in Git. Add the position-major TIFF files beneath `z_slices/` and configure the
notebook with a suitable detector checkpoint before running the example.
