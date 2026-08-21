# Frozen benchmark workflow

`python -m benchmarks.run_benchmark` evaluates a frozen checkpoint without editing input images,
labels, paper files, or prior results.  It writes each run to a unique folder
under `D:\CelFDriveBenchmark\runs` by default.

Run it from the GPU environment:

```powershell
conda run -n celfdrive-gpu python -m benchmarks.run_benchmark run --config D:\path\to\paper_test.yaml --name internal_paper_test
```

The label CSV must be a *test-only* file with these columns:

```text
image_id,image_path,group_id,object_id,source_label,x_min,y_min,x_max,y_max
```

`group_id` is the independent acquisition unit (for example well/day), and
each `object_id` is unique within its image. `image_path` is relative to
`dataset.images_root` unless absolute. Use `coordinate_format:
yolo_normalized` only when the CSV instead supplies `x_center,y_center,width,height`.

```yaml
schema_version: 1
dataset:
  images_root: D:\internal_experiment\images
  pixel_size_um: 0.325       # omit only if not known
labels:
  csv_path: D:\internal_experiment\paper_test_labels.csv
  coordinate_format: xyxy_px
  annotation_provenance: blinded expert review v1
  class_map:
    prophase: prophase
    earlyprometaphase: earlyprometaphase
    prometaphase: prometaphase
    metaphase: metaphase
    anaphase: anaphase
    telophase: telophase
model:
  weights_path: C:\Users\Brook\Documents\Work\CelFDrive\Models\yolo11x_p99p99_bg05\weights\best.pt
  sha256: B23834AD8276E8B33A70A4E178CA84A5D0260F5A45B3DF8C52FB3FF999BE7A0A
inference:
  confidence: 0.25
  iou: 0.7
  imgsz: 640
  device: 0
  threshold_selection_source: internal-validation-run-ID
split:
  name: test
  test_group_ids: [well_A_day_1, well_B_day_2]
  excluded_group_ids: [all_training_or_tuning_groups]
bootstrap_iterations: 1000
minimum_manual_review_labels: 50
review_seed: 42
```

The runner rejects labels outside `test_group_ids`, a test/training overlap,
unknown stage labels, duplicate object IDs, missing images, invalid boxes, a
checkpoint hash mismatch, and a checkpoint with a different class map. It
exports CSV and Parquet tables, confidence intervals, stage metrics, target
operating points, matched-target centre errors, a stage-stratified
`manual_review_queue`, overlays for that queue, and a run manifest.
The supplied YOLO11x checkpoint declares a three-channel input. The runner
therefore replicates the selected single chromatin image into all three model
channels and records this in `run_manifest.json`; it never combines markers.

The downloaded public sources are deliberately separate from the paper test:

```powershell
python -m benchmarks.run_benchmark inventory-ctc --images-root D:\CelFDriveBenchmark\raw\Fluo-N2DL-HeLa --output D:\CelFDriveBenchmark\derived\ctc_inventory.csv
python -m benchmarks.run_benchmark inventory-cellcognition --images-root D:\CelFDriveBenchmark\raw\H2b_aTub_MD20x_exp911_classifier --output D:\CelFDriveBenchmark\derived\cellcognition_inventory.csv
```

Those inventory commands do not represent the public data as six-stage ground
truth; they are provenance and pipeline checks only.
