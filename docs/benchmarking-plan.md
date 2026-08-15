# Fluorescence-microscopy mitosis benchmark plan

This plan evaluates a fixed CelFDrive weight file as a detector of mitotic
chromatin objects.  It does not treat the bundled synthetic image as
biological evidence.  Keep raw external data outside the Git repository and
record every derived file needed to reproduce a result.

## Scope and current model contract

The current prediction configuration accepts one two-dimensional intensity
image per field of view.  For multi-channel sources, explicitly select the
chromatin channel, then apply the configured percentile clipping and min--max
normalization.  Do not pass a colour composite or an unspecified first channel.

The model classes are:

| ID | CelFDrive label |
| --- | --- |
| 0 | prophase |
| 1 | earlyprometaphase |
| 2 | prometaphase |
| 3 | metaphase |
| 4 | anaphase |
| 5 | telophase |

The first release is an *external, zero-shot* benchmark: weights are frozen,
and no source image, frame, track, or well from a benchmark set may be used
for fine-tuning or threshold selection.  A later, separately reported,
fine-tuned experiment may use a training split, but must retain a held-out
source/sequence test set.

## Candidate datasets

| Priority | Dataset | Why it belongs in this benchmark | What it can validly measure |
| --- | --- | --- | --- |
| 1 | CellCognition chromatin + microtubules | Raw widefield HeLa Kyoto H2B-mCherry images (20x), tracking/event analysis, and annotated H2B classifier data are published together. | Whole-field detection and phase classification once the raw-image/object/event correspondence is verified. |
| 1 | LiveCellMiner / RNN OSF data | H2B-mCherry HeLa cell crops; expert-corrected frame labels include interphase and five mitotic stages. | A clean single-cell, six-stage classification sanity check.  Not whole-field detection, because the images are pre-cropped. |
| 2 | MitoPhase | Very close biology: 54 H2B-mCherry HeLa 3-D time-lapse sequences, manually annotated mitotic tracks, five phases. | Ideal external whole-field stage benchmark *if raw images and ground truth can be obtained from the authors*.  The public web page currently describes the data but does not expose a download link. |
| 2 | Cell Tracking Challenge Fluo-N2DL-HeLa | Public H2B-GFP whole-field time-lapse data with segmentation/tracking reference annotations. | Cross-marker nucleus detection and division-event recall; not six-stage phase accuracy because CTC does not supply phase labels. |
| exploratory | Cell Tracking Challenge Fluo-N2DH-SIM+ | Synthetic Hoechst-stained nuclei with perfect masks. | Pipeline and box-from-mask validation, not biological generalisation. |

Do not call coarse labels such as "mitotic", "interphase/mitosis/post-mitosis",
or a division link a six-stage annotation.  They support only a coarsened or
event-level analysis.  DAPI can be added as a named domain in the same layout,
but only after selecting a source that supplies object-level mitotic labels;
generic DAPI nuclear segmentation is not a mitotic-stage benchmark.

## Storage layout on D:

Create one durable benchmark root, separate from the checkout and model
weights:

```text
D:\CelFDriveBenchmark\
  archives\                 # immutable downloaded ZIP/TAR files
  raw\<dataset_id>\         # extraction exactly as distributed
  derived\<dataset_id>\     # projections, selected channels, converted labels
  manifests\                # source metadata, checksums, mappings, inventories
  runs\<run_id>\            # immutable prediction/evaluation outputs
```

The D: drive currently has about 648 GiB free, enough for the proposed staged
downloads.  Start with the CTC archive (182 MB), then CellCognition's reduced
raw image archive (about 900 MB) and analysis archive (about 445 MB).  Do not
download the much larger optional visualisation archive unless its metadata is
shown to be required for label alignment.

For every archive, create a manifest entry before extraction containing the
dataset ID, landing-page URL, direct URL, licence/citation, retrieval date,
archive byte count, SHA-256 calculated locally, and extractor command.  Raw
files and manifests are read-only inputs; generated images and labels belong
only below `derived/`.

## Label alignment protocol

1. Inventory the downloaded files before conversion: image dimensions, dtype,
   channel names/order, time/frame identifiers, segmentation/object IDs,
   track IDs, source phase labels, and coordinate origin.  Fail conversion if
   an image/object/track relation is ambiguous.
2. Create a versioned `label_mapping.yaml` per source.  It records the source
   label text, its intended meaning, CelFDrive evaluation label, the source of
   each bounding box (published box, contour/mask, or full crop), and excluded
   labels.  Preserve source labels in the converted table.
3. For sources with the conventional five stages, evaluate the pre-registered
   five-class view: `prophase -> prophase`,
   `{earlyprometaphase, prometaphase} -> prometaphase`,
   `metaphase -> metaphase`, `anaphase -> anaphase`, and
   `telophase -> telophase`.  Before matching, apply cross-class NMS to the
   two predicted prometaphase subclasses so duplicate boxes cannot inflate
   counts.  Report the original six-class predictions as well, but do not
   fabricate a ground-truth early-prometaphase label.
4. Convert masks/contours to one axis-aligned box per annotated object using
   `x_min, y_min, x_max, y_max` in the source pixel frame.  Record the exact
   source mask ID and conversion version.  Never derive a box from a predicted
   segmentation.
5. Overlay a stratified random sample of at least 50 converted labels per
   source (or all labels when fewer) on the selected-channel images.  Review
   boundary objects, cropped objects, and division frames; store the review
   CSV and image paths.  Resolve errors in the converter, not by silently
   editing result files.

## Evaluation outputs

Run the frozen weight file on exactly the selected channel/projection and save
one row per prediction and ground-truth object.  Use deterministic file names,
model SHA-256, the complete prediction config, package versions, and the Git
commit in `runs/<run_id>/run_manifest.json`.

Report these views separately:

* Detection: per-class and overall AP50, AP50:95, precision, recall, and F1;
  one-to-one IoU matching at 0.50, plus false positives per field of view.
* Classification: confusion matrix and macro-F1 among ground-truth objects
  matched at IoU >= 0.50.  Unmatched predictions/objects stay in the detection
  report rather than disappearing from the classification denominator.
* Temporal behaviour where tracks are available: division-event recall,
  fraction of mitotic tracks detected at least once, and phase-transition
  timing error in frames/minutes.  Split by sequence/well/track, never by
  individual frames, to avoid temporal leakage.
* Domain breakdown: H2B-mCherry, H2B-GFP, DAPI/Hoechst (if added), microscope,
  pixel size, and treatment.  These are generalisation strata, not pooled as
  interchangeable replicates.

## First implementation sequence

1. Add a small downloader that uses a checked-in dataset registry but writes
   only to `D:\CelFDriveBenchmark`; it will resume downloads and write the
   archive manifest.
2. Implement one source adapter at a time, beginning with CTC
   Fluo-N2DL-HeLa to validate TIFF discovery, masks-to-boxes, and evaluation
   mechanics.  Its result is detection/event-only.
3. Implement the CellCognition adapter and verify that its analysis objects
   map exactly to raw H2B frames.  This is the first intended whole-field
   stage result.
4. Add the LiveCellMiner crop adapter for the six-stage classification check.
5. Add MitoPhase only after obtaining the raw data and machine-readable ground
   truth under a documented licence.  Do not scrape visualisation pages as a
   substitute for the data release.

Each adapter should have focused tests covering coordinate conversion, channel
selection, class mapping, an excluded coarse label, and a missing image/object
link.  No external download is required for unit tests.

## Sources

* CellCognition publishes the H2B-mCherry + alpha-tubulin raw and analysis
  downloads, acquisition details, and H2B classifier data:
  <https://cellcognition-project.org/demo_data.html>
* The LiveCellMiner/RNN paper links the underlying OSF data and describes the
  H2B-mCherry, expert-corrected stage annotations:
  <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0297356>
* MitoPhase data description and phase definitions:
  <https://www.robots.ox.ac.uk/~vgg/research/mitosis/data.html>
* Cell Tracking Challenge download URLs and H2B-GFP/Hoechst dataset metadata:
  <https://celltrackingchallenge.net/2d-datasets/>
