# Synthetic smoke-test fixture

`synthetic_blank_image.csv` is a 32 × 32, single-channel uint8-equivalent blank image. It contains no biological image data or personal information and may be redistributed with this repository. `annotations.json` records its intentionally empty annotation set and pixel-coordinate convention.

The bundled model should return no detections for this fixture. `expected_detections.json` records that expected result and a coordinate tolerance for future non-empty fixtures.

This fixture verifies installation and the prediction path only. It is **not** evidence of model accuracy, biological performance, or generalisation. Replace it with an approved, anonymised annotated microscopy subset before using the tutorial as a reviewer-facing data demonstration.
