"""Run the bundled model against the synthetic installation smoke-test fixture."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import predict


SAMPLE_DIRECTORY = REPOSITORY_ROOT / "sample_data"


def serialise_detections(detections):
    """Convert prediction results to JSON-compatible values."""
    return [
        {
            "class_id": int(class_id),
            "x": float(x),
            "y": float(y),
            "width": float(width),
            "height": float(height),
            "confidence": float(confidence),
        }
        for class_id, x, y, width, height, confidence in detections
    ]


def matches_expected(detections, expected):
    """Compare detections with the versioned fixture result."""
    expected_detections = expected["detections"]
    tolerance = float(expected["coordinate_tolerance_px"])
    if len(detections) != len(expected_detections):
        return False

    for observed, target in zip(serialise_detections(detections), expected_detections):
        if observed["class_id"] != target["class_id"]:
            return False
        for field in ["x", "y", "width", "height", "confidence"]:
            if abs(observed[field] - float(target[field])) > tolerance:
                return False
    return True


def main():
    """Load the smoke fixture, run inference, and verify its expected output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "celfdrive_predict.yaml",
        help="Prediction configuration to use.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=SAMPLE_DIRECTORY / "synthetic_blank_image.csv",
        help="Single-channel CSV image to process.",
    )
    args = parser.parse_args()

    config = predict.load_predict_config(args.config)
    config["project"]["repo_path"] = str(REPOSITORY_ROOT)
    config["model"]["weights_path"] = str(
        REPOSITORY_ROOT / "Models" / "yolo11x_p99p99_bg05" / "weights" / "best.pt"
    )
    config["logging"]["enabled"] = False
    config["plotting"]["enabled"] = False
    predict.configure_prediction_runtime(config)

    image = np.loadtxt(args.image, delimiter=",", dtype=np.uint8)
    with open(SAMPLE_DIRECTORY / "expected_detections.json", encoding="utf-8") as expected_file:
        expected = json.load(expected_file)

    class_info = predict.get_class_info(config["profile"])
    detections = predict.process_image(image, class_info=class_info)
    print(json.dumps(serialise_detections(detections), indent=2))

    if not matches_expected(detections, expected):
        raise SystemExit("Sample output does not match sample_data/expected_detections.json")


if __name__ == "__main__":
    main()
