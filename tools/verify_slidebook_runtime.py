"""Verify the Python runtime used by the direct SlideBook callback.

The test imports the native libraries in the callback's order, then runs the
bundled detector on a tracked TIFF from the max-projection tutorial.  It is
intended for the acquisition computer before SlideBook is configured to use
the environment.
"""

from pathlib import Path
import importlib.util
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_IMAGE = (
    REPOSITORY_ROOT
    / "examples"
    / "max_projection_sahi"
    / "max_projections"
    / "p1_max.tif"
)
BRIDGE_PATH = REPOSITORY_ROOT / "SlideBook" / "find_locations_of_interest_montage.py"


def _load_slidebook_bridge():
    """Load the bridge exactly as SlideBook loads the copied script."""
    spec = importlib.util.spec_from_file_location("slidebook_runtime_bridge", BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SlideBook bridge: {BRIDGE_PATH}")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge


def load_tutorial_montage(tif_reader, numpy_module, bridge):
    """Load the tracked tutorial TIFF and apply the SlideBook axis conversion."""
    image = tif_reader(EXAMPLE_IMAGE)
    if image.ndim != 2:
        raise ValueError(f"Tutorial TIFF must be two-dimensional; received {image.shape}")
    montage = bridge._to_celfdrive_montage(image[numpy_module.newaxis, :, :])
    if montage.shape != image.shape + (1,):
        raise RuntimeError(f"SlideBook axis conversion failed: {montage.shape}")
    return image, montage


def main():
    """Import, run inference on the tutorial image, and report success."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    if not EXAMPLE_IMAGE.is_file():
        raise FileNotFoundError(f"Tutorial TIFF is missing: {EXAMPLE_IMAGE}")

    # These imports intentionally happen before the application import and
    # inference, so duplicate OpenMP DLLs fail here rather than in hardware run.
    import numpy as np
    import torch
    import cv2  # noqa: F401
    import ultralytics  # noqa: F401
    import sahi  # noqa: F401
    import tifffile
    import predict

    bridge = _load_slidebook_bridge()
    image, montage = load_tutorial_montage(tifffile.imread, np, bridge)

    config = predict.load_predict_config(REPOSITORY_ROOT / "celfdrive_predict.yaml")
    config["project"]["repo_path"] = str(REPOSITORY_ROOT)
    config["model"]["weights_path"] = str(
        REPOSITORY_ROOT / "Models" / "yolo11x_p99p99_bg05" / "weights" / "best.pt"
    )
    config["logging"]["enabled"] = False
    config["plotting"]["enabled"] = False
    predict.configure_prediction_runtime(config)
    detections = predict.process_image(
        image,
        class_info=predict.get_class_info(config["profile"]),
    )
    print(
        "SlideBook runtime verification passed: "
        f"image={EXAMPLE_IMAGE.name}, shape={montage.shape}, "
        f"detections={len(detections)}, torch={torch.__version__}"
    )


if __name__ == "__main__":
    main()
