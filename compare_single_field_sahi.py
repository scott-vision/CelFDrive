"""Render full-frame and low-threshold SAHI predictions for one microscopy field."""
import argparse
import json
from pathlib import Path

import pandas as pd

from benchmarking import _sha256, create_run_directory, prepare_model_input, preprocess_image
from run_cellcognition_full_label_sahi_benchmark import batched_sahi_prediction_table
from run_cellcognition_target_benchmark import MODEL_PATH, PREDICTION_MAP


def _draw(axis, image, predictions, title):
    import matplotlib.pyplot as plt

    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    for _, row in predictions.iterrows():
        axis.add_patch(plt.Rectangle((row.x_min, row.y_min), row.x_max - row.x_min, row.y_max - row.y_min, fill=False, edgecolor="white", linewidth=1.6))
        axis.text(row.x_min, row.y_min, f"{row.class_name} {row.confidence:.2f}", color="white", fontsize=8, va="bottom", bbox={"facecolor": "black", "alpha": .75, "pad": 1.5, "edgecolor": "none"})
    axis.set_title(f"{title} ({len(predictions)} boxes)")
    axis.axis("off")


def run(image_path, output_root=r"D:\CelFDriveBenchmark\runs", confidence=.25, merge_iou=.1, device="cuda:0"):
    """Run normal and SAHI inference on one raw H2B field and save a comparison."""
    import matplotlib.pyplot as plt
    import tifffile
    from sahi import AutoDetectionModel
    from ultralytics import YOLO

    image_path = Path(image_path).resolve()
    raw = tifffile.imread(image_path)
    image = preprocess_image(raw)
    model_path = MODEL_PATH.resolve()
    run_directory = create_run_directory(output_root, "single_field_sahi_comparison")
    model = YOLO(str(model_path))
    normal_result = model.predict(prepare_model_input(image, 3), imgsz=640, conf=confidence, device=device, verbose=False)[0]
    normal_rows = []
    for box, score, class_id in zip(normal_result.boxes.xyxy.cpu().numpy(), normal_result.boxes.conf.cpu().numpy(), normal_result.boxes.cls.cpu().numpy().astype(int)):
        class_name = model.names[class_id]
        normal_rows.append({"image_id": image_path.stem, "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3], "confidence": score, "class_name": class_name, "coarse_class": PREDICTION_MAP[class_name]})
    normal = pd.DataFrame(normal_rows)
    detection_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(model_path), confidence_threshold=confidence, device=device, image_size=640)
    source = pd.DataFrame([{"image_id": image_path.stem, "group_id": "0037", "resolved_image_path": str(image_path)}])
    sahi = batched_sahi_prediction_table(source, detection_model, 640, 640, .25, 6, postprocess_metric="IOU", postprocess_threshold=merge_iou)
    normal.to_csv(run_directory / "normal_predictions.csv", index=False)
    sahi.to_csv(run_directory / "sahi_low_merge_predictions.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    _draw(axes[0], image, normal, "Normal full-frame YOLO")
    _draw(axes[1], image, sahi, f"SAHI: class-aware merge IoU {merge_iou:.2f}")
    figure.savefig(run_directory / "normal_vs_sahi_low_merge.png", dpi=180)
    plt.close(figure)
    (run_directory / "run_manifest.json").write_text(json.dumps({"image": str(image_path), "model_sha256": _sha256(model_path), "confidence": confidence, "device": device, "sahi_slice_px": 640, "sahi_overlap_ratio": .25, "sahi_merge_metric": "IOU", "sahi_merge_threshold": merge_iou, "sahi_merge_class_agnostic": False}, indent=2), encoding="utf-8")
    return run_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--merge-iou", type=float, default=.1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(run(args.image, args.output_root, args.confidence, args.merge_iou, args.device))
