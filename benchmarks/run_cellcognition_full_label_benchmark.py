"""Full-frame Ultralytics benchmark against released CellCognition tracks."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .core import _sha256, create_run_directory, prepare_model_input, preprocess_image, write_quality_overlays
from .run_cellcognition_full_label_sahi_benchmark import load_full_track_labels, match_any_class
from .run_cellcognition_target_benchmark import MODEL_PATH, PREDICTION_MAP


def full_frame_predictions(labels, model, confidence, batch_size, device):
    """Run same-sized normalised fields in GPU batches without SAHI slicing."""
    import tifffile

    rows = []
    fields = list(labels.groupby("image_id", sort=True))
    for start in range(0, len(fields), batch_size):
        batch = fields[start:start + batch_size]
        images = [prepare_model_input(preprocess_image(tifffile.imread(group.resolved_image_path.iloc[0])), 3) for _, group in batch]
        results = model.predict(images, imgsz=640, conf=confidence, device=device, batch=batch_size, verbose=False)
        for (image_id, group), result in zip(batch, results):
            if result.boxes is None:
                continue
            for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                class_name = model.names[class_id]
                rows.append({"image_id": image_id, "group_id": group.group_id.iloc[0], "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3], "centre_x": (box[0] + box[2]) / 2, "centre_y": (box[1] + box[3]) / 2, "confidence": score, "class_name": class_name, "coarse_class": PREDICTION_MAP[class_name]})
    return pd.DataFrame(rows, columns=("image_id", "group_id", "x_min", "y_min", "x_max", "y_max", "centre_x", "centre_y", "confidence", "class_name", "coarse_class"))


def run(images_root, analysis_root, output_root=r"D:\CelFDriveBenchmark\runs", confidence=.25, batch_size=8, device="cuda:0"):
    """Evaluate full-frame GPU-batched predictions on full released labels."""
    import torch
    from ultralytics import YOLO

    labels, selected_positions, excluded_positions = load_full_track_labels(images_root, analysis_root)
    run_directory = create_run_directory(output_root, "cellcognition_full_tracks_fullframe")
    model_path = MODEL_PATH.resolve()
    try:
        model = YOLO(str(model_path))
        predictions = full_frame_predictions(labels, model, confidence, batch_size, device)
        iou_matches = match_any_class(predictions, labels)
        matches = labels.copy()
        matches["prediction_class"], matches["confidence"], matches["iou"] = None, None, None
        for match in iou_matches.itertuples(index=False):
            prediction = predictions.loc[match.prediction_index]
            matches.loc[match.label_index, ["prediction_class", "confidence", "iou"]] = [prediction.coarse_class, prediction.confidence, match.iou]
        matches["detected"] = matches.index.isin(set(iou_matches.label_index))
        summary = pd.DataFrame([{"metric": "detection_recall_iou50", "value": matches.detected.mean()}, {"metric": "prediction_precision_iou50", "value": len(iou_matches) / len(predictions) if len(predictions) else 0.0}, {"metric": "coarsened_stage_accuracy_among_detected", "value": (matches.loc[matches.detected, "class_name"] == matches.loc[matches.detected, "prediction_class"]).mean() if matches.detected.any() else 0.0}, {"metric": "released_classifier_mitotic_objects", "value": len(labels)}, {"metric": "unique_h2b_fields", "value": labels.image_id.nunique()}, {"metric": "predictions", "value": len(predictions)}])
        labels.to_csv(run_directory / "released_track_labels.csv", index=False)
        predictions.to_csv(run_directory / "predictions.csv", index=False)
        matches.to_csv(run_directory / "matches.csv", index=False)
        iou_matches.to_csv(run_directory / "iou_matches.csv", index=False)
        summary.to_csv(run_directory / "summary.csv", index=False)
        write_quality_overlays(predictions, labels, run_directory / "overlays", labels.groupby("class_name").image_id.first().tolist())
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "CellCognition H2B-mCherry chromatin + microtubules", "labels": "full released CellCognition track-table segmentation boxes and classifier labels", "label_limit": "released classifier annotations, not manual expert ground truth", "selected_raw_image_positions": selected_positions, "excluded_labelled_positions_without_local_raw_images": excluded_positions, "inference_method": "full-frame batched Ultralytics inference", "model_sha256": _sha256(model_path), "confidence": confidence, "field_batch_size": batch_size, "cuda_available": torch.cuda.is_available(), "device": device}
        (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        (run_directory / "FAILED.txt").write_text("Full-frame full-label benchmark did not complete; do not use this directory.", encoding="utf-8")
        raise
    return run_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(run(args.images_root, args.analysis_root, args.output_root, args.confidence, args.batch_size, args.device))
