"""Evaluate frozen CelFDrive target finding on CTC lineage-defined divisions.

This external, zero-shot analysis uses Fluo-N2DL-HeLa's public tracking masks
and lineage table. A positive is the parent nucleus in its final annotated
frame before a recorded division. It is not a mitotic-stage classification
benchmark: CTC does not publish stage labels.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarking import (
    _sha256,
    box_iou,
    create_run_directory,
    prepare_model_input,
    preprocess_image,
    target_centre_errors,
    write_quality_overlays,
)


MODEL_PATH = Path("Models/yolo11x_p99p99_bg05/weights/best.pt")


def lineage_division_targets(dataset_root):
    """Return final parent-frame boxes for every CTC lineage division target."""
    import tifffile

    rows = []
    for sequence_dir in sorted(Path(dataset_root).glob("[0-9][0-9]")):
        track_file = sequence_dir.parent / f"{sequence_dir.name}_GT" / "TRA" / "man_track.txt"
        if not track_file.is_file():
            continue
        tracks = pd.read_csv(track_file, sep=" ", header=None, names=("track_id", "start_frame", "end_frame", "parent_id"))
        parent_ids = set(tracks.loc[tracks.parent_id != 0, "parent_id"])
        for parent_id in sorted(parent_ids):
            parent = tracks.loc[tracks.track_id == parent_id]
            if len(parent) != 1:
                raise ValueError(f"CTC sequence {sequence_dir.name} has ambiguous parent track {parent_id}")
            frame = int(parent.end_frame.iloc[0])
            image_path = sequence_dir / f"t{frame:03}.tif"
            mask_path = sequence_dir.parent / f"{sequence_dir.name}_GT" / "TRA" / f"man_track{frame:03}.tif"
            mask = tifffile.imread(mask_path)
            ys, xs = np.where(mask == parent_id)
            if not len(xs):
                raise ValueError(f"CTC parent {parent_id} is absent from {mask_path}")
            rows.append({"image_id": f"{sequence_dir.name}_t{frame:03}_parent{parent_id}", "group_id": sequence_dir.name,
                         "object_id": str(parent_id), "resolved_image_path": str(image_path.resolve()),
                         "class_name": "lineage_division_target", "x_min": float(xs.min()), "y_min": float(ys.min()),
                         "x_max": float(xs.max() + 1), "y_max": float(ys.max() + 1),
                         "centre_x": float((xs.min() + xs.max() + 1) / 2), "centre_y": float((ys.min() + ys.max() + 1) / 2),
                         "frame": frame})
    if not rows:
        raise ValueError("No CTC lineage division targets were found")
    return pd.DataFrame(rows)


def run(dataset_root, output_root=r"D:\CelFDriveBenchmark\runs", confidence=.25, iou=.7, device=0):
    """Run the frozen checkpoint and save external target-finding results."""
    import tifffile
    import torch
    from ultralytics import YOLO

    targets = lineage_division_targets(dataset_root)
    run_directory = create_run_directory(output_root, "ctc_lineage_division")
    model_path = MODEL_PATH.resolve()
    try:
        model = YOLO(str(model_path))
        if int(model.model.yaml.get("ch", 3)) != 3:
            raise ValueError("This CTC runner expects the supplied 3-channel YOLO11x checkpoint")
        prediction_rows = []
        for _, target in targets.iterrows():
            image = preprocess_image(tifffile.imread(target.resolved_image_path))
            result = model.predict(prepare_model_input(image, 3), imgsz=640, conf=confidence, iou=iou, device=device, verbose=False)[0]
            if result.boxes is None:
                continue
            for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                prediction_rows.append({"image_id": target.image_id, "group_id": target.group_id, "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3], "centre_x": (box[0] + box[2]) / 2, "centre_y": (box[1] + box[3]) / 2, "confidence": score, "class_id": class_id, "class_name": model.names[class_id]})
        predictions = pd.DataFrame(prediction_rows, columns=("image_id", "group_id", "x_min", "y_min", "x_max", "y_max", "centre_x", "centre_y", "confidence", "class_id", "class_name"))
        matches = []
        for _, target in targets.iterrows():
            candidates = predictions[predictions.image_id == target.image_id]
            best = max((box_iou((prediction.x_min, prediction.y_min, prediction.x_max, prediction.y_max), (target.x_min, target.y_min, target.x_max, target.y_max)) for _, prediction in candidates.iterrows()), default=0.0)
            matches.append({"image_id": target.image_id, "group_id": target.group_id, "object_id": target.object_id, "best_iou": best, "detected_iou50": best >= .5})
        matches = pd.DataFrame(matches)
        true_positives = int(matches.detected_iou50.sum())
        false_positives = len(predictions) - true_positives
        summary = pd.DataFrame([{"metric": "lineage_parent_recall_iou50", "value": true_positives / len(targets)}, {"metric": "precision_iou50", "value": true_positives / len(predictions) if len(predictions) else 0.0}, {"metric": "false_positives_per_field", "value": false_positives / len(targets)}, {"metric": "division_targets", "value": len(targets)}, {"metric": "predictions", "value": len(predictions)}])
        targets.to_csv(run_directory / "ground_truth_lineage_division_targets.csv", index=False)
        predictions.to_csv(run_directory / "predictions.csv", index=False)
        matches.to_csv(run_directory / "target_matches.csv", index=False)
        summary.to_csv(run_directory / "summary.csv", index=False)
        target_centre_errors(predictions, targets).to_csv(run_directory / "target_centre_errors.csv", index=False)
        write_quality_overlays(predictions, targets, run_directory / "overlays", targets.image_id)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "Cell Tracking Challenge Fluo-N2DL-HeLa training", "endpoint": "lineage parent at final pre-division frame", "not_measured": "mitotic stage classification", "model_path": str(model_path), "model_sha256": _sha256(model_path), "confidence": confidence, "iou": iou, "device": device, "cuda_available": torch.cuda.is_available()}
        (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        (run_directory / "FAILED.txt").write_text("CTC benchmark did not complete; do not use this directory.", encoding="utf-8")
        raise
    return run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.7)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()
    print(run(args.dataset_root, args.output_root, args.confidence, args.iou, args.device))


if __name__ == "__main__":
    main()
