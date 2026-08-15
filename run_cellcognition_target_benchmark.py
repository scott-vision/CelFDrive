"""Evaluate frozen target finding on released CellCognition H2B-mCherry labels.

The CellCognition event tables provide per-object centres and classifier labels.
This script reports centre-in-box recall and coarsened stage agreement; the
labels are released classifier outputs, not manual ground truth.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarking import _sha256, create_run_directory, prepare_model_input, preprocess_image, write_quality_overlays

MODEL_PATH = Path("Models/yolo11x_p99p99_bg05/weights/best.pt")
SOURCE_MAP = {"pro": "prophase", "prometa": "prometaphase", "meta": "metaphase", "earlyana": "anaphase", "lateana": "anaphase", "telo": "telophase"}
PREDICTION_MAP = {"prophase": "prophase", "earlyprometaphase": "prometaphase", "prometaphase": "prometaphase", "metaphase": "metaphase", "anaphase": "anaphase", "telophase": "telophase"}


def load_targets(images_root, analysis_root):
    """Convert published event-table centre coordinates to H2B image targets."""
    images_root, analysis_root = Path(images_root), Path(analysis_root)
    rows = []
    for event_path in analysis_root.rglob("*Crfp__Rprimary.tsv"):
        position = event_path.parents[2].name
        table = pd.read_csv(event_path, sep="\t", usecols=lambda column: column in {"Frame", "objId__A", "objId__B", "class__A__name", "class__B__name", "tracking__A__center_x", "tracking__A__center_y", "tracking__B__center_x", "tracking__B__center_y"})
        for suffix in ("A", "B"):
            required = {"Frame", f"objId__{suffix}", f"class__{suffix}__name", f"tracking__{suffix}__center_x", f"tracking__{suffix}__center_y"}
            if not required <= set(table.columns):
                continue
            subset = table[list(required)].dropna()
            for _, row in subset.iterrows():
                source_label = str(row[f"class__{suffix}__name"])
                if source_label not in SOURCE_MAP:
                    continue
                frame = int(row.Frame)
                image_path = images_root / position / f"tubulin_P{position}_T{frame:05}_Crfp_Z1_S1.tif"
                if not image_path.is_file():
                    continue
                rows.append({"image_id": f"{position}_t{frame:05}", "group_id": position, "object_id": f"{event_path.stem}:{suffix}:{int(row[f'objId__{suffix}'])}", "resolved_image_path": str(image_path.resolve()), "source_label": source_label, "class_name": SOURCE_MAP[source_label], "centre_x": float(row[f"tracking__{suffix}__center_x"]), "centre_y": float(row[f"tracking__{suffix}__center_y"])})
    targets = pd.DataFrame(rows).drop_duplicates(["image_id", "object_id"])
    if targets.empty:
        raise ValueError("No CellCognition event targets could be linked to raw H2B images")
    return targets


def run(images_root, analysis_root, output_root=r"D:\CelFDriveBenchmark\runs", confidence=.25, iou=.7, device=0):
    """Run the fixed checkpoint once per unique H2B field and save external results."""
    import tifffile
    import torch
    from ultralytics import YOLO

    targets = load_targets(images_root, analysis_root)
    run_directory = create_run_directory(output_root, "cellcognition_h2b_targets")
    model_path = MODEL_PATH.resolve()
    try:
        model = YOLO(str(model_path))
        predictions = []
        for image_id, image_targets in targets.groupby("image_id", sort=True):
            image = preprocess_image(tifffile.imread(image_targets.resolved_image_path.iloc[0]))
            result = model.predict(prepare_model_input(image, 3), imgsz=640, conf=confidence, iou=iou, device=device, verbose=False)[0]
            if result.boxes is None:
                continue
            for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                predictions.append({"image_id": image_id, "group_id": image_targets.group_id.iloc[0], "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3], "centre_x": (box[0]+box[2])/2, "centre_y": (box[1]+box[3])/2, "confidence": score, "class_name": model.names[class_id], "coarse_class": PREDICTION_MAP[model.names[class_id]]})
        predictions = pd.DataFrame(predictions, columns=("image_id", "group_id", "x_min", "y_min", "x_max", "y_max", "centre_x", "centre_y", "confidence", "class_name", "coarse_class"))
        matches, used = [], set()
        for _, target in targets.sort_values("image_id").iterrows():
            candidates = predictions[predictions.image_id == target.image_id].sort_values("confidence", ascending=False)
            candidate = next((row for index, row in candidates.iterrows() if index not in used and row.x_min <= target.centre_x <= row.x_max and row.y_min <= target.centre_y <= row.y_max), None)
            if candidate is not None:
                used.add(candidate.name)
                matches.append({"image_id": target.image_id, "group_id": target.group_id, "object_id": target.object_id, "source_label": target.source_label, "ground_truth_class": target.class_name, "prediction_class": candidate.coarse_class, "confidence": candidate.confidence, "centre_hit": True})
            else:
                matches.append({"image_id": target.image_id, "group_id": target.group_id, "object_id": target.object_id, "source_label": target.source_label, "ground_truth_class": target.class_name, "prediction_class": None, "confidence": None, "centre_hit": False})
        matches = pd.DataFrame(matches)
        detected = matches[matches.centre_hit]
        stage_accuracy = (detected.ground_truth_class == detected.prediction_class).mean() if len(detected) else 0.0
        summary = pd.DataFrame([{"metric": "centre_in_box_recall", "value": matches.centre_hit.mean()}, {"metric": "coarsened_stage_accuracy_among_detected", "value": stage_accuracy}, {"metric": "classifier_labelled_targets", "value": len(matches)}, {"metric": "unique_h2b_fields", "value": targets.image_id.nunique()}, {"metric": "predictions", "value": len(predictions)}])
        targets.to_csv(run_directory / "released_classifier_targets.csv", index=False)
        predictions.to_csv(run_directory / "predictions.csv", index=False)
        matches.to_csv(run_directory / "target_matches.csv", index=False)
        summary.to_csv(run_directory / "summary.csv", index=False)
        review_images = targets.groupby("class_name").image_id.first().tolist()[:50]
        write_quality_overlays(predictions, targets.assign(x_min=lambda x: x.centre_x-1, y_min=lambda x: x.centre_y-1, x_max=lambda x: x.centre_x+1, y_max=lambda x: x.centre_y+1), run_directory / "overlays", review_images)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "CellCognition H2B-mCherry chromatin + microtubules", "labels": "released CellCognition classifier event labels", "endpoint": "object centre inside predicted box", "not_measured": "manual-ground-truth AP or precision", "model_path": str(model_path), "model_sha256": _sha256(model_path), "confidence": confidence, "iou": iou, "cuda_available": torch.cuda.is_available()}
        (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        (run_directory / "FAILED.txt").write_text("CellCognition benchmark did not complete; do not use this directory.", encoding="utf-8")
        raise
    return run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.7)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()
    print(run(args.images_root, args.analysis_root, args.output_root, args.confidence, args.iou, args.device))


if __name__ == "__main__":
    main()
