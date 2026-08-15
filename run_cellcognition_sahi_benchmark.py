"""Evaluate CellCognition targets with SAHI tiled Ultralytics inference.

This separate external-data experiment retains the full-frame benchmark's
normalisation, released classifier labels, confidence threshold, and endpoint.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmarking import _sha256, create_run_directory, prepare_model_input, preprocess_image, write_quality_overlays
from run_cellcognition_target_benchmark import MODEL_PATH, PREDICTION_MAP, load_targets


def _prediction_table(targets, detection_model, slice_height, slice_width, overlap_ratio):
    """Run SAHI once per image and return de-duplicated tiled predictions."""
    import tifffile
    from sahi.predict import get_sliced_prediction

    rows = []
    for image_id, image_targets in targets.groupby("image_id", sort=True):
        image = prepare_model_input(preprocess_image(tifffile.imread(image_targets.resolved_image_path.iloc[0])), 3)
        result = get_sliced_prediction(image, detection_model=detection_model, slice_height=slice_height, slice_width=slice_width, overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio, perform_standard_pred=False, postprocess_type="GREEDYNMM", postprocess_match_metric="IOS", postprocess_match_threshold=.5, postprocess_class_agnostic=False, verbose=0)
        for prediction in result.object_prediction_list:
            box, class_name = prediction.bbox, prediction.category.name
            rows.append({"image_id": image_id, "group_id": image_targets.group_id.iloc[0], "x_min": box.minx, "y_min": box.miny, "x_max": box.maxx, "y_max": box.maxy, "centre_x": (box.minx + box.maxx) / 2, "centre_y": (box.miny + box.maxy) / 2, "confidence": prediction.score.value, "class_name": class_name, "coarse_class": PREDICTION_MAP[class_name]})
    return pd.DataFrame(rows, columns=("image_id", "group_id", "x_min", "y_min", "x_max", "y_max", "centre_x", "centre_y", "confidence", "class_name", "coarse_class"))


def _match_targets(targets, predictions):
    """Make a confidence-ordered one-to-one centre-in-box target match."""
    matches, used_prediction_indices = [], set()
    for _, target in targets.sort_values("image_id").iterrows():
        candidates = predictions[predictions.image_id == target.image_id].sort_values("confidence", ascending=False)
        candidate = next((row for index, row in candidates.iterrows() if index not in used_prediction_indices and row.x_min <= target.centre_x <= row.x_max and row.y_min <= target.centre_y <= row.y_max), None)
        if candidate is not None:
            used_prediction_indices.add(candidate.name)
        matches.append({"image_id": target.image_id, "group_id": target.group_id, "object_id": target.object_id, "source_label": target.source_label, "ground_truth_class": target.class_name, "prediction_class": candidate.coarse_class if candidate is not None else None, "confidence": candidate.confidence if candidate is not None else None, "centre_hit": candidate is not None})
    return pd.DataFrame(matches)


def run(images_root, analysis_root, output_root=r"D:\CelFDriveBenchmark\runs", confidence=.25, slice_height=640, slice_width=640, overlap_ratio=.25, device="cuda:0"):
    """Run frozen checkpoint tiled inference and save comparable external outputs."""
    import sahi
    import torch
    from sahi import AutoDetectionModel

    if slice_height <= 0 or slice_width <= 0 or not 0 <= overlap_ratio < 1:
        raise ValueError("Slice dimensions must be positive and overlap_ratio must be in [0, 1)")
    targets = load_targets(images_root, analysis_root)
    run_directory = create_run_directory(output_root, "cellcognition_h2b_targets_sahi")
    model_path = MODEL_PATH.resolve()
    try:
        detection_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(model_path), confidence_threshold=confidence, device=device, image_size=640)
        predictions = _prediction_table(targets, detection_model, slice_height, slice_width, overlap_ratio)
        matches = _match_targets(targets, predictions)
        detected = matches[matches.centre_hit]
        summary = pd.DataFrame([
            {"metric": "centre_in_box_recall", "value": matches.centre_hit.mean()},
            {"metric": "coarsened_stage_accuracy_among_detected", "value": (detected.ground_truth_class == detected.prediction_class).mean() if len(detected) else 0.0},
            {"metric": "classifier_labelled_targets", "value": len(matches)},
            {"metric": "unique_h2b_fields", "value": targets.image_id.nunique()},
            {"metric": "predictions", "value": len(predictions)},
        ])
        targets.to_csv(run_directory / "released_classifier_targets.csv", index=False)
        predictions.to_csv(run_directory / "predictions.csv", index=False)
        matches.to_csv(run_directory / "target_matches.csv", index=False)
        summary.to_csv(run_directory / "summary.csv", index=False)
        review_images = targets.groupby("class_name").image_id.first().tolist()[:50]
        write_quality_overlays(predictions, targets.assign(x_min=lambda x: x.centre_x - 1, y_min=lambda x: x.centre_y - 1, x_max=lambda x: x.centre_x + 1, y_max=lambda x: x.centre_y + 1), run_directory / "overlays", review_images)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "CellCognition H2B-mCherry chromatin + microtubules", "labels": "released CellCognition classifier event labels", "endpoint": "object centre inside predicted box", "not_measured": "manual-ground-truth AP or precision", "inference_method": "SAHI tiled Ultralytics inference", "sahi_version": sahi.__version__, "model_path": str(model_path), "model_sha256": _sha256(model_path), "confidence": confidence, "slice_height_px": slice_height, "slice_width_px": slice_width, "overlap_ratio": overlap_ratio, "sahi_postprocess": {"type": "GREEDYNMM", "metric": "IOS", "threshold": .5, "class_agnostic": False}, "cuda_available": torch.cuda.is_available(), "device": device}
        (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        (run_directory / "FAILED.txt").write_text("SAHI CellCognition benchmark did not complete; do not use this directory.", encoding="utf-8")
        raise
    return run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--slice-height", type=int, default=640)
    parser.add_argument("--slice-width", type=int, default=640)
    parser.add_argument("--overlap-ratio", type=float, default=.25)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(run(args.images_root, args.analysis_root, args.output_root, args.confidence, args.slice_height, args.slice_width, args.overlap_ratio, args.device))


if __name__ == "__main__":
    main()
