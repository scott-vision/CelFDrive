"""Benchmark tiled YOLO inference against full released CellCognition tracks.

The track tables contain a released CellCognition segmentation box and
classifier label for every tracked object.  This is an external
classifier-to-classifier comparison, not a manual expert-ground-truth study.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .core import _sha256, box_iou, create_run_directory, prepare_model_input, preprocess_image, write_quality_overlays
from .run_cellcognition_target_benchmark import MODEL_PATH, PREDICTION_MAP, SOURCE_MAP


def batched_sahi_prediction_table(labels, detection_model, slice_height, slice_width, overlap_ratio, tile_batch_size, postprocess_metric="IOS", postprocess_threshold=.5):
    """Slice with SAHI, infer batches of tiles on GPU, then merge with SAHI NMM."""
    from sahi.postprocess.combine import GreedyNMMPostprocess
    from sahi.prediction import ObjectPrediction
    from sahi.slicing import slice_image

    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be positive")
    rows = []
    postprocess = GreedyNMMPostprocess(match_threshold=postprocess_threshold, match_metric=postprocess_metric, class_agnostic=False)
    for image_id, image_labels in labels.groupby("image_id", sort=True):
        image_path = Path(image_labels.resolved_image_path.iloc[0])
        if image_path.suffix.lower() == ".png":
            from PIL import Image

            image_array = np.asarray(Image.open(image_path))
        else:
            import tifffile

            image_array = tifffile.imread(image_path)
        image = prepare_model_input(preprocess_image(image_array), 3)
        slices = slice_image(image, slice_height=slice_height, slice_width=slice_width, overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio, auto_slice_resolution=False)
        objects = []
        for start in range(0, len(slices), tile_batch_size):
            batch = slices[start:start + tile_batch_size]
            # SAHI's Ultralytics adapter reverses RGB to BGR before inference.
            results = detection_model.model.predict([item["image"][:, :, ::-1] for item in batch], imgsz=640, conf=detection_model.confidence_threshold, device=detection_model.device, batch=tile_batch_size, verbose=False)
            for item, result in zip(batch, results):
                shift_x, shift_y = item["starting_pixel"]
                for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                    class_name = detection_model.category_mapping[str(class_id)]
                    # GreedyNMM expects full-image coordinates.  SAHI's stock
                    # loop calls get_shifted_object_prediction before merging;
                    # apply that translation explicitly for our batched path.
                    full_box = [box[0] + shift_x, box[1] + shift_y, box[2] + shift_x, box[3] + shift_y]
                    objects.append(ObjectPrediction(bbox=full_box, category_id=int(class_id), category_name=class_name, score=float(score), shift_amount=[0, 0], full_shape=[image.shape[0], image.shape[1]]))
        for prediction in postprocess(objects):
            box, class_name = prediction.bbox, prediction.category.name
            rows.append({"image_id": image_id, "group_id": image_labels.group_id.iloc[0], "x_min": box.minx, "y_min": box.miny, "x_max": box.maxx, "y_max": box.maxy, "centre_x": (box.minx + box.maxx) / 2, "centre_y": (box.miny + box.maxy) / 2, "confidence": prediction.score.value, "class_name": class_name, "coarse_class": PREDICTION_MAP[class_name]})
    return pd.DataFrame(rows, columns=("image_id", "group_id", "x_min", "y_min", "x_max", "y_max", "centre_x", "centre_y", "confidence", "class_name", "coarse_class"))


def load_full_track_labels(images_root, analysis_root):
    """Read mitotic track labels only for positions whose raw images are present."""
    required_columns = ["Frame", "ObjectID", "class__name", "tracking__center_x", "tracking__center_y", "tracking__upperleft_x", "tracking__upperleft_y", "tracking__lowerright_x", "tracking__lowerright_y"]
    rows, missing_images = [], []
    images_root = Path(images_root)
    available_positions = {path.name for path in images_root.iterdir() if path.is_dir()}
    track_paths = sorted(Path(analysis_root).rglob("*_tracking/_features_tracks/*Crfp__Rprimary.tsv.bz2"))
    labelled_positions = {path.parents[2].name for path in track_paths}
    selected_positions = available_positions & labelled_positions
    if not selected_positions:
        raise ValueError("No CellCognition image-position directories match released track-table positions")
    for track_path in track_paths:
        position = track_path.parents[2].name
        if position not in selected_positions:
            continue
        tracks = pd.read_csv(track_path, sep="\t", compression="bz2", usecols=required_columns)
        tracks = tracks[tracks["class__name"].isin(SOURCE_MAP)].copy()
        for _, track in tracks.iterrows():
            frame = int(track.Frame)
            image_path = images_root / position / f"tubulin_P{position}_T{frame:05}_Crfp_Z1_S1.tif"
            if not image_path.is_file():
                missing_images.append(str(image_path))
                continue
            x_min, y_min = float(track.tracking__upperleft_x), float(track.tracking__upperleft_y)
            x_max, y_max = float(track.tracking__lowerright_x), float(track.tracking__lowerright_y)
            if x_max <= x_min or y_max <= y_min:
                raise ValueError(f"Invalid CellCognition box for {position}, frame {frame}, object {track.ObjectID}")
            source_label = str(track["class__name"])
            rows.append({"image_id": f"{position}_t{frame:05}", "group_id": position, "object_id": f"{position}:{frame}:{int(track.ObjectID)}", "resolved_image_path": str(image_path.resolve()), "source_label": source_label, "class_name": SOURCE_MAP[source_label], "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "centre_x": float(track.tracking__center_x), "centre_y": float(track.tracking__center_y)})
    if missing_images:
        raise FileNotFoundError(f"{len(set(missing_images))} released track-label images are absent; first: {missing_images[0]}")
    labels = pd.DataFrame(rows).drop_duplicates("object_id")
    if labels.empty:
        raise ValueError("No mitotic labels found in released CellCognition track tables")
    return labels, sorted(selected_positions), sorted(labelled_positions - selected_positions)


def match_any_class(predictions, labels, iou_threshold=.5):
    """One-to-one IoU matching, independent of class, for detection and confusion."""
    records, used_labels = [], set()
    for prediction_index, prediction in predictions.sort_values("confidence", ascending=False).iterrows():
        candidates = []
        for label_index, label in labels[labels.image_id == prediction.image_id].iterrows():
            if label_index not in used_labels:
                iou = box_iou((prediction.x_min, prediction.y_min, prediction.x_max, prediction.y_max), (label.x_min, label.y_min, label.x_max, label.y_max))
                if iou >= iou_threshold:
                    candidates.append((iou, label_index))
        if candidates:
            iou, label_index = max(candidates)
            used_labels.add(label_index)
            records.append({"prediction_index": prediction_index, "label_index": label_index, "iou": iou})
    return pd.DataFrame(records, columns=("prediction_index", "label_index", "iou"))


def run(images_root, analysis_root, output_root=r"D:\CelFDriveBenchmark\runs", confidence=.25, slice_height=640, slice_width=640, overlap_ratio=.25, tile_batch_size=6, device="cuda:0"):
    """Run SAHI against every fully-labelled mitotic CellCognition track object."""
    import sahi
    import torch
    from sahi import AutoDetectionModel

    labels, selected_positions, excluded_labelled_positions = load_full_track_labels(images_root, analysis_root)
    run_directory = create_run_directory(output_root, "cellcognition_full_tracks_sahi")
    model_path = MODEL_PATH.resolve()
    try:
        model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(model_path), confidence_threshold=confidence, device=device, image_size=640)
        predictions = batched_sahi_prediction_table(labels, model, slice_height, slice_width, overlap_ratio, tile_batch_size)
        match_table = match_any_class(predictions, labels)
        matched_predictions = set(match_table.prediction_index)
        matched_labels = set(match_table.label_index)
        matches = labels.copy()
        matches["prediction_class"] = None
        matches["confidence"] = None
        matches["iou"] = None
        for match in match_table.itertuples(index=False):
            prediction = predictions.loc[match.prediction_index]
            matches.loc[match.label_index, ["prediction_class", "confidence", "iou"]] = [prediction.coarse_class, prediction.confidence, match.iou]
        matches["detected"] = matches.index.isin(matched_labels)
        background = "background / missed"
        classes = ["prophase", "prometaphase", "metaphase", "anaphase", "telophase"]
        matrix = pd.crosstab(matches.class_name, matches.prediction_class.fillna(background)).reindex(index=classes, columns=classes + [background], fill_value=0)
        summary = pd.DataFrame([
            {"metric": "detection_recall_iou50", "value": matches.detected.mean()},
            {"metric": "prediction_precision_iou50", "value": len(matched_predictions) / len(predictions) if len(predictions) else 0.0},
            {"metric": "coarsened_stage_accuracy_among_detected", "value": (matches.loc[matches.detected, "class_name"] == matches.loc[matches.detected, "prediction_class"]).mean() if matches.detected.any() else 0.0},
            {"metric": "released_classifier_mitotic_objects", "value": len(labels)},
            {"metric": "unique_h2b_fields", "value": labels.image_id.nunique()},
            {"metric": "predictions", "value": len(predictions)},
            {"metric": "false_positives_per_field", "value": (len(predictions) - len(matched_predictions)) / labels.image_id.nunique()},
        ])
        labels.to_csv(run_directory / "released_track_labels.csv", index=False)
        predictions.to_csv(run_directory / "predictions.csv", index=False)
        matches.to_csv(run_directory / "matches.csv", index=False)
        match_table.to_csv(run_directory / "iou_matches.csv", index=False)
        matrix.to_csv(run_directory / "confusion_matrix_with_background.csv")
        summary.to_csv(run_directory / "summary.csv", index=False)
        review_images = labels.groupby("class_name").image_id.first().tolist()
        write_quality_overlays(predictions, labels, run_directory / "overlays", review_images)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "CellCognition H2B-mCherry chromatin + microtubules", "labels": "full released CellCognition track-table segmentation boxes and classifier labels", "label_limit": "released classifier annotations, not manual expert ground truth", "selected_raw_image_positions": selected_positions, "excluded_labelled_positions_without_local_raw_images": excluded_labelled_positions, "endpoint": "class-agnostic IoU >= 0.50 for detection; matched objects used for stage agreement", "inference_method": "SAHI slicing and NMM with batched Ultralytics tile inference", "sahi_version": sahi.__version__, "model_sha256": _sha256(model_path), "confidence": confidence, "slice_height_px": slice_height, "slice_width_px": slice_width, "overlap_ratio": overlap_ratio, "tile_batch_size": tile_batch_size, "cuda_available": torch.cuda.is_available(), "device": device}
        (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        (run_directory / "FAILED.txt").write_text("Full-label SAHI benchmark did not complete; do not use this directory.", encoding="utf-8")
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
    parser.add_argument("--tile-batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(run(args.images_root, args.analysis_root, args.output_root, args.confidence, args.slice_height, args.slice_width, args.overlap_ratio, args.tile_batch_size, args.device))


if __name__ == "__main__":
    main()
