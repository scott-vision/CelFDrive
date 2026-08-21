"""Evaluate frozen YOLO target finding against CellClicker track annotations."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .core import _sha256, box_iou, create_run_directory, prepare_model_input, preprocess_image
from .run_cellcognition_target_benchmark import MODEL_PATH


def load_cellclicker_labels(project_directory, raw_images_directory):
    """Convert CellClicker's backward time-series XML labels to pixel boxes."""
    from CellClicker.manageXML import cell_xml_to_dataframe_absfilenames

    from CellClicker.project_paths import resolve_cell_regions_xml

    project_directory, raw_images_directory = Path(project_directory), Path(raw_images_directory)
    xml_path = resolve_cell_regions_xml(project_directory).path
    labels = cell_xml_to_dataframe_absfilenames(str(xml_path))
    if labels.empty:
        raise ValueError(f"No CellClicker labels found in {xml_path}")
    rows = []
    for row in labels.itertuples(index=False):
        image_path = Path(row.PathName)
        match = re.fullmatch(r"P(?P<position>\d+)_t(?P<frame>\d+)\.png", image_path.name)
        if match is None:
            raise ValueError(f"Unexpected CellClicker image name: {image_path.name}")
        position, frame = match.group("position"), int(match.group("frame"))
        raw_path = raw_images_directory / position / f"tubulin_P{position}_T{frame:05}_Crfp_Z1_S1.tif"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        rows.append({"image_id": f"{position}_t{frame:05}", "group_id": position, "track_id": f"{image_path.stem}:series_{row.SeriesID}", "source_step_index": int(row.ClassID), "display_image_path": str(image_path), "resolved_image_path": str(raw_path), "x_center": float(row.XCenter), "y_center": float(row.YCenter), "width": float(row.Width), "height": float(row.Height)})
    labels = pd.DataFrame(rows)
    labels[["x_min", "x_max"]] = pd.DataFrame({"x_min": (labels.x_center - labels.width / 2) * 1392, "x_max": (labels.x_center + labels.width / 2) * 1392})
    labels[["y_min", "y_max"]] = pd.DataFrame({"y_min": (labels.y_center - labels.height / 2) * 1040, "y_max": (labels.y_center + labels.height / 2) * 1040})
    labels["centre_x"], labels["centre_y"] = labels.x_center * 1392, labels.y_center * 1040
    return labels


def _match(predictions, labels):
    """Greedily match predictions to manual target boxes, independent of class."""
    records, used = [], set()
    for prediction_index, prediction in predictions.sort_values("confidence", ascending=False).iterrows():
        candidates = []
        for label_index, label in labels[labels.image_id == prediction.image_id].iterrows():
            if label_index not in used:
                iou = box_iou((prediction.x_min, prediction.y_min, prediction.x_max, prediction.y_max), (label.x_min, label.y_min, label.x_max, label.y_max))
                if iou >= .5:
                    candidates.append((iou, label_index))
        if candidates:
            iou, label_index = max(candidates)
            used.add(label_index)
            records.append({"prediction_index": prediction_index, "label_index": label_index, "iou": iou})
    return pd.DataFrame(records, columns=("prediction_index", "label_index", "iou"))


def run(project_directory, raw_images_directory, output_root=r"D:\CelFDriveBenchmark\runs", confidence=.25, batch_size=8, device="cuda:0"):
    """Run GPU-batched full-frame inference and save class-agnostic target metrics."""
    import tifffile
    import torch
    from ultralytics import YOLO

    labels = load_cellclicker_labels(project_directory, raw_images_directory)
    run_directory = create_run_directory(output_root, "cellclicker_p0037_target_evaluation")
    model_path = MODEL_PATH.resolve()
    try:
        model, rows = YOLO(str(model_path)), []
        fields = list(labels.groupby("image_id", sort=True))
        for start in range(0, len(fields), batch_size):
            batch = fields[start:start + batch_size]
            images = [prepare_model_input(preprocess_image(tifffile.imread(group.resolved_image_path.iloc[0])), 3) for _, group in batch]
            for (image_id, group), result in zip(batch, model.predict(images, imgsz=640, conf=confidence, device=device, batch=batch_size, verbose=False)):
                for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                    rows.append({"image_id": image_id, "group_id": group.group_id.iloc[0], "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3], "centre_x": (box[0] + box[2]) / 2, "centre_y": (box[1] + box[3]) / 2, "confidence": score, "class_name": model.names[class_id]})
        predictions = pd.DataFrame(rows, columns=("image_id", "group_id", "x_min", "y_min", "x_max", "y_max", "centre_x", "centre_y", "confidence", "class_name"))
        matches = _match(predictions, labels)
        labels["matched_prediction_class"], labels["matched_confidence"], labels["matched_iou"] = None, None, None
        for match in matches.itertuples(index=False):
            prediction = predictions.loc[match.prediction_index]
            labels.loc[match.label_index, ["matched_prediction_class", "matched_confidence", "matched_iou"]] = [prediction.class_name, prediction.confidence, match.iou]
        labels["detected_iou50"] = labels.index.isin(set(matches.label_index))
        centre_hits = []
        for _, label in labels.iterrows():
            candidates = predictions[predictions.image_id == label.image_id]
            centre_hits.append(any(row.x_min <= label.centre_x <= row.x_max and row.y_min <= label.centre_y <= row.y_max for _, row in candidates.iterrows()))
        labels["centre_hit"] = centre_hits
        summary = pd.DataFrame([{"metric": "manual_target_recall_iou50", "value": labels.detected_iou50.mean()}, {"metric": "manual_target_centre_in_box_recall", "value": labels.centre_hit.mean()}, {"metric": "manual_target_boxes", "value": len(labels)}, {"metric": "labelled_timepoints", "value": labels.image_id.nunique()}, {"metric": "predictions", "value": len(predictions)}])
        labels.to_csv(run_directory / "cellclicker_manual_targets.csv", index=False)
        predictions.to_csv(run_directory / "predictions.csv", index=False)
        matches.to_csv(run_directory / "iou_matches.csv", index=False)
        summary.to_csv(run_directory / "summary.csv", index=False)
        manifest = {"project": str(Path(project_directory).resolve()), "labels": "CellClicker manually reviewed track boxes; class_id is temporal step index, not stage", "endpoint": "class-agnostic target detection", "model_sha256": _sha256(model_path), "confidence": confidence, "batch_size": batch_size, "device": device, "cuda_available": torch.cuda.is_available()}
        (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        (run_directory / "FAILED.txt").write_text("CellClicker target evaluation did not complete; do not use this directory.", encoding="utf-8")
        raise
    return run_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-directory", required=True)
    parser.add_argument("--raw-images-directory", required=True)
    parser.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(run(args.project_directory, args.raw_images_directory, args.output_root, args.confidence, args.batch_size, args.device))
