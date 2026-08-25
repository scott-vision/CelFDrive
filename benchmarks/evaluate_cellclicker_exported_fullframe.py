"""Evaluate full-frame YOLO inference against CellClicker exported stage labels."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .core import _sha256, create_run_directory, detection_metrics, prepare_model_input, preprocess_image
from .evaluate_cellclicker_exported_sahi import (
    CLASSES,
    full_confusion,
    load_exported_labels,
    match_any_class,
    precision_recall_table,
    read_exported_image,
    write_matrix_png,
    write_precision_recall_png,
)
from .run_cellcognition_target_benchmark import MODEL_PATH


def full_frame_predictions(labels, model, confidence, batch_size, device):
    """Run normalized full fields in GPU batches without image slicing."""
    rows = []
    fields = list(labels.groupby("image_id", sort=True))
    for start in range(0, len(fields), batch_size):
        batch = fields[start:start + batch_size]
        images = [prepare_model_input(preprocess_image(read_exported_image(group.resolved_image_path.iloc[0])), 3) for _, group in batch]
        results = model.predict(images, imgsz=640, conf=confidence, device=device, batch=len(batch), verbose=False)
        for (image_id, group), result in zip(batch, results):
            if result.boxes is None:
                continue
            for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                rows.append({"image_id": image_id, "group_id": group.group_id.iloc[0], "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3], "centre_x": (box[0] + box[2]) / 2, "centre_y": (box[1] + box[3]) / 2, "confidence": score, "class_name": model.names[class_id]})
    return pd.DataFrame(rows, columns=("image_id", "group_id", "x_min", "y_min", "x_max", "y_max", "centre_x", "centre_y", "confidence", "class_name"))


def run(project_directory, raw_images_directory, output_root=r"D:\CelFDriveBenchmark\runs", confidence=.5, batch_size=8, device="cuda:0", model_path=MODEL_PATH):
    """Run full-frame batched GPU inference and save comparable benchmark artifacts."""
    import torch
    from ultralytics import YOLO

    labels = load_exported_labels(project_directory, raw_images_directory)
    run_directory = create_run_directory(output_root, "cellclicker_exported_labels_fullframe")
    model_path = Path(model_path).resolve()
    try:
        predictions = full_frame_predictions(labels, YOLO(str(model_path)), confidence, batch_size, device)
        matches = match_any_class(predictions, labels)
        matrix = full_confusion(labels, predictions, matches)
        metrics = precision_recall_table(matrix)
        display_matrix = matrix.T
        display_matrix.index.name = "predicted_class"
        display_matrix.columns.name = "ground_truth_class"
        labels.to_csv(run_directory / "exported_cellclicker_labels.csv", index=False)
        predictions.to_csv(run_directory / "predictions.csv", index=False)
        matches.to_csv(run_directory / "iou_matches.csv", index=False)
        display_matrix.to_csv(run_directory / "full_confusion_matrix_with_background.csv")
        write_matrix_png(display_matrix, run_directory / "full_confusion_matrix_with_background.png", "YOLO prediction", "CellClicker exported ground truth")
        metrics[metrics.class_name.isin(CLASSES)].to_csv(run_directory / "precision_recall_by_stage.csv", index=False)
        write_precision_recall_png(metrics[metrics.class_name.isin(CLASSES)], run_directory / "precision_recall_by_stage.png")
        detection_metrics(predictions, labels, CLASSES).to_csv(run_directory / "ultralytics_style_ap_by_stage.csv", index=False)
        pd.DataFrame([{"metric": "class_agnostic_detection_recall_iou50", "value": len(matches) / len(labels)}, {"metric": "class_aware_precision_iou50", "value": metrics.iloc[-1].precision}, {"metric": "class_aware_recall_iou50", "value": metrics.iloc[-1].recall}, {"metric": "stage_accuracy_among_iou_matches", "value": sum(labels.loc[m.label_index, "class_name"] == predictions.loc[m.prediction_index, "class_name"] for m in matches.itertuples(index=False)) / len(matches) if len(matches) else 0.0}, {"metric": "ground_truth_boxes", "value": len(labels)}, {"metric": "labelled_timepoints", "value": labels.image_id.nunique()}, {"metric": "predictions", "value": len(predictions)}]).to_csv(run_directory / "summary.csv", index=False)
        manifest = {"project": str(Path(project_directory).resolve()), "label_source": "CellClicker user_selections/exported_labels", "class_names": CLASSES, "model_sha256": _sha256(model_path), "confidence": confidence, "device": device, "cuda_available": torch.cuda.is_available(), "inference_method": "full-frame Ultralytics inference", "field_batch_size": batch_size, "sahi_used": False}
        (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        (run_directory / "FAILED.txt").write_text("Full-frame exported-label evaluation did not complete; do not use this directory.", encoding="utf-8")
        raise
    return run_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-directory", required=True)
    parser.add_argument("--raw-images-directory", required=True)
    parser.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    parser.add_argument("--confidence", type=float, default=.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    arguments = parser.parse_args()
    print(run(arguments.project_directory, arguments.raw_images_directory, arguments.output_root, arguments.confidence, arguments.batch_size, arguments.device, arguments.model_path))
