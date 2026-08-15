"""Evaluate SAHI against CellClicker's exported six-stage YOLO labels."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from benchmarking import _sha256, box_iou, create_run_directory, detection_metrics
from run_cellcognition_full_label_sahi_benchmark import batched_sahi_prediction_table
from run_cellcognition_target_benchmark import MODEL_PATH

CLASSES = ["prophase", "earlyprometaphase", "prometaphase", "metaphase", "anaphase", "telophase"]
BACKGROUND = "background / missed"


def load_exported_labels(project_directory, raw_images_directory):
    """Read CellClicker's phase-labelled YOLO exports as pixel-space boxes."""
    labels_directory = Path(project_directory) / "user_selections" / "exported_labels"
    raw_images_directory = Path(raw_images_directory)
    rows = []
    for label_path in sorted(labels_directory.glob("P*_t*.txt")):
        match = re.fullmatch(r"P(?P<position>\d+)_t(?P<frame>\d+)\.txt", label_path.name)
        if match is None:
            continue
        position, frame = match.group("position"), int(match.group("frame"))
        raw_path = raw_images_directory / position / f"tubulin_P{position}_T{frame:05}_Crfp_Z1_S1.tif"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        for object_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{object_index + 1}")
            class_id = int(values[0])
            if class_id < 0 or class_id >= len(CLASSES):
                raise ValueError(f"Unknown stage class {class_id} at {label_path}:{object_index + 1}")
            x_center, y_center, width, height = map(float, values[1:])
            rows.append({"image_id": f"{position}_t{frame:05}", "group_id": position, "object_id": f"{label_path.stem}:{object_index}", "resolved_image_path": str(raw_path), "class_name": CLASSES[class_id], "x_min": (x_center - width / 2) * 1392, "y_min": (y_center - height / 2) * 1040, "x_max": (x_center + width / 2) * 1392, "y_max": (y_center + height / 2) * 1040, "centre_x": x_center * 1392, "centre_y": y_center * 1040})
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise ValueError(f"No exported YOLO labels found in {labels_directory}")
    return labels


def match_any_class(predictions, labels):
    """Make confidence-ordered class-agnostic IoU>=.50 matches."""
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


def full_confusion(labels, predictions, matches):
    """Include unmatched reference boxes and predictions as background bins."""
    matrix = pd.DataFrame(0, index=CLASSES + [BACKGROUND], columns=CLASSES + [BACKGROUND], dtype=int)
    matched_labels, matched_predictions = set(matches.label_index), set(matches.prediction_index)
    for match in matches.itertuples(index=False):
        matrix.loc[labels.loc[match.label_index, "class_name"], predictions.loc[match.prediction_index, "class_name"]] += 1
    for label_index in set(labels.index) - matched_labels:
        matrix.loc[labels.loc[label_index, "class_name"], BACKGROUND] += 1
    for prediction_index in set(predictions.index) - matched_predictions:
        matrix.loc[BACKGROUND, predictions.loc[prediction_index, "class_name"]] += 1
    matrix.index.name = "ground_truth_class"
    return matrix


def write_matrix_png(matrix, path, row_label, column_label):
    """Write a readable static heatmap for review and manuscript assembly."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            axis.text(column, row, str(matrix.iloc[row, column]), ha="center", va="center", fontsize=9)
    axis.set_xlabel(column_label)
    axis.set_ylabel(row_label)
    figure.colorbar(image, ax=axis, label="count")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def precision_recall_table(matrix):
    """Calculate stage-aware precision and recall from the full IoU confusion matrix.

    The background bin is retained in the matrix for accounting, but is not a
    biological class.  For each stage, false negatives include its missed
    detections and wrong-stage calls; false positives include unmatched model
    calls and calls assigned to that stage from another reference stage.
    """
    rows = []
    for class_name in CLASSES:
        true_positive = int(matrix.loc[class_name, class_name])
        false_positive = int(matrix[class_name].sum() - true_positive)
        false_negative = int(matrix.loc[class_name].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class_name": class_name, "true_positives": true_positive, "false_positives": false_positive, "false_negatives": false_negative, "precision": precision, "recall": recall, "f1": f1, "ground_truth_support": int(matrix.loc[class_name].sum()), "predicted_support": int(matrix[class_name].sum())})
    table = pd.DataFrame(rows)
    totals = table[["true_positives", "false_positives", "false_negatives"]].sum()
    overall_precision = totals.true_positives / (totals.true_positives + totals.false_positives)
    overall_recall = totals.true_positives / (totals.true_positives + totals.false_negatives)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    return pd.concat([table, pd.DataFrame([{"class_name": "overall (micro-average)", "true_positives": int(totals.true_positives), "false_positives": int(totals.false_positives), "false_negatives": int(totals.false_negatives), "precision": overall_precision, "recall": overall_recall, "f1": overall_f1, "ground_truth_support": int(table.ground_truth_support.sum()), "predicted_support": int(table.predicted_support.sum())}])], ignore_index=True)


def write_precision_recall_png(table, path):
    """Write a compact manuscript-ready precision/recall table image."""
    import matplotlib.pyplot as plt

    display = table[["class_name", "precision", "recall", "f1", "ground_truth_support"]].copy()
    for column in ("precision", "recall", "f1"):
        display[column] = display[column].map("{:.3f}".format)
    display.columns = ["Stage", "Precision", "Recall", "F1", "GT support"]
    figure, axis = plt.subplots(figsize=(8.4, 2.6), constrained_layout=True)
    axis.axis("off")
    table_artist = axis.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center")
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(10)
    table_artist.scale(1, 1.45)
    figure.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(project_directory, raw_images_directory, output_root=r"D:\CelFDriveBenchmark\runs", confidence=.5, merge_iou=.1, device="cuda:0"):
    """Run full exported-label evaluation with GPU-batched SAHI inference."""
    import sahi
    import torch
    from sahi import AutoDetectionModel

    labels = load_exported_labels(project_directory, raw_images_directory)
    run_directory = create_run_directory(output_root, "cellclicker_exported_labels_sahi")
    model_path = MODEL_PATH.resolve()
    try:
        model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(model_path), confidence_threshold=confidence, device=device, image_size=640)
        predictions = batched_sahi_prediction_table(labels, model, 640, 640, .25, 6, postprocess_metric="IOU", postprocess_threshold=merge_iou)
        matches = match_any_class(predictions, labels)
        matrix = full_confusion(labels, predictions, matches)
        labels.to_csv(run_directory / "exported_cellclicker_labels.csv", index=False)
        predictions.to_csv(run_directory / "predictions.csv", index=False)
        matches.to_csv(run_directory / "iou_matches.csv", index=False)
        display_matrix = matrix.T
        display_matrix.index.name = "predicted_class"
        display_matrix.columns.name = "ground_truth_class"
        display_matrix.to_csv(run_directory / "full_confusion_matrix_with_background.csv")
        write_matrix_png(display_matrix, run_directory / "full_confusion_matrix_with_background.png", "YOLO / SAHI prediction", "CellClicker exported ground truth")
        metrics = precision_recall_table(matrix)
        metrics[metrics.class_name.isin(CLASSES)].to_csv(run_directory / "precision_recall_by_stage.csv", index=False)
        write_precision_recall_png(metrics[metrics.class_name.isin(CLASSES)], run_directory / "precision_recall_by_stage.png")
        detection_metrics(predictions, labels, CLASSES).to_csv(run_directory / "ultralytics_style_ap_by_stage.csv", index=False)
        pd.DataFrame([{"metric": "class_agnostic_detection_recall_iou50", "value": len(matches) / len(labels)}, {"metric": "class_aware_precision_iou50", "value": sum(labels.loc[m.label_index, "class_name"] == predictions.loc[m.prediction_index, "class_name"] for m in matches.itertuples(index=False)) / len(predictions) if len(predictions) else 0.0}, {"metric": "stage_accuracy_among_iou_matches", "value": sum(labels.loc[m.label_index, "class_name"] == predictions.loc[m.prediction_index, "class_name"] for m in matches.itertuples(index=False)) / len(matches) if len(matches) else 0.0}, {"metric": "ground_truth_boxes", "value": len(labels)}, {"metric": "labelled_timepoints", "value": labels.image_id.nunique()}, {"metric": "predictions", "value": len(predictions)}]).to_csv(run_directory / "summary.csv", index=False)
        manifest = {"project": str(Path(project_directory).resolve()), "label_source": "CellClicker user_selections/exported_labels", "class_names": CLASSES, "model_sha256": _sha256(model_path), "confidence": confidence, "device": device, "cuda_available": torch.cuda.is_available(), "sahi_slice_px": 640, "sahi_overlap_ratio": .25, "sahi_merge_metric": "IOU", "sahi_merge_threshold": merge_iou, "sahi_merge_class_agnostic": False}
        (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        (run_directory / "FAILED.txt").write_text("Exported-label SAHI evaluation did not complete; do not use this directory.", encoding="utf-8")
        raise
    return run_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-directory", required=True)
    parser.add_argument("--raw-images-directory", required=True)
    parser.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    parser.add_argument("--confidence", type=float, default=.5)
    parser.add_argument("--merge-iou", type=float, default=.1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(run(args.project_directory, args.raw_images_directory, args.output_root, args.confidence, args.merge_iou, args.device))
