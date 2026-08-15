"""Evaluate saved full-frame YOLO predictions against released manual H2B samples."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from benchmarking import box_iou, create_run_directory, preprocess_image

CLASSES = ["prophase", "earlyprometaphase", "prometaphase", "metaphase", "anaphase", "telophase"]
MISSED = "background / missed"


def _match(predictions, labels):
    """Greedily pair partial-manual labels to full-frame predictions by IoU."""
    matches, used = [], set()
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
            matches.append({"prediction_index": prediction_index, "label_index": label_index, "iou": iou})
    return pd.DataFrame(matches, columns=("prediction_index", "label_index", "iou"))


def _render_pairs(labels, predictions, output_directory):
    """Render native-size manual sample and corresponding detector crop pairs."""
    output_directory.mkdir(parents=True, exist_ok=True)
    for index, label in labels.iterrows():
        sample = Image.open(label.sample_image).convert("RGB")
        width, height = sample.size
        left = sample.copy()
        right = sample.copy()
        left_draw, right_draw = ImageDraw.Draw(left), ImageDraw.Draw(right)
        left_draw.rectangle((label.local_x_min, label.local_y_min, label.local_x_max - 1, label.local_y_max - 1), outline="lime", width=2)
        left_draw.text((1, 1), label.class_name, fill="lime", stroke_width=1, stroke_fill="black")
        field_predictions = predictions[predictions.image_id == label.image_id]
        for _, prediction in field_predictions.iterrows():
            x_min, y_min = prediction.x_min - label.crop_left, prediction.y_min - label.crop_top
            x_max, y_max = prediction.x_max - label.crop_left, prediction.y_max - label.crop_top
            if x_max <= 0 or y_max <= 0 or x_min >= width or y_min >= height:
                continue
            right_draw.rectangle((max(0, x_min), max(0, y_min), min(width - 1, x_max), min(height - 1, y_max)), outline="red", width=2)
            right_draw.text((max(0, x_min), max(0, y_min)), f"{prediction.class_name} {prediction.confidence:.2f}", fill="red", stroke_width=1, stroke_fill="black")
        panel_width = max(width, 150)
        canvas = Image.new("RGB", (panel_width * 2 + 8, height + 18), "black")
        canvas.paste(left, ((panel_width - width) // 2, 18))
        canvas.paste(right, (panel_width + 8 + (panel_width - width) // 2, 18))
        draw = ImageDraw.Draw(canvas)
        draw.text((1, 2), "manual label", fill="lime")
        draw.text((panel_width + 9, 2), "YOLO prediction", fill="red")
        canvas.save(output_directory / f"{index:04d}_{Path(label.sample_image).stem}.png")


def _render_whole_field_pairs(labels, predictions, output_directory):
    """Render each complete source field with all manual labels and predictions."""
    import tifffile

    output_directory.mkdir(parents=True, exist_ok=True)
    for image_id, field_labels in labels.groupby("image_id", sort=True):
        intensity = preprocess_image(tifffile.imread(field_labels.full_image.iloc[0]))
        image = Image.fromarray(intensity).convert("RGB")
        width, height = image.size
        left, right = image.copy(), image.copy()
        left_draw, right_draw = ImageDraw.Draw(left), ImageDraw.Draw(right)
        for _, label in field_labels.iterrows():
            left_draw.rectangle((label.x_min, label.y_min, label.x_max - 1, label.y_max - 1), outline="lime", width=2)
            left_draw.text((label.x_min, label.y_min), label.class_name, fill="lime", stroke_width=1, stroke_fill="black")
        for _, prediction in predictions[predictions.image_id == image_id].iterrows():
            right_draw.rectangle((prediction.x_min, prediction.y_min, prediction.x_max - 1, prediction.y_max - 1), outline="red", width=2)
            right_draw.text((prediction.x_min, prediction.y_min), f"{prediction.class_name} {prediction.confidence:.2f}", fill="red", stroke_width=1, stroke_fill="black")
        canvas = Image.new("RGB", (width * 2 + 10, height + 20), "black")
        canvas.paste(left, (0, 20))
        canvas.paste(right, (width + 10, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((1, 3), "manual labels (partial)", fill="lime")
        draw.text((width + 11, 3), "YOLO detections", fill="red")
        canvas.save(output_directory / f"{image_id}.png")


def run(dataset_root, prediction_csv, output_root=r"D:\CelFDriveBenchmark\runs"):
    """Evaluate manual sample boxes using an existing frozen full-frame prediction table."""
    dataset_root = Path(dataset_root)
    labels = pd.read_csv(dataset_root / "manifest.csv")
    labels["class_name"] = labels.yolo_class_id.map(dict(enumerate(CLASSES)))
    labels["image_id"] = labels.full_image_id.str.replace("P", "", regex=False).str.replace("_T", "_t", regex=False)
    sizes = labels.apply(lambda row: Image.open(row.sample_image).size, axis=1)
    labels["crop_width"], labels["crop_height"] = [size[0] for size in sizes], [size[1] for size in sizes]
    labels["centre_x"] = labels.sample_image.str.extract(r"_X(\d+)_")[0].astype(int)
    labels["crop_left"] = labels.centre_x - labels.crop_width // 2
    labels["crop_top"] = labels.sample_image.str.extract(r"_Y(\d+)_")[0].astype(int) - labels.crop_height // 2 + 1
    labels["local_x_min"] = labels.x_min - labels.crop_left
    labels["local_y_min"] = labels.y_min - labels.crop_top
    labels["local_x_max"] = labels.x_max - labels.crop_left
    labels["local_y_max"] = labels.y_max - labels.crop_top
    predictions = pd.read_csv(prediction_csv)
    predictions = predictions[predictions.image_id.isin(labels.image_id)].reset_index(drop=True)
    matches = _match(predictions, labels)
    labels["prediction_class"], labels["confidence"], labels["iou"] = None, None, None
    for match in matches.itertuples(index=False):
        prediction = predictions.loc[match.prediction_index]
        labels.loc[match.label_index, ["prediction_class", "confidence", "iou"]] = [prediction.class_name, prediction.confidence, match.iou]
    labels["detected"] = labels.index.isin(set(matches.label_index))
    matrix = pd.crosstab(labels.class_name, labels.prediction_class.fillna(MISSED)).reindex(index=CLASSES, columns=CLASSES + [MISSED], fill_value=0)
    run_directory = create_run_directory(output_root, "cellcognition_manual_samples_fullframe")
    labels.to_csv(run_directory / "manual_sample_labels_and_matches.csv", index=False)
    predictions.to_csv(run_directory / "predictions.csv", index=False)
    matches.to_csv(run_directory / "iou_matches.csv", index=False)
    matrix.to_csv(run_directory / "confusion_matrix_with_background.csv")
    pd.DataFrame([{"metric": "manual_sample_recall_iou50", "value": labels.detected.mean()}, {"metric": "stage_accuracy_among_matched", "value": (labels.loc[labels.detected, "class_name"] == labels.loc[labels.detected, "prediction_class"]).mean() if labels.detected.any() else 0.0}, {"metric": "manual_samples", "value": len(labels)}, {"metric": "source_fields", "value": labels.image_id.nunique()}, {"metric": "prediction_boxes_in_source_fields", "value": len(predictions)}]).to_csv(run_directory / "summary.csv", index=False)
    _render_pairs(labels, predictions, run_directory / "crop_pairs")
    _render_whole_field_pairs(labels, predictions, run_directory / "whole_field_pairs")
    manifest = {"dataset": "released CellCognition manually annotated H2B classifier samples", "endpoint": "IoU >= 0.50", "predictions_source": str(Path(prediction_csv).resolve()), "annotation_completeness": "manual sample annotations are partial within full frames; unmatched predictions are not false positives", "confidence": .25}
    (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    args = parser.parse_args()
    print(run(args.dataset_root, args.prediction_csv, args.output_root))
