"""Render one label-versus-detection PNG pair for every CellCognition target."""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tifffile

from benchmarking import preprocess_image


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _font():
    return ImageFont.load_default()


def _draw_target(draw, target, colour):
    x, y = float(target.centre_x), float(target.centre_y)
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=colour, width=3)
    draw.text((x + 9, y + 8), f"{target.source_label} ({target.class_name})", fill=colour, font=_font())


def _draw_predictions(draw, predictions):
    for _, prediction in predictions.iterrows():
        box = (float(prediction.x_min), float(prediction.y_min), float(prediction.x_max), float(prediction.y_max))
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], max(0, box[1] - 12)), f"{prediction.class_name} {float(prediction.confidence):.2f}", fill="red", font=_font())


def render_pairs(run_directory, output_directory=None):
    """Create a side-by-side PNG for every released target in a completed run."""
    run_directory = Path(run_directory)
    output_directory = Path(output_directory) if output_directory else run_directory / "label_detection_pairs"
    output_directory.mkdir(parents=True, exist_ok=False)
    targets = pd.read_csv(run_directory / "released_classifier_targets.csv")
    predictions = pd.read_csv(run_directory / "predictions.csv")
    matches = pd.read_csv(run_directory / "target_matches.csv").set_index("object_id")
    manifest_rows = []
    output_index = 0
    for image_id, image_targets in targets.sort_values("image_id").groupby("image_id", sort=False):
        raw = tifffile.imread(image_targets.resolved_image_path.iloc[0])
        source = Image.fromarray(preprocess_image(raw)).convert("RGB")
        image_predictions = predictions[predictions.image_id == image_id]
        for _, target in image_targets.iterrows():
            left, right = source.copy(), source.copy()
            left_draw, right_draw = ImageDraw.Draw(left), ImageDraw.Draw(right)
            _draw_target(left_draw, target, "lime")
            _draw_predictions(right_draw, image_predictions)
            _draw_target(right_draw, target, "lime")
            header_height = 22
            pair = Image.new("RGB", (source.width * 2, source.height + header_height), "black")
            pair.paste(left, (0, header_height))
            pair.paste(right, (source.width, header_height))
            header = ImageDraw.Draw(pair)
            header.text((4, 4), "Released CellCognition label (green)", fill="white", font=_font())
            header.text((source.width + 4, 4), "YOLO detections (red); reference centre (green)", fill="white", font=_font())
            filename = f"{output_index:05d}_{_safe_name(target.image_id)}_{_safe_name(target.object_id)}.png"
            output_index += 1
            pair.save(output_directory / filename, optimize=True)
            match = matches.loc[str(target.object_id)]
            manifest_rows.append({"pair_png": filename, "image_id": target.image_id, "object_id": target.object_id,
                                  "source_label": target.source_label, "class_name": target.class_name,
                                  "centre_hit": match.centre_hit, "prediction_class": match.prediction_class})
    pd.DataFrame(manifest_rows).to_csv(output_directory / "pair_manifest.csv", index=False)
    return output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    print(render_pairs(args.run_directory, args.output_directory))


if __name__ == "__main__":
    main()
