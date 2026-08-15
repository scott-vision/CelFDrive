"""Export released manual H2B classifier samples to auditable YOLO labels.

Full-frame labels are intentionally marked partial: the released manual samples
do not exhaustively annotate every mitotic object in their source fields.
"""
import csv
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

SOURCE_TO_MODEL = {"pro": 0, "prometa": 2, "meta": 3, "earlyana": 4, "lateana": 4, "telo": 5}
CLASS_NAMES = ["prophase", "earlyprometaphase", "prometaphase", "metaphase", "anaphase", "telophase"]
FILENAME = re.compile(r"P(?P<position>\d+)_T(?P<frame>\d+)_X(?P<x>\d+)_Y(?P<y>\d+)__img\.png")


def _yolo_line(class_id, x_min, y_min, x_max, y_max, width, height):
    centre_x, centre_y = (x_min + x_max) / 2 / width, (y_min + y_max) / 2 / height
    box_width, box_height = (x_max - x_min) / width, (y_max - y_min) / height
    return f"{class_id} {centre_x:.8f} {centre_y:.8f} {box_width:.8f} {box_height:.8f}"


def build(samples_root, raw_images_root, output_root):
    """Build crop-complete and full-frame-partial YOLO exports from sample masks."""
    samples_root, raw_images_root, output_root = map(Path, (samples_root, raw_images_root, output_root))
    crops_images, crops_labels = output_root / "crops_complete" / "images", output_root / "crops_complete" / "labels"
    frames_images, frames_labels = output_root / "full_frames_partial" / "images", output_root / "full_frames_partial" / "labels"
    for directory in (crops_images, crops_labels, frames_images, frames_labels):
        directory.mkdir(parents=True, exist_ok=True)
    rows, frame_boxes, image_cache = [], {}, {}
    for image_path in sorted(samples_root.glob("*/*__img.png")):
        source_class = image_path.parent.name
        if source_class not in SOURCE_TO_MODEL:
            continue
        match = FILENAME.fullmatch(image_path.name)
        if match is None:
            raise ValueError(f"Unexpected sample filename: {image_path.name}")
        info = match.groupdict()
        mask_path = image_path.with_name(image_path.name.replace("__img.png", "__msk.png"))
        crop, mask = np.asarray(Image.open(image_path)), np.asarray(Image.open(mask_path)) > 0
        ys, xs = np.where(mask)
        if not len(xs):
            raise ValueError(f"Empty mask: {mask_path}")
        height, width = crop.shape[:2]
        # Verified against source TIFF pixels: x is centred conventionally, y is
        # one-based in the CellCognition sample-name convention.
        left, top = int(info["x"]) - width // 2, int(info["y"]) - height // 2 + 1
        x_min, y_min, x_max, y_max = left + int(xs.min()), top + int(ys.min()), left + int(xs.max()) + 1, top + int(ys.max()) + 1
        frame_key = f"P{info['position']}_T{int(info['frame']):05}"
        full_image = raw_images_root / info["position"] / f"tubulin_P{info['position']}_T{int(info['frame']):05}_Crfp_Z1_S1.tif"
        if not full_image.is_file():
            raise FileNotFoundError(full_image)
        if frame_key not in image_cache:
            image_cache[frame_key] = tifffile.imread(full_image).shape
        full_height, full_width = image_cache[frame_key][:2]
        if not (0 <= x_min < x_max <= full_width and 0 <= y_min < y_max <= full_height):
            raise ValueError(f"Mask box outside full image for {image_path.name}")
        class_id = SOURCE_TO_MODEL[source_class]
        crop_target = crops_images / image_path.name
        if not crop_target.exists():
            crop_target.write_bytes(image_path.read_bytes())
        (crops_labels / image_path.with_suffix(".txt").name).write_text(_yolo_line(class_id, int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, width, height) + "\n", encoding="utf-8")
        frame_boxes.setdefault(frame_key, {"source": full_image, "size": (full_width, full_height), "boxes": []})["boxes"].append(_yolo_line(class_id, x_min, y_min, x_max, y_max, full_width, full_height))
        rows.append({"sample_image": str(image_path), "sample_mask": str(mask_path), "full_image_id": frame_key, "full_image": str(full_image), "source_class": source_class, "yolo_class_id": class_id, "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max})
    for frame_key, details in frame_boxes.items():
        target = frames_images / f"{frame_key}.tif"
        if not target.exists():
            target.hardlink_to(details["source"])
        (frames_labels / f"{frame_key}.txt").write_text("\n".join(details["boxes"]) + "\n", encoding="utf-8")
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "dataset.yaml").write_text("names:\n" + "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)) + "\n", encoding="utf-8")
    (output_root / "full_frames_partial" / "README.txt").write_text("These full-frame labels cover only released manual classifier samples. They are partial annotations and must not be used as ordinary YOLO detection training labels or for precision/AP.\n", encoding="utf-8")
    return len(rows), len(frame_boxes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--raw-images-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(build(args.samples_root, args.raw_images_root, args.output_root))
