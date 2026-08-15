"""Read-only viewer for the YOLO dataset exported from a CellClicker project."""

import math
import os
import tkinter as tk

from PIL import Image, ImageTk

from .tracking_xml import read_tracking_xml


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
BOX_COLORS = ("#39a9ff", "#ff6961", "#63d471", "#ffd166", "#c792ea", "#ff9f43")


def find_exported_label_pairs(project_dir):
    """Return image and label paths for every label in the exported dataset.

    Labels are stored below ``user_selections/exported_labels`` using the same
    path relative to ``images`` as their source image.  A missing source image
    is an invalid export and is reported rather than silently omitted.
    """
    project_dir = os.path.normpath(project_dir)
    labels_root = os.path.join(project_dir, "user_selections", "exported_labels")
    images_root = os.path.join(project_dir, "images")
    if not os.path.isdir(labels_root):
        raise FileNotFoundError(f"Exported labels folder was not found: {labels_root}")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Project images folder was not found: {images_root}")

    label_paths = []
    for directory, _, filenames in os.walk(labels_root):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                label_paths.append(os.path.join(directory, filename))

    if not label_paths:
        raise ValueError(f"No YOLO label files were found in {labels_root}")

    pairs = []
    missing_images = []
    for label_path in sorted(label_paths):
        relative_label_path = os.path.relpath(label_path, labels_root)
        image_stem = os.path.splitext(relative_label_path)[0]
        image_path = None
        for extension in IMAGE_EXTENSIONS:
            candidate = os.path.join(images_root, image_stem + extension)
            if os.path.isfile(candidate):
                image_path = candidate
                break
        if image_path is None:
            missing_images.append(relative_label_path)
        else:
            pairs.append((image_path, label_path))

    if missing_images:
        preview = ", ".join(missing_images[:3])
        if len(missing_images) > 3:
            preview += f", and {len(missing_images) - 3} more"
        raise FileNotFoundError(
            "Exported labels have no matching source image below images/: " + preview
        )
    return pairs


def read_yolo_labels(label_path):
    """Read normalized YOLO boxes, validating the label file with line context."""
    labels = []
    with open(label_path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"{label_path}, line {line_number}: expected 5 YOLO values.")
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"{label_path}, line {line_number}: invalid YOLO value.") from exc
            if class_id < 0 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{label_path}, line {line_number}: invalid class or coordinate.")
            x_center, y_center, width, height = values
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"{label_path}, line {line_number}: coordinates must be normalized to 0--1.")
            labels.append((class_id, x_center, y_center, width, height))
    return labels


def yolo_box_to_display_coordinates(label, display_width, display_height):
    """Convert one normalized YOLO box to display-space corner coordinates."""
    _, x_center, y_center, width, height = label
    x_min = (x_center - width / 2.0) * display_width
    y_min = (y_center - height / 2.0) * display_height
    x_max = (x_center + width / 2.0) * display_width
    y_max = (y_center + height / 2.0) * display_height
    return x_min, y_min, x_max, y_max


def box_label_position(x_min, y_min, x_max, y_max, display_height, margin=3):
    """Place a class name immediately outside its box rather than over the cell."""
    if y_min >= 16:
        return x_min, y_min - margin, tk.SW
    return x_min, min(y_max + margin, display_height - margin), tk.NW


class ExportedDatasetViewer:
    """Display exported labels over source images without permitting changes."""

    def __init__(self, master, project_dir):
        self.project_dir = os.path.normpath(project_dir)
        self.image_label_pairs = find_exported_label_pairs(self.project_dir)
        tracking_path = os.path.join(self.project_dir, "user_selections", "tracking_review.xml")
        self.class_names = read_tracking_xml(tracking_path).get("classes", {})
        self.current_index = 0
        self.photo_image = None
        self._initial_canvas_rendered = False

        self.window = tk.Toplevel(master)
        self.window.title("Exported Dataset Viewer")
        self.window.geometry("1100x820")
        self.window.minsize(600, 500)
        self.window.transient(master)

        self.summary_var = tk.StringVar()
        tk.Label(self.window, textvariable=self.summary_var, anchor=tk.W).pack(fill=tk.X, padx=10, pady=(10, 4))
        tk.Label(
            self.window,
            text="Read-only final export preview. Left/Right arrows move between images.",
            anchor=tk.W,
        ).pack(fill=tk.X, padx=10, pady=(0, 6))

        self.canvas = tk.Canvas(self.window, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self.canvas.bind("<Configure>", self._display_after_canvas_layout)

        controls = tk.Frame(self.window)
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(controls, text="Previous", command=self.show_previous, width=12).pack(side=tk.LEFT)
        tk.Button(controls, text="Next", command=self.show_next, width=12).pack(side=tk.LEFT, padx=6)
        self.position_var = tk.StringVar()
        tk.Label(controls, textvariable=self.position_var).pack(side=tk.RIGHT)

        self.window.bind("<Left>", lambda event: self.show_previous())
        self.window.bind("<Right>", lambda event: self.show_next())

    def _display_after_canvas_layout(self, event):
        """Render only after Tk has assigned the canvas its actual size."""
        if self._initial_canvas_rendered or event.width <= 20 or event.height <= 20:
            return
        self._initial_canvas_rendered = True
        self.display_current_image()

    def show_previous(self):
        self.current_index = (self.current_index - 1) % len(self.image_label_pairs)
        self.display_current_image()

    def show_next(self):
        self.current_index = (self.current_index + 1) % len(self.image_label_pairs)
        self.display_current_image()

    def display_current_image(self):
        """Fit the current source image in the canvas and overlay final labels."""
        image_path, label_path = self.image_label_pairs[self.current_index]
        labels = read_yolo_labels(label_path)
        with Image.open(image_path) as source_image:
            source_image = source_image.convert("RGB")
            available_width = max(self.canvas.winfo_width() - 20, 1)
            available_height = max(self.canvas.winfo_height() - 20, 1)
            scale = min(available_width / source_image.width, available_height / source_image.height, 1.0)
            display_width = max(1, int(round(source_image.width * scale)))
            display_height = max(1, int(round(source_image.height * scale)))
            display_image = source_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(display_image)
        self.canvas.delete("all")
        x_offset = max((self.canvas.winfo_width() - display_width) // 2, 0)
        y_offset = max((self.canvas.winfo_height() - display_height) // 2, 0)
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo_image)
        for label in labels:
            class_id = label[0]
            color = BOX_COLORS[class_id % len(BOX_COLORS)]
            x_min, y_min, x_max, y_max = yolo_box_to_display_coordinates(label, display_width, display_height)
            self.canvas.create_rectangle(x_offset + x_min, y_offset + y_min, x_offset + x_max, y_offset + y_max, outline=color, width=2)
            class_name = self.class_names.get(class_id, f"class {class_id}")
            text_x, text_y, anchor = box_label_position(
                x_min, y_min, x_max, y_max, display_height,
            )
            self.canvas.create_text(
                x_offset + text_x,
                y_offset + text_y,
                anchor=anchor,
                text=class_name,
                fill=color,
            )

        relative_image_path = os.path.relpath(image_path, os.path.join(self.project_dir, "images"))
        self.summary_var.set(f"{relative_image_path} — {len(labels)} final box(es)")
        self.position_var.set(f"Image {self.current_index + 1} of {len(self.image_label_pairs)}")
