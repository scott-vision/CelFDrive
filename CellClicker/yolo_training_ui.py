"""Prepare reviewed CellClicker projects and launch Ultralytics YOLO training.

Projects provide images plus normalized YOLO labels; this module synchronizes
labels into the layout expected by Ultralytics and writes a dataset YAML file.
"""

import csv
import os
import shutil
import tempfile
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .tracking_xml import read_tracking_xml
from .tracking_export import exported_labels_are_current


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_CLASS_NAMES = [
    "prophase",
    "earlyprometaphase",
    "prometaphase",
    "metaphase",
    "anaphase",
    "telophase",
]


def _normalize_path(path):
    return os.path.normpath(os.fspath(path))


def _yaml_path(path):
    return _normalize_path(path).replace("\\", "/")


def _ensure_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for YOLO training. Install `ultralytics` in the active environment."
        ) from exc
    return YOLO


def _iter_image_files(images_dir):
    for dirpath, _, filenames in os.walk(images_dir):
        for filename in filenames:
            suffix = os.path.splitext(filename)[1].lower()
            if suffix in IMAGE_EXTENSIONS:
                yield os.path.join(dirpath, filename)


def _project_key(project_dir):
    return os.path.basename(_normalize_path(os.fspath(project_dir).rstrip("\\/")))


def _load_project_classes(project_dir):
    """Read ordered class names from a project's tracking XML."""
    tracking_xml = os.path.join(project_dir, "user_selections", "tracking_review.xml")
    if not os.path.exists(tracking_xml):
        raise FileNotFoundError(
            f"Training project `{project_dir}` is missing `user_selections/tracking_review.xml`."
        )

    tracking_data = read_tracking_xml(tracking_xml)
    classes = tracking_data.get("classes", {})
    if not classes:
        return DEFAULT_CLASS_NAMES

    ordered = []
    for class_id in sorted(classes, key=lambda value: int(value)):
        ordered.append(str(classes[class_id]))
    return ordered


def _resolve_project_pairs(project_dir):
    """Return image/label pairs whose image-relative paths match exactly."""
    project_dir = _normalize_path(project_dir)
    images_dir = os.path.join(project_dir, "images")
    labels_dir = os.path.join(project_dir, "user_selections", "exported_labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Training project `{project_dir}` is missing `images/`.")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(
            f"Training project `{project_dir}` is missing `user_selections/exported_labels/`. Export labels first."
        )
    pairs = []
    for image_path in sorted(_iter_image_files(images_dir)):
        relative_image = os.path.relpath(image_path, images_dir)
        relative_label = os.path.splitext(relative_image)[0] + ".txt"
        label_path = os.path.join(labels_dir, relative_label)
        if os.path.exists(label_path):
            pairs.append((_normalize_path(image_path), _normalize_path(label_path), relative_image))
    return pairs


def _summarize_projects(project_dirs):
    summary = []
    class_names = None
    total_pairs = 0
    for project_dir in project_dirs:
        pairs = _resolve_project_pairs(project_dir)
        if pairs and not exported_labels_are_current(project_dir):
            raise ValueError(
                f"Training project `{project_dir}` has stale exported labels. Rebuild YOLO labels from tracking review first."
            )
        project_classes = _load_project_classes(project_dir)
        if class_names is None:
            class_names = project_classes
        elif class_names != project_classes:
            raise ValueError(
                "Selected projects do not share the same class list in `tracking_review.xml`."
            )
        summary.append(
            {
                "project_dir": _normalize_path(project_dir),
                "project_name": _project_key(project_dir),
                "count": len(pairs),
                "pairs": pairs,
            }
        )
        total_pairs += len(pairs)
    return summary, (class_names or list(DEFAULT_CLASS_NAMES)), total_pairs


def _write_dataset_yaml(yaml_path, train_image_dirs, val_image_dirs, class_names):
    lines = [
        "train:",
    ]
    for train_dir in train_image_dirs:
        lines.append(f"  - {_yaml_path(train_dir)}")
    lines.append("val:")
    for val_dir in val_image_dirs:
        lines.append(f"  - {_yaml_path(val_dir)}")
    lines.extend([
        f"nc: {len(class_names)}",
        "names:",
    ])
    for index, class_name in enumerate(class_names):
        lines.append(f"  {index}: {class_name}")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _sync_project_labels(project_dir, progress_callback=None, current_offset=0, total_items=1):
    project_dir = _normalize_path(project_dir)
    images_dir = os.path.join(project_dir, "images")
    exported_labels_dir = os.path.join(project_dir, "user_selections", "exported_labels")
    labels_dir = os.path.join(project_dir, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Training project `{project_dir}` is missing `images/`.")
    if not os.path.isdir(exported_labels_dir):
        raise FileNotFoundError(
            f"Training project `{project_dir}` is missing `user_selections/exported_labels/`. Export labels first."
        )
    pairs = _resolve_project_pairs(project_dir)
    if not pairs:
        raise ValueError(
            f"Training project `{project_dir}` has no images with exported labels. Export labels first."
        )
    if not exported_labels_are_current(project_dir):
        raise ValueError(
            f"Training project `{project_dir}` has stale exported labels. Rebuild YOLO labels from tracking review first."
        )

    staging_dir = tempfile.mkdtemp(prefix="celfdrive-labels-", dir=project_dir)
    copied_successfully = False
    try:
        for index, (_, label_path, relative_image) in enumerate(pairs, start=1):
            if progress_callback is not None:
                progress_callback(current_offset + index, total_items, label_path)
            relative_label = os.path.splitext(relative_image)[0] + ".txt"
            target_label_path = os.path.join(staging_dir, relative_label)
            os.makedirs(os.path.dirname(target_label_path), exist_ok=True)
            shutil.copy2(label_path, target_label_path)

        if os.path.isdir(labels_dir):
            shutil.rmtree(labels_dir)
        os.replace(staging_dir, labels_dir)
        copied_successfully = True
    finally:
        if not copied_successfully and os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)
    return len(pairs)


def prepare_yolo_sources(train_dirs, val_dirs, yaml_path, progress_callback=None):
    """Synchronize project labels and write an Ultralytics dataset YAML file.

    Project roots must contain ``images`` and exported normalized YOLO labels.
    ``progress_callback``, when supplied, receives ``(current, total, path)``.
    Returns YAML metadata and image/label counts. Project ``labels`` directories
    are replaced only after a complete staged copy succeeds.
    """
    train_summary, class_names, train_count = _summarize_projects(train_dirs)
    val_summary, val_class_names, val_count = _summarize_projects(val_dirs)
    if class_names != val_class_names:
        raise ValueError("Training and validation projects do not share the same class list.")

    total_items = train_count + val_count
    current = 0
    image_dirs = {"train": [], "val": []}

    for split_name, split_summary in (("train", train_summary), ("val", val_summary)):
        for project in split_summary:
            copied = _sync_project_labels(
                project["project_dir"],
                progress_callback=progress_callback,
                current_offset=current,
                total_items=total_items,
            )
            current += copied
            image_dirs[split_name].append(os.path.join(project["project_dir"], "images"))

    yaml_path = _normalize_path(yaml_path)
    yaml_directory = os.path.dirname(yaml_path)
    if yaml_directory:
        os.makedirs(yaml_directory, exist_ok=True)
    _write_dataset_yaml(yaml_path, image_dirs["train"], image_dirs["val"], class_names)

    return {
        "yaml_path": yaml_path,
        "class_names": class_names,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "train_count": train_count,
        "val_count": val_count,
        "train_image_dirs": image_dirs["train"],
        "val_image_dirs": image_dirs["val"],
    }


def _read_last_results_row(results_csv):
    if not os.path.exists(results_csv):
        return {}

    with open(results_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        last_row = None
        for row in reader:
            cleaned = {key.strip(): value.strip() for key, value in row.items() if key is not None}
            last_row = cleaned
    return last_row or {}


def _append_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_yolo_model(
    train_dirs,
    val_dirs,
    output_root,
    run_name,
    model_path,
    epochs,
    imgsz,
    batch,
    patience,
    device,
    progress_callback=None,
):
    """Prepare projects, execute Ultralytics training, and record a run summary.

    ``output_root`` receives the run directory, generated dataset YAML, and CSV
    summaries. Raises ``FileExistsError`` rather than overwriting a named run.
    """
    YOLO = _ensure_ultralytics()

    output_root = _normalize_path(output_root)
    run_dir = os.path.join(output_root, run_name)
    yaml_path = os.path.join(output_root, "_yamls", f"{run_name}.yaml")
    if os.path.exists(run_dir):
        raise FileExistsError(f"Training run directory already exists: `{run_dir}`.")
    if os.path.exists(yaml_path):
        raise FileExistsError(f"Training YAML already exists: `{yaml_path}`.")

    dataset_info = prepare_yolo_sources(train_dirs, val_dirs, yaml_path, progress_callback=progress_callback)

    model = YOLO(model_path)
    train_results = model.train(
        data=dataset_info["yaml_path"],
        project=output_root,
        name=run_name,
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        patience=patience,
        device=device,
    )

    save_dir = getattr(train_results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is None:
        save_dir = run_dir
    save_dir = _normalize_path(str(save_dir))

    results_csv = os.path.join(save_dir, "results.csv")
    metrics = _read_last_results_row(results_csv)
    summary_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "model_path": _normalize_path(model_path),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "patience": int(patience),
        "device": str(device),
        "dataset_yaml": _normalize_path(dataset_info["yaml_path"]),
        "train_images": int(dataset_info["train_count"]),
        "val_images": int(dataset_info["val_count"]),
        "train_projects": ";".join(project["project_dir"] for project in dataset_info["train_summary"]),
        "val_projects": ";".join(project["project_dir"] for project in dataset_info["val_summary"]),
        "train_image_dirs": ";".join(dataset_info["train_image_dirs"]),
        "val_image_dirs": ";".join(dataset_info["val_image_dirs"]),
        "class_names": ";".join(dataset_info["class_names"]),
        "run_dir": save_dir,
        "best_weights": os.path.join(save_dir, "weights", "best.pt"),
        "last_weights": os.path.join(save_dir, "weights", "last.pt"),
        "results_csv": results_csv,
        "results_png": os.path.join(save_dir, "results.png"),
        "confusion_matrix_png": os.path.join(save_dir, "confusion_matrix.png"),
        "confusion_matrix_normalized_png": os.path.join(save_dir, "confusion_matrix_normalized.png"),
        "pr_curve_png": os.path.join(save_dir, "PR_curve.png"),
        "p_curve_png": os.path.join(save_dir, "P_curve.png"),
        "r_curve_png": os.path.join(save_dir, "R_curve.png"),
        "f1_curve_png": os.path.join(save_dir, "F1_curve.png"),
    }
    summary_row.update(metrics)

    run_summary_csv = os.path.join(save_dir, "training_summary.csv")
    global_summary_csv = os.path.join(output_root, "training_runs.csv")
    _append_csv_row(run_summary_csv, summary_row)
    _append_csv_row(global_summary_csv, summary_row)

    return {
        "dataset_info": dataset_info,
        "run_dir": save_dir,
        "summary_csv": run_summary_csv,
        "global_summary_csv": global_summary_csv,
        "summary_row": summary_row,
    }


class YOLOTrainingUI:
    """Tk interface for selecting reviewed projects and training a YOLO model.

    Training synchronizes exported normalized labels into project ``labels``
    directories and writes run artifacts beneath the chosen output root.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Training")
        self.root.geometry("1220x840")

        self.train_projects = []
        self.val_projects = []

        self.train_count_var = tk.StringVar(value="Train: 0 folders, 0 usable images")
        self.val_count_var = tk.StringVar(value="Validation: 0 folders, 0 usable images")
        self.status_var = tk.StringVar(value="Add training and validation project folders.")

        default_model = os.path.join(os.getcwd(), "Models", "Base", "yolov9c.pt")
        default_output = os.path.join(os.getcwd(), "Models", "training_ui_runs")
        self.model_path_var = tk.StringVar(value=default_model)
        self.output_root_var = tk.StringVar(value=default_output)
        self.run_name_var = tk.StringVar(value=datetime.now().strftime("train_%Y%m%d_%H%M%S"))
        self.epochs_var = tk.StringVar(value="150")
        self.imgsz_var = tk.StringVar(value="640")
        self.batch_var = tk.StringVar(value="16")
        self.patience_var = tk.StringVar(value="50")
        self.device_var = tk.StringVar(value="0")

        self.train_listbox = None
        self.val_listbox = None
        self.progress_bar = None
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_label_var = tk.StringVar(value="0 / 0")
        self.log_text = None
        self.train_button = None

        self._build_layout()
        self.refresh_counts()

    def _build_layout(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        selection_frame = tk.Frame(top)
        selection_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        selection_frame.grid_columnconfigure(0, weight=1)
        selection_frame.grid_columnconfigure(2, weight=1)
        selection_frame.grid_rowconfigure(0, weight=1)

        self.train_listbox = self._build_project_panel(
            selection_frame,
            "Training Projects",
            self.train_count_var,
            self.add_train_project,
            self.remove_train_project,
            self.clear_train_projects,
            0,
        )

        move_frame = tk.Frame(selection_frame)
        move_frame.grid(row=0, column=1, sticky="ns", padx=8)
        tk.Button(move_frame, text="->", command=self.move_selected_to_val, width=6).pack(pady=(100, 4))
        tk.Button(move_frame, text="<-", command=self.move_selected_to_train, width=6).pack(pady=4)

        self.val_listbox = self._build_project_panel(
            selection_frame,
            "Validation Projects",
            self.val_count_var,
            self.add_val_project,
            self.remove_val_project,
            self.clear_val_projects,
            2,
        )

        settings = tk.LabelFrame(top, text="Training Settings")
        settings.pack(side=tk.TOP, fill=tk.X, pady=(10, 10))
        settings.grid_columnconfigure(1, weight=1)

        self._add_setting_row(settings, 0, "Base model", self.model_path_var, self.browse_model)
        self._add_setting_row(settings, 1, "Output root", self.output_root_var, self.browse_output_root)
        self._add_setting_row(settings, 2, "Run name", self.run_name_var)
        self._add_setting_row(settings, 3, "Epochs", self.epochs_var)
        self._add_setting_row(settings, 4, "Image size", self.imgsz_var)
        self._add_setting_row(settings, 5, "Batch", self.batch_var)
        self._add_setting_row(settings, 6, "Patience", self.patience_var)
        self._add_setting_row(settings, 7, "Device", self.device_var)

        action_row = tk.Frame(top)
        action_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        tk.Button(action_row, text="Refresh Counts", command=self.refresh_counts, width=18).pack(side=tk.LEFT)
        self.train_button = tk.Button(action_row, text="Prepare And Train", command=self.start_training, width=20)
        self.train_button.pack(side=tk.LEFT, padx=8)

        progress_row = tk.Frame(top)
        progress_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self.progress_bar = ttk.Progressbar(progress_row, orient="horizontal", mode="determinate", maximum=1, variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(progress_row, textvariable=self.progress_label_var, width=14, anchor=tk.E).pack(side=tk.LEFT, padx=(8, 0))

        log_frame = tk.LabelFrame(top, text="Status")
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tk.Label(log_frame, textvariable=self.status_var, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=(8, 0))
        log_container = tk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        log_container.grid_rowconfigure(0, weight=1)
        log_container.grid_columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_container, height=12, wrap="none")
        log_scroll_y = tk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scroll_x = tk.Scrollbar(log_container, orient=tk.HORIZONTAL, command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=log_scroll_y.set, xscrollcommand=log_scroll_x.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll_y.grid(row=0, column=1, sticky="ns")
        log_scroll_x.grid(row=1, column=0, sticky="ew")

    def _build_project_panel(self, parent, title, count_var, add_command, remove_command, clear_command, column):
        panel = tk.LabelFrame(parent, text=title)
        panel.grid(row=0, column=column, sticky="nsew")
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        list_container = tk.Frame(panel)
        list_container.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)

        listbox = tk.Listbox(list_container, height=14, width=40, exportselection=False)
        list_scroll_y = tk.Scrollbar(list_container, orient=tk.VERTICAL, command=listbox.yview)
        list_scroll_x = tk.Scrollbar(list_container, orient=tk.HORIZONTAL, command=listbox.xview)
        listbox.configure(yscrollcommand=list_scroll_y.set, xscrollcommand=list_scroll_x.set)
        listbox.grid(row=0, column=0, sticky="nsew")
        list_scroll_y.grid(row=0, column=1, sticky="ns")
        list_scroll_x.grid(row=1, column=0, sticky="ew")

        button_row = tk.Frame(panel)
        button_row.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        tk.Button(button_row, text="Add Folder", command=add_command, width=12).pack(side=tk.LEFT)
        tk.Button(button_row, text="Remove", command=remove_command, width=12).pack(side=tk.LEFT, padx=4)
        tk.Button(button_row, text="Clear", command=clear_command, width=12).pack(side=tk.LEFT)

        tk.Label(panel, textvariable=count_var, anchor=tk.W, justify=tk.LEFT).grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        return listbox

    def _add_setting_row(self, parent, row, label_text, variable, browse_command=None):
        tk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=8, pady=4)
        entry = tk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        if browse_command is not None:
            tk.Button(parent, text="Browse", command=browse_command, width=10).grid(row=row, column=2, padx=8, pady=4)

    def _append_log(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _set_projects(self, target, listbox, projects):
        target.clear()
        target.extend(projects)
        listbox.delete(0, tk.END)
        for project in target:
            listbox.insert(tk.END, project)
        self.refresh_counts()

    def _choose_project(self):
        project_dir = filedialog.askdirectory(title="Select Project Folder")
        if not project_dir:
            return None
        return _normalize_path(project_dir)

    def add_train_project(self):
        project_dir = self._choose_project()
        if project_dir and project_dir not in self.train_projects:
            self.train_projects.append(project_dir)
            self.train_listbox.insert(tk.END, project_dir)
            self.refresh_counts()

    def add_val_project(self):
        project_dir = self._choose_project()
        if project_dir and project_dir not in self.val_projects:
            self.val_projects.append(project_dir)
            self.val_listbox.insert(tk.END, project_dir)
            self.refresh_counts()

    def remove_train_project(self):
        selected = list(self.train_listbox.curselection())
        if not selected:
            return
        for index in reversed(selected):
            del self.train_projects[index]
            self.train_listbox.delete(index)
        self.refresh_counts()

    def remove_val_project(self):
        selected = list(self.val_listbox.curselection())
        if not selected:
            return
        for index in reversed(selected):
            del self.val_projects[index]
            self.val_listbox.delete(index)
        self.refresh_counts()

    def clear_train_projects(self):
        self._set_projects(self.train_projects, self.train_listbox, [])

    def clear_val_projects(self):
        self._set_projects(self.val_projects, self.val_listbox, [])

    def _move_projects(self, source_projects, source_listbox, target_projects, target_listbox):
        selected = list(source_listbox.curselection())
        if not selected:
            return

        moving = [source_projects[index] for index in selected]
        for project_dir in moving:
            if project_dir not in target_projects:
                target_projects.append(project_dir)
                target_listbox.insert(tk.END, project_dir)

        for index in reversed(selected):
            del source_projects[index]
            source_listbox.delete(index)

        self.refresh_counts()

    def move_selected_to_val(self):
        self._move_projects(self.train_projects, self.train_listbox, self.val_projects, self.val_listbox)

    def move_selected_to_train(self):
        self._move_projects(self.val_projects, self.val_listbox, self.train_projects, self.train_listbox)

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Base YOLO Model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(_normalize_path(path))

    def browse_output_root(self):
        path = filedialog.askdirectory(title="Select Training Output Root")
        if path:
            self.output_root_var.set(_normalize_path(path))

    def refresh_counts(self):
        try:
            _, _, train_count = _summarize_projects(self.train_projects)
            self.train_count_var.set(f"Train: {len(self.train_projects)} folders, {train_count} usable images")
        except Exception as exc:
            self.train_count_var.set(f"Train: error - {exc}")

        try:
            _, _, val_count = _summarize_projects(self.val_projects)
            self.val_count_var.set(f"Validation: {len(self.val_projects)} folders, {val_count} usable images")
        except Exception as exc:
            self.val_count_var.set(f"Validation: error - {exc}")

    def _validate_inputs(self):
        if not self.train_projects:
            raise ValueError("Add at least one training project.")
        if not self.val_projects:
            raise ValueError("Add at least one validation project.")
        if not self.run_name_var.get().strip():
            raise ValueError("Run name is required.")
        if not self.model_path_var.get().strip():
            raise ValueError("Base model path is required.")
        return {
            "model_path": _normalize_path(self.model_path_var.get().strip()),
            "output_root": _normalize_path(self.output_root_var.get().strip()),
            "run_name": self.run_name_var.get().strip(),
            "epochs": int(self.epochs_var.get().strip()),
            "imgsz": int(self.imgsz_var.get().strip()),
            "batch": int(self.batch_var.get().strip()),
            "patience": int(self.patience_var.get().strip()),
            "device": self.device_var.get().strip(),
        }

    def start_training(self):
        try:
            config = self._validate_inputs()
        except Exception as exc:
            messagebox.showerror("Training Configuration", str(exc), parent=self.root)
            return

        self.train_button.config(state=tk.DISABLED)
        self.progress_bar.configure(mode="determinate", maximum=1)
        self.progress_var.set(0)
        self.progress_label_var.set("0 / 0")
        self.status_var.set("Syncing project labels and starting training...")
        self._append_log("Syncing exported labels into each project's top-level labels/ folder.")

        state = {"error": None, "result": None}

        def progress_callback(current, total, image_path):
            def update():
                self.progress_bar.configure(mode="determinate", maximum=max(1, total))
                self.progress_var.set(current)
                self.progress_label_var.set(f"{current} / {max(1, total)}")
                if image_path:
                    self.status_var.set(f"Syncing labels: {os.path.basename(image_path)}")
            self.root.after(0, update)

        def worker():
            try:
                result = train_yolo_model(
                    train_dirs=list(self.train_projects),
                    val_dirs=list(self.val_projects),
                    output_root=config["output_root"],
                    run_name=config["run_name"],
                    model_path=config["model_path"],
                    epochs=config["epochs"],
                    imgsz=config["imgsz"],
                    batch=config["batch"],
                    patience=config["patience"],
                    device=config["device"],
                    progress_callback=progress_callback,
                )
                state["result"] = result
            except Exception as exc:
                state["error"] = exc
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(10)
        self.root.after(150, lambda: self._poll_training(thread, state))

    def _poll_training(self, thread, state):
        if thread.is_alive():
            self.root.after(250, lambda: self._poll_training(thread, state))
            return

        self.progress_bar.stop()
        self.train_button.config(state=tk.NORMAL)

        if state["error"] is not None:
            self.status_var.set("YOLO training failed.")
            self._append_log(f"Training failed: {state['error']}")
            messagebox.showerror("YOLO Training Failed", str(state["error"]), parent=self.root)
            return

        result = state["result"]
        self.progress_bar.configure(mode="determinate", maximum=1)
        self.progress_var.set(1)
        self.progress_label_var.set("done")
        self.status_var.set(f"YOLO training complete. Run saved to {result['run_dir']}.")
        self._append_log(f"Training complete. Run directory: {result['run_dir']}")
        self._append_log(f"Summary CSV: {result['summary_csv']}")
        messagebox.showinfo(
            "YOLO Training Complete",
            f"Run directory:\n{result['run_dir']}\n\nSummary CSV:\n{result['summary_csv']}",
            parent=self.root,
        )


def launch_yolo_training_ui():
    """Create and run the standalone YOLO training Tk interface."""
    root = tk.Tk()
    app = YOLOTrainingUI(root)
    root.mainloop()
    return app
