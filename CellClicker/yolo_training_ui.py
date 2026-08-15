"""Tk interface for configuration-driven YOLO training."""

import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .yolo_training import (
    _normalize_path,
    _summarize_projects,
    load_training_config,
    run_training_config,
    validate_training_config,
    write_training_config,
)
from .tooltips import add_tooltip


TRAINING_SETTING_TOOLTIPS = {
    "Base model": "Path to the pretrained Ultralytics .pt checkpoint used to initialize training.",
    "Output root": "Directory under which the new training run and its artifacts will be created.",
    "Run name": "Name of the run subdirectory. Use a unique, descriptive value.",
    "Epochs": "Maximum number of complete passes through the training dataset.",
    "Image size": "Square input size in pixels used for training and inference.",
    "Batch": "Images processed per training step. Reduce this if GPU memory is exhausted.",
    "Patience": "Stop after this many epochs without validation improvement.",
    "Device": "Ultralytics device, such as 0 for the first GPU or cpu.",
}


class YOLOTrainingUI:
    """Tk interface for selecting reviewed projects and training a YOLO model.

    Training synchronizes exported normalized labels into project ``labels``
    directories and writes run artifacts beneath the chosen output root.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Training")
        self.root.geometry("1500x840")

        self.train_projects = []
        self.val_projects = []
        self.test_projects = []

        self.train_count_var = tk.StringVar(value="Train: 0 folders, 0 usable images")
        self.val_count_var = tk.StringVar(value="Validation: 0 folders, 0 usable images")
        self.test_count_var = tk.StringVar(value="Test: 0 folders, 0 usable images")
        self.status_var = tk.StringVar(value="Add training, validation, and test project folders.")

        default_model = os.path.join(
            os.getcwd(), "Models", "yolo11x_p99p99_bg05", "weights", "best.pt"
        )
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
        self.test_listbox = None
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
        selection_frame.grid_columnconfigure(1, weight=1)
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

        self.val_listbox = self._build_project_panel(
            selection_frame,
            "Validation Projects",
            self.val_count_var,
            self.add_val_project,
            self.remove_val_project,
            self.clear_val_projects,
            1,
        )

        self.test_listbox = self._build_project_panel(
            selection_frame,
            "Test Projects",
            self.test_count_var,
            self.add_test_project,
            self.remove_test_project,
            self.clear_test_projects,
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
        tk.Button(action_row, text="Load Configuration", command=self.load_configuration, width=18).pack(side=tk.LEFT)
        tk.Button(action_row, text="Save Configuration", command=self.save_configuration, width=18).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(action_row, text="Refresh Counts", command=self.refresh_counts, width=18).pack(side=tk.LEFT)
        self.train_button = tk.Button(action_row, text="Prepare And Train", command=self.start_training, width=20)
        add_tooltip(
            self.train_button,
            "Prepare the dataset, train the model, and evaluate the best checkpoint on the test split.",
        )
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
        split_guidance = {
            "Training Projects": "Used to fit model weights. Projects must not also appear in validation or test.",
            "Validation Projects": "Used for model selection and early stopping. Keep separate from training and test.",
            "Test Projects": "Held out for final evaluation. Do not use these projects for training decisions.",
        }
        add_tooltip(listbox, split_guidance[title])
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
        add_tooltip(entry, TRAINING_SETTING_TOOLTIPS[label_text])
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

    def add_test_project(self):
        project_dir = self._choose_project()
        if project_dir and project_dir not in self.test_projects:
            self.test_projects.append(project_dir)
            self.test_listbox.insert(tk.END, project_dir)
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

    def remove_test_project(self):
        selected = list(self.test_listbox.curselection())
        if not selected:
            return
        for index in reversed(selected):
            del self.test_projects[index]
            self.test_listbox.delete(index)
        self.refresh_counts()

    def clear_train_projects(self):
        self._set_projects(self.train_projects, self.train_listbox, [])

    def clear_val_projects(self):
        self._set_projects(self.val_projects, self.val_listbox, [])

    def clear_test_projects(self):
        self._set_projects(self.test_projects, self.test_listbox, [])

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

    def _apply_training_config(self, config):
        self._set_projects(self.train_projects, self.train_listbox, config["splits"]["train"])
        self._set_projects(self.val_projects, self.val_listbox, config["splits"]["val"])
        self._set_projects(self.test_projects, self.test_listbox, config["splits"]["test"])
        self.model_path_var.set(config["model"]["path"])
        self.output_root_var.set(config["run"]["output_root"])
        self.run_name_var.set(config["run"]["name"])
        self.epochs_var.set(str(config["training"]["epochs"]))
        self.imgsz_var.set(str(config["training"]["imgsz"]))
        self.batch_var.set(str(config["training"]["batch"]))
        self.patience_var.set(str(config["training"]["patience"]))
        self.device_var.set(config["training"]["device"])
        self.refresh_counts()

    def load_configuration(self):
        config_path = filedialog.askopenfilename(
            title="Load YOLO Training Configuration",
            filetypes=[("YAML configuration", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not config_path:
            return
        try:
            config = load_training_config(config_path)
            self._apply_training_config(config)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Load Training Configuration", str(exc), parent=self.root)
            return
        self.status_var.set(f"Loaded training configuration: {_normalize_path(config_path)}")

    def save_configuration(self):
        try:
            config = self._validate_inputs()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save Training Configuration", str(exc), parent=self.root)
            return
        config_path = filedialog.asksaveasfilename(
            title="Save YOLO Training Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML configuration", "*.yaml"), ("YAML configuration", "*.yml")],
        )
        if not config_path:
            return
        try:
            write_training_config(config_path, config)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save Training Configuration", str(exc), parent=self.root)
            return
        self.status_var.set(f"Saved training configuration: {_normalize_path(config_path)}")

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

        try:
            _, _, test_count = _summarize_projects(self.test_projects)
            self.test_count_var.set(f"Test: {len(self.test_projects)} folders, {test_count} usable images")
        except Exception as exc:
            self.test_count_var.set(f"Test: error - {exc}")

    def _validate_inputs(self):
        config = {
            "schema_version": 1,
            "splits": {
                "train": list(self.train_projects),
                "val": list(self.val_projects),
                "test": list(self.test_projects),
            },
            "model": {"path": self.model_path_var.get().strip()},
            "run": {
                "output_root": self.output_root_var.get().strip(),
                "name": self.run_name_var.get().strip(),
            },
            "training": {
                "epochs": self.epochs_var.get().strip(),
                "imgsz": self.imgsz_var.get().strip(),
                "batch": self.batch_var.get().strip(),
                "patience": self.patience_var.get().strip(),
                "device": self.device_var.get().strip(),
            },
        }
        return validate_training_config(config, config_directory=os.getcwd())

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
                result = run_training_config(config, progress_callback=progress_callback)
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
        self._append_log(f"Resolved configuration: {result['training_config']}")
        self._append_log(f"Summary CSV: {result['summary_csv']}")
        self._append_log(f"Test metrics: {result['test_metrics']}")
        messagebox.showinfo(
            "YOLO Training Complete",
            f"Run directory:\n{result['run_dir']}\n\nTest metrics:\n{result['test_metrics']}\n\nSummary CSV:\n{result['summary_csv']}",
            parent=self.root,
        )


def launch_yolo_training_ui():
    """Create and run the standalone YOLO training Tk interface."""
    root = tk.Tk()
    app = YOLOTrainingUI(root)
    root.mainloop()
    return app
