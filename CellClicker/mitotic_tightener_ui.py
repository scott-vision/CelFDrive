"""Small Tk interface for building and training mitotic tightener datasets."""

import json
import os
import queue
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .mitotic_tightener import summarise_projects, train_tightener_model


SETTINGS_FORMAT_VERSION = 1


def write_tightener_training_settings(path, projects, settings):
    """Write reusable split selection and training fields to a JSON file."""
    payload = {
        "format_version": SETTINGS_FORMAT_VERSION,
        "projects": {split: [os.path.normpath(item) for item in projects[split]] for split in ("train", "val", "test")},
        "settings": settings,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_tightener_training_settings(path):
    """Read and validate a tightener settings JSON file."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format_version") != SETTINGS_FORMAT_VERSION:
        raise ValueError(f"Unsupported tightener settings format in `{path}`.")
    projects = payload.get("projects")
    settings = payload.get("settings")
    if not isinstance(projects, dict) or not isinstance(settings, dict):
        raise ValueError(f"Tightener settings file `{path}` is missing projects or settings.")
    for split in ("train", "val", "test"):
        if not isinstance(projects.get(split), list) or not all(isinstance(item, str) for item in projects[split]):
            raise ValueError(f"Tightener settings file `{path}` has an invalid {split} project list.")
    required = ("output_root", "run_name", "epochs", "batch", "patience", "device")
    if any(key not in settings for key in required):
        raise ValueError(f"Tightener settings file `{path}` is missing a training setting.")
    return projects, settings


class MitoticTightenerTrainingUI:
    """Select explicit train/validation/test project folders and train YOLO11n."""

    def __init__(self, root):
        self.root = root
        root.title("Mitotic Tightener Training")
        root.geometry("1060x760")
        self.projects = {"train": [], "val": [], "test": []}
        self.count_vars = {name: tk.StringVar(value=f"{name.title()}: 0 folders, 0 usable, 0 skipped") for name in self.projects}
        self.listboxes = {}
        self.status_var = tk.StringVar(value="Add train, validation, and test project folders.")
        self.output_var = tk.StringVar(value=os.path.join(os.getcwd(), "Models", "tightener_runs"))
        self.name_var = tk.StringVar(value=datetime.now().strftime("tightener_%Y%m%d_%H%M%S"))
        self.epochs_var, self.batch_var, self.patience_var, self.device_var = (tk.StringVar(value=value) for value in ("150", "16", "50", "0"))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_label = tk.StringVar(value="0 / 0")
        self.progress_bar = None
        self.events = queue.Queue()
        self.train_button = None
        self.log = None
        self._build_layout()

    def _build_layout(self):
        top = tk.Frame(self.root)
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        split_frame = tk.Frame(top)
        split_frame.pack(fill=tk.X)
        for column, split in enumerate(("train", "val", "test")):
            panel = tk.LabelFrame(split_frame, text={"train": "Training", "val": "Validation", "test": "Test"}[split])
            panel.grid(row=0, column=column, sticky="nsew", padx=4)
            split_frame.grid_columnconfigure(column, weight=1)
            box = tk.Listbox(panel, height=8, width=42, exportselection=False)
            box.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
            self.listboxes[split] = box
            actions = tk.Frame(panel); actions.pack(fill=tk.X, padx=6, pady=(0, 4))
            tk.Button(actions, text="Add Folder", command=lambda name=split: self.add_folder(name)).pack(side=tk.LEFT)
            tk.Button(actions, text="Remove", command=lambda name=split: self.remove_selected(name)).pack(side=tk.LEFT, padx=4)
            tk.Label(panel, textvariable=self.count_vars[split], anchor=tk.W).pack(fill=tk.X, padx=6, pady=(0, 6))
        settings = tk.LabelFrame(top, text="Training Settings")
        settings.pack(fill=tk.X, pady=10)
        settings.grid_columnconfigure(1, weight=1)
        self._setting(settings, 0, "Output root", self.output_var, self.browse_output)
        self._setting(settings, 1, "Run name", self.name_var)
        self._setting(settings, 2, "Epochs", self.epochs_var)
        self._setting(settings, 3, "Batch", self.batch_var)
        self._setting(settings, 4, "Patience", self.patience_var)
        self._setting(settings, 5, "Device", self.device_var)
        action = tk.Frame(top); action.pack(fill=tk.X, pady=(0, 8))
        tk.Button(action, text="Refresh Summary", command=self.refresh_counts).pack(side=tk.LEFT)
        tk.Button(action, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(action, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT, padx=4)
        self.train_button = tk.Button(action, text="Prepare And Train YOLO11n", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=8)
        self.progress_bar = ttk.Progressbar(action, maximum=1, variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        tk.Label(action, textvariable=self.progress_label, width=12).pack(side=tk.LEFT)
        status = tk.LabelFrame(top, text="Status"); status.pack(fill=tk.BOTH, expand=True)
        tk.Label(status, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X, padx=6, pady=6)
        self.log = tk.Text(status, height=14, wrap="word"); self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

    def _setting(self, parent, row, label, variable, browse=None):
        tk.Label(parent, text=label + ":").grid(row=row, column=0, sticky=tk.W, padx=6, pady=3)
        tk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        if browse:
            tk.Button(parent, text="Browse", command=browse).grid(row=row, column=2, padx=6, pady=3)

    def _append_log(self, text):
        self.log.insert(tk.END, text + "\n"); self.log.see(tk.END)

    def add_project_to_split(self, split, project_dir):
        project_dir = os.path.normpath(project_dir)
        if project_dir not in self.projects[split]:
            self.projects[split].append(project_dir); self.listboxes[split].insert(tk.END, project_dir)

    def add_folder(self, split):
        path = filedialog.askdirectory(title=f"Select {split} tightener project")
        if path:
            self.add_project_to_split(split, path); self.refresh_counts()

    def remove_selected(self, split):
        for index in reversed(self.listboxes[split].curselection()):
            del self.projects[split][index]; self.listboxes[split].delete(index)
        self.refresh_counts()

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Tightener Training Output Root")
        if path:
            self.output_var.set(path)

    def _settings_fields(self):
        return {
            "output_root": self.output_var.get().strip(), "run_name": self.name_var.get().strip(),
            "epochs": self.epochs_var.get().strip(), "batch": self.batch_var.get().strip(),
            "patience": self.patience_var.get().strip(), "device": self.device_var.get().strip(),
        }

    def save_settings(self):
        path = filedialog.asksaveasfilename(
            title="Save Mitotic Tightener Settings", defaultextension=".json",
            initialfile="mitotic_tightener_settings.json", filetypes=[("JSON settings", "*.json")], parent=self.root,
        )
        if not path:
            return
        try:
            write_tightener_training_settings(path, self.projects, self._settings_fields())
        except OSError as exc:
            messagebox.showerror("Save Tightener Settings", str(exc), parent=self.root)
            return
        self.status_var.set(f"Saved tightener settings to {path}")

    def load_settings(self):
        path = filedialog.askopenfilename(
            title="Load Mitotic Tightener Settings", filetypes=[("JSON settings", "*.json"), ("All files", "*.*")], parent=self.root,
        )
        if not path:
            return
        try:
            projects, settings = read_tightener_training_settings(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Load Tightener Settings", str(exc), parent=self.root)
            return
        for split in self.projects:
            self.projects[split] = []
            self.listboxes[split].delete(0, tk.END)
            for project_dir in projects[split]:
                self.add_project_to_split(split, project_dir)
        self.output_var.set(settings["output_root"])
        self.name_var.set(settings["run_name"])
        self.epochs_var.set(str(settings["epochs"]))
        self.batch_var.set(str(settings["batch"]))
        self.patience_var.set(str(settings["patience"]))
        self.device_var.set(str(settings["device"]))
        self.refresh_counts()
        self.status_var.set(f"Loaded tightener settings from {path}")

    def refresh_counts(self):
        for split in self.projects:
            try:
                result = summarise_projects(self.projects[split])
                self.count_vars[split].set(f"{split.title()}: {len(self.projects[split])} folders, {result['valid']} usable, {result['skipped']} skipped")
            except Exception as exc:
                self.count_vars[split].set(f"{split.title()}: error - {exc}")

    def _config(self):
        if any(not self.projects[name] for name in self.projects):
            raise ValueError("Add at least one project to each train, validation, and test split.")
        if not self.name_var.get().strip():
            raise ValueError("Run name is required.")
        root = os.path.normpath(self.output_var.get().strip())
        return {"output_root": root, "run_name": self.name_var.get().strip(), "epochs": int(self.epochs_var.get()),
                "batch": int(self.batch_var.get()), "patience": int(self.patience_var.get()), "device": self.device_var.get().strip(),
                "dataset_dir": os.path.join(os.getcwd(), "Models", "tightener_datasets", self.name_var.get().strip())}

    def start_training(self):
        try:
            config = self._config()
        except Exception as exc:
            messagebox.showerror("Tightener Configuration", str(exc), parent=self.root); return
        self.train_button.config(state=tk.DISABLED); self.status_var.set("Preparing review-style crop dataset...")
        state = {"result": None, "error": None}
        def preparation(current, total, path): self.events.put(("prepare", current, total, path))
        def epoch(epoch_number, total_epochs, metrics): self.events.put(("epoch", epoch_number, total_epochs, metrics))
        def dataset_ready(info): self.events.put(("dataset", info, None, None))
        def worker():
            try:
                state["result"] = train_tightener_model(self.projects["train"], self.projects["val"], self.projects["test"],
                    progress_callback=preparation, epoch_callback=epoch, dataset_callback=dataset_ready, **config)
            except Exception as exc:
                state["error"] = exc
        thread = threading.Thread(target=worker, daemon=True); thread.start()
        self.root.after(100, lambda: self._poll(thread, state))

    def _poll(self, thread, state):
        while not self.events.empty():
            kind, first, second, detail = self.events.get_nowait()
            if kind == "prepare":
                self.progress_bar.configure(maximum=max(1, second))
                self.progress_var.set(first); self.progress_label.set(f"{first} / {second}"); self.status_var.set(f"Preparing: {os.path.basename(detail)}")
            elif kind == "epoch":
                self.status_var.set(f"Training epoch {first} / {second}"); self._append_log(f"Epoch {first}/{second}: {detail}")
            else:
                self.status_var.set(f"Dataset ready. Training YOLO11n at {first['imgsz']} px...")
                self._append_log(f"Dataset: {first['dataset_dir']}; selected image size: {first['imgsz']} px")
                self._append_log(f"Training manifest: {first['manifest_path']}")
        if thread.is_alive():
            self.root.after(150, lambda: self._poll(thread, state)); return
        self.train_button.config(state=tk.NORMAL)
        if state["error"]:
            self.status_var.set("Mitotic tightener training failed."); self._append_log(f"Failed: {state['error']}")
            messagebox.showerror("Mitotic Tightener Failed", str(state["error"]), parent=self.root); return
        result = state["result"]
        self.status_var.set(f"Training complete. Best weights: {result['best_weights']}")
        self._append_log(f"Run directory: {result['run_dir']}")
        self._append_log(f"Training manifest: {result['dataset_info']['manifest_path']}")
        self._append_log(f"Best weights: {result['best_weights']}")
        self._append_log(f"Last weights: {result['last_weights']}")
        self._append_log(f"Validation metrics: {result['validation_metrics']}")
        self._append_log(f"Test metrics: {result['test_metrics']}")
        messagebox.showinfo("Mitotic Tightener Complete", f"Best weights:\n{result['best_weights']}\n\nTest metrics:\n{result['test_metrics']}", parent=self.root)
