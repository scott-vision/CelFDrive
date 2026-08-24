"""Unified Tk application for the CellClicker annotation-to-training workflow."""

import os
import logging
import tkinter as tk
import threading
import time
from glob import glob
from tkinter import filedialog, messagebox, simpledialog, ttk

from .cell_clicker import ImageViewer
from .convert_selections_multiphase import aggregate_xml
from .image_selector_multiphase import load_ui_for_project
from .tracking_otsu import run_otsu_on_tracking_xml
from .tracking_export import export_tracking_xml_to_coco, export_tracking_xml_to_miniseries, export_tracking_xml_to_yolo
from .tracking_export import exported_labels_are_current
from .exported_dataset_viewer import ExportedDatasetViewer
from .tracking_sam2 import DEFAULT_SAM2_DEVICE, DEFAULT_SAM2_MODEL, run_sam2_on_tracking_xml
from .mitotic_tightener import (
    DEFAULT_TIGHTENER_SELECTION,
    DEFAULT_TIGHTENER_MODELS_ROOT,
    TIGHTENER_SELECTION_METADATA_KEY,
    configure_tightener_weights,
    run_tightener_on_tracking_xml,
)
from .tracking_review_ui import TrackingReviewUI
from .tracking_workflow import build_tracking_xml_from_dataset
from .tracking_xml import read_tracking_xml
from .yolo_training_ui import YOLOTrainingUI
from .mitotic_tightener_ui import MitoticTightenerTrainingUI
from .project_reconciliation import tracking_is_current
from .workflow_state import annotator_selection_files
from .duplicate_tracks import find_near_duplicate_tracks
from .project_reconciliation import delete_track_from_project
from .phase_settings import DEFAULT_PHASES, load_phases, phase_signature, save_phases, settings_path
from .tooltips import add_tooltip
from .project_paths import (
    CELL_REGIONS_FILENAME,
    migrate_legacy_cell_regions_xml,
    resolve_cell_regions_xml,
)


LOGGER = logging.getLogger(__name__)


class ProjectGUI:
    """Coordinate the GUI workflow from annotation through training export.

    The selected project follows the documented ``images`` and
    ``user_selections`` layout. Button callbacks may create XML, labels, crops,
    datasets, or model-training output within that project.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("CellClicker Project GUI")
        self.root.geometry("1100x760")
        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

        self.project_dir = None
        self.review_window = None

        self.export_box_type_var = tk.StringVar(value="preferred")

        self.project_var = tk.StringVar(value="No project loaded")
        self.images_status_var = tk.StringVar(value="images/: not checked")
        self.cell_xml_status_var = tk.StringVar(value=f"{CELL_REGIONS_FILENAME}: not checked")
        self.user_selections_status_var = tk.StringVar(value="user_selections/: not checked")
        self.aggregated_status_var = tk.StringVar(value="aggregated_tracking.xml: not checked")
        self.tracking_status_var = tk.StringVar(value="tracking_review.xml: not checked")
        self.status_var = tk.StringVar(value="Load a project to begin.")

        self._build_layout()
        self._refresh_project_status()

    def close_application(self):
        for child in list(self.root.winfo_children()):
            try:
                if isinstance(child, tk.Toplevel) and child.winfo_exists():
                    child.destroy()
            except tk.TclError:
                continue

        if self.review_window and self.review_window.winfo_exists():
            try:
                self.review_window.destroy()
            except tk.TclError:
                pass

        try:
            self.root.quit()
        except tk.TclError:
            pass

        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _build_layout(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        add_tooltip(
            tk.Button(top, text="Load Project", command=self.load_project),
            "Choose the project folder containing images/, not the images folder itself.",
        ).pack(side=tk.LEFT)
        tk.Button(top, text="Refresh Status", command=self._refresh_project_status).pack(side=tk.LEFT, padx=4)
        tk.Label(top, textvariable=self.project_var, anchor=tk.W).pack(side=tk.LEFT, padx=12)

        body = tk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.LabelFrame(body, text="Project")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        tk.Label(left, textvariable=self.images_status_var, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(left, textvariable=self.cell_xml_status_var, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(left, textvariable=self.user_selections_status_var, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(left, textvariable=self.aggregated_status_var, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(left, textvariable=self.tracking_status_var, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)

        right = tk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        stage1 = tk.LabelFrame(right, text="1. Project Annotation")
        stage1.pack(fill=tk.X, pady=(0, 10))
        tk.Button(stage1, text="Open CellClicker", command=self.open_cell_clicker, width=24).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Button(stage1, text="Open Phase Selector", command=self.open_phase_selector, width=24).pack(side=tk.LEFT, padx=8, pady=8)
        add_tooltip(
            tk.Button(stage1, text="Phase Selector Settings", command=self.open_phase_settings, width=24),
            "Define the ordered phase names and class IDs used by selection, export, and training. Changing them makes existing selections stale.",
        ).pack(side=tk.LEFT, padx=8, pady=8)

        stage2 = tk.LabelFrame(right, text="2. Aggregation and Build")
        stage2.pack(fill=tk.X, pady=(0, 10))
        add_tooltip(
            tk.Button(stage2, text="Aggregate User Selections", command=self.aggregate_user_selections, width=24),
            "Combine all reviewers' current phase selections into aggregated_tracking.xml.",
        ).pack(side=tk.LEFT, padx=8, pady=8)
        add_tooltip(
            tk.Button(stage2, text="Build Tracking XML", command=self.build_tracking_xml, width=24),
            "Create or reconcile the canonical tracking_review.xml used for review and export.",
        ).pack(side=tk.LEFT, padx=8, pady=8)

        stage3 = tk.LabelFrame(right, text="3. Box Generation")
        stage3.pack(fill=tk.X, pady=(0, 10))
        add_tooltip(
            tk.Button(stage3, text="Apply Otsu", command=self.apply_otsu, width=24),
            "Add threshold-derived box alternatives without replacing existing boxes.",
        ).pack(side=tk.LEFT, padx=8, pady=8)
        add_tooltip(
            tk.Button(stage3, text="Run SAM/SAM2", command=self.run_sam2, width=24),
            "Add segmentation-derived box alternatives without replacing existing boxes.",
        ).pack(side=tk.LEFT, padx=8, pady=8)
        add_tooltip(
            tk.Button(stage3, text="Configure Cell Tightener", command=self.configure_tightener, width=24),
            "Select trained weights and the detection-selection rule used by the cell tightener.",
        ).pack(side=tk.LEFT, padx=8, pady=8)
        add_tooltip(
            tk.Button(stage3, text="Run YOLO11 Cell Tightener", command=self.run_tightener, width=24),
            "Add tightener-generated box alternatives using the saved configuration.",
        ).pack(side=tk.LEFT, padx=8, pady=8)

        stage4 = tk.LabelFrame(right, text="4. Review")
        stage4.pack(fill=tk.X, pady=(0, 10))
        add_tooltip(
            tk.Button(stage4, text="Open Tracking Review", command=self.open_tracking_review, width=24),
            "Choose the preferred box and phase for each frame, then save the reviewed result.",
        ).pack(side=tk.LEFT, padx=8, pady=8)
        add_tooltip(
            tk.Button(stage4, text="Check Duplicate Tracks", command=self.check_duplicate_tracks, width=24),
            "Find tracks that may describe the same cell based on overlap across shared frames.",
        ).pack(side=tk.LEFT, padx=8, pady=8)

        stage5 = tk.LabelFrame(right, text="5. Export")
        stage5.pack(fill=tk.X, pady=(0, 10))
        export_options = tk.Frame(stage5)
        export_options.pack(fill=tk.X)
        tk.Label(export_options, text="Export box type:").pack(side=tk.LEFT, padx=(8, 4), pady=(6, 0))
        export_box_menu = tk.OptionMenu(export_options, self.export_box_type_var, "preferred", "original", "otsu", "sam2", "yolo11_tightened", "tightened")
        add_tooltip(
            export_box_menu,
            "Choose which box variant to export. preferred uses the per-frame choices made in Tracking Review; named variants may be missing on some frames.",
        )
        export_box_menu.pack(side=tk.LEFT, padx=(0, 8), pady=(6, 0))
        export_actions = tk.Frame(stage5)
        export_actions.pack(fill=tk.X)
        add_tooltip(
            tk.Button(export_actions, text="Export YOLO Labels", command=self.export_yolo_labels, width=24),
            "Create a complete YOLO training snapshot from the current reviewed data.",
        ).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Button(export_actions, text="Export COCO Labels", command=self.export_coco_labels, width=24).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Button(export_actions, text="Export Miniseries", command=self.export_miniseries, width=24).pack(side=tk.LEFT, padx=8, pady=8)
        add_tooltip(
            tk.Button(stage5, text="View Exported Dataset", command=self.view_exported_dataset, width=24),
            "Open a read-only preview of the current YOLO export.",
        ).pack(anchor=tk.W, padx=8, pady=(0, 8))

        stage6 = tk.LabelFrame(right, text="Notes")
        stage6.pack(fill=tk.X, pady=(0, 10))
        tk.Button(stage6, text="Open YOLO Training", command=self.open_yolo_training, width=24).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Button(stage6, text="Train Cell Tightener", command=self.open_tightener_training, width=24).pack(side=tk.LEFT, padx=8, pady=8)

        stage7 = tk.LabelFrame(right, text="Notes")
        stage7.pack(fill=tk.BOTH, expand=True)
        notes = (
            "Workflow:\n"
            "1. Load a dataset folder.\n"
            "2. Run CellClicker to create tracks in images/cell_regions.xml.\n"
            "3. Run the phase selector to create user XMLs in user_selections/.\n"
            "4. Aggregate selections and build tracking_review.xml with original boxes.\n"
            "5. Optionally apply Otsu or SAM2 to generate alternative boxes in tracking_review.xml.\n"
            "6. Review tracks, choose/edit boxes, then export YOLO or COCO labels.\n"
            "7. Open YOLO Training to assemble train/val sets from exported labels and run Ultralytics training."
        )
        tk.Label(stage7, text=notes, anchor=tk.NW, justify=tk.LEFT).pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        bottom = tk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(bottom, textvariable=self.status_var, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X)

    def load_project(self):
        project_dir = filedialog.askdirectory(title="Select Project Directory")
        if not project_dir:
            return
        self.project_dir = os.path.normpath(project_dir)
        self.project_var.set(f"Project: {self.project_dir}")
        resolution = resolve_cell_regions_xml(self.project_dir)
        if resolution.using_legacy_file and messagebox.askyesno(
            "Migrate Annotation File?",
            "This project uses the legacy images/cell_reigons.xml filename.\n\n"
            "Rename it to images/cell_regions.xml now? Selecting No keeps the "
            "legacy file unchanged and CelFDrive will continue to use it.",
            parent=self.root,
        ):
            resolution = migrate_legacy_cell_regions_xml(self.project_dir)
            if resolution.migrated_legacy_file:
                messagebox.showinfo(
                    "Annotation File Migrated",
                    "Renamed legacy images/cell_reigons.xml to images/cell_regions.xml.",
                    parent=self.root,
                )
        if resolution.both_files_present:
            messagebox.showwarning(
                "Duplicate Annotation Files",
                "Both images/cell_regions.xml and legacy images/cell_reigons.xml were found. "
                "CelFDrive will use cell_regions.xml; the legacy file was left unchanged for manual review.",
                parent=self.root,
            )
        self._refresh_project_status()
        self.status_var.set("Project loaded.")

    def _require_project(self):
        if not self.project_dir:
            messagebox.showerror("Project Required", "Load a project directory first.")
            return False
        return True

    def _run_with_progress_dialog(self, title, initial_text, worker_func):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("460x170+240+240")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.lift()

        state = {"current": 0, "total": 1, "image_path": "", "result": None, "error": None, "cancelled": False}

        def cancel():
            state["cancelled"] = True

        dialog.protocol("WM_DELETE_WINDOW", cancel)

        label = tk.Label(dialog, text=initial_text, anchor=tk.W, justify=tk.LEFT)
        label.pack(fill=tk.X, padx=12, pady=(12, 8))

        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(
            dialog,
            orient="horizontal",
            mode="determinate",
            maximum=1,
            variable=progress_var,
            length=420,
        )
        progress_bar.pack(fill=tk.X, padx=12, pady=(0, 8))

        progress_text = tk.Label(dialog, text="0 / 0", anchor=tk.W, justify=tk.LEFT)
        progress_text.pack(fill=tk.X, padx=12, pady=(0, 12))

        tk.Button(dialog, text="Cancel", command=cancel, width=12).pack(pady=(0, 8))
        dialog.update()

        def progress_callback(current, total, image_path):
            state["current"] = current
            state["total"] = max(1, total)
            state["image_path"] = image_path or ""

        def worker():
            try:
                state["result"] = worker_func(progress_callback)
            except Exception as exc:
                state["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while thread.is_alive() and not state["cancelled"]:
            progress_bar.configure(maximum=state["total"])
            progress_var.set(state["current"])
            if state["image_path"]:
                label.config(text=f"{initial_text}\n{os.path.basename(state['image_path'])}")
            progress_text.config(text=f"{state['current']} / {state['total']}")
            self.root.update()
            time.sleep(0.05)

        progress_bar.configure(maximum=state["total"])
        progress_var.set(state["current"])
        dialog.destroy()

        if state["cancelled"]:
            raise RuntimeError("Operation cancelled by user.")

        if state["error"] is not None:
            raise state["error"]

        return state["result"]

    def _refresh_project_status(self):
        if not self.project_dir:
            self.images_status_var.set("images/: not checked")
            self.cell_xml_status_var.set(f"{CELL_REGIONS_FILENAME}: not checked")
            self.user_selections_status_var.set("user_selections/: not checked")
            self.aggregated_status_var.set("aggregated_tracking.xml: not checked")
            self.tracking_status_var.set("tracking_review.xml: not checked")
            return

        images_dir = os.path.join(self.project_dir, "images")
        selections_dir = os.path.join(self.project_dir, "user_selections")
        resolution = resolve_cell_regions_xml(self.project_dir)
        cell_xml = resolution.path
        aggregated_xml = os.path.join(selections_dir, "aggregated_tracking.xml")
        tracking_xml = os.path.join(selections_dir, "tracking_review.xml")

        self.images_status_var.set(f"images/: {'found' if os.path.isdir(images_dir) else 'missing'}")
        if cell_xml.is_file():
            xml_status = "found (legacy file in use)" if resolution.using_legacy_file else "found"
        else:
            xml_status = "missing"
        self.cell_xml_status_var.set(f"{cell_xml.name}: {xml_status}")
        self.user_selections_status_var.set(f"user_selections/: {'found' if os.path.isdir(selections_dir) else 'missing'}")
        self.aggregated_status_var.set(f"aggregated_tracking.xml: {'found' if os.path.exists(aggregated_xml) else 'missing'}")
        self.tracking_status_var.set(f"tracking_review.xml: {'found' if os.path.exists(tracking_xml) else 'missing'}")

    def open_cell_clicker(self):
        if not self._require_project():
            return
        window = tk.Toplevel(self.root)
        ImageViewer(window, project_dir=self.project_dir)
        self.status_var.set("CellClicker opened for the loaded project.")

    def open_phase_selector(self):
        if not self._require_project():
            return
        load_ui_for_project(self.project_dir, parent=self.root)
        self._refresh_project_status()
        self.status_var.set("Phase selector completed or updated user selections.")

    def open_phase_settings(self):
        """Edit the ordered phase IDs used by this project's selector and labels."""
        if not self._require_project():
            return
        try:
            phases = load_phases(self.project_dir)
        except ValueError as exc:
            messagebox.showerror("Phase Selector Settings", str(exc), parent=self.root)
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Phase Selector Settings")
        dialog.transient(self.root)
        dialog.grab_set()
        tk.Label(
            dialog,
            text="One phase per line as ID: phase_name. IDs must start at 0 and be consecutive.\n"
            "Default: six ordered mitosis stages.",
            justify=tk.LEFT, anchor=tk.W,
        ).pack(fill=tk.X, padx=12, pady=(12, 8))
        editor = tk.Text(dialog, width=42, height=10)
        add_tooltip(
            editor,
            "Enter one ID: phase_name pair per line in chronological order. IDs must start at 0 and be consecutive.",
        )
        editor.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        editor.insert("1.0", "\n".join(f"{index}: {name}" for index, name in enumerate(phases)))

        def restore_default():
            editor.delete("1.0", tk.END)
            editor.insert("1.0", "\n".join(f"{index}: {name}" for index, name in enumerate(DEFAULT_PHASES)))

        def save():
            entries = []
            try:
                for line_number, line in enumerate(editor.get("1.0", tk.END).splitlines(), start=1):
                    if not line.strip():
                        continue
                    identifier, name = line.split(":", 1)
                    entries.append({"id": int(identifier.strip()), "name": name.strip()})
            except ValueError:
                messagebox.showerror("Phase Selector Settings", f"Line {line_number} must use `ID: phase_name`.", parent=dialog)
                return
            try:
                saved_phases = save_phases(self.project_dir, entries)
            except ValueError as exc:
                messagebox.showerror("Phase Selector Settings", str(exc), parent=dialog)
                return
            dialog.destroy()
            self.status_var.set(
                "Phase settings saved. Run phase selection, aggregation, and tracking build before exporting. "
                f"Configured phases: {', '.join(saved_phases)}."
            )
            messagebox.showinfo(
                "Phase Settings Saved",
                "Phase settings changed. Existing phase selections must be updated before aggregation and tracking rebuild.",
                parent=self.root,
            )

        buttons = tk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 12))
        add_tooltip(
            tk.Button(buttons, text="Restore Mitosis Default", command=restore_default),
            "Replace the editor contents with the standard six-stage mitosis mapping.",
        ).pack(side=tk.LEFT)
        tk.Button(buttons, text="Save", command=save).pack(side=tk.RIGHT)
        tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=8)

    def aggregate_user_selections(self):
        if not self._require_project():
            return

        selections_dir = os.path.join(self.project_dir, "user_selections")
        os.makedirs(selections_dir, exist_ok=True)
        xml_files = annotator_selection_files(selections_dir)

        if not xml_files:
            messagebox.showerror("No User XMLs", "No user selection XML files were found to aggregate.")
            return

        output_xml = os.path.join(selections_dir, "aggregated_tracking.xml")
        try:
            aggregate_xml(
                xml_files, output_xml,
                cell_regions_xml=str(resolve_cell_regions_xml(self.project_dir).path),
                phases=load_phases(self.project_dir),
                phase_signature=(
                    phase_signature(load_phases(self.project_dir))
                    if os.path.exists(settings_path(self.project_dir)) else None
                ),
            )
        except ValueError as exc:
            messagebox.showerror("Selections Need Updating", str(exc), parent=self.root)
            self.status_var.set("Aggregation blocked: phase selections need updating.")
            return
        self._refresh_project_status()
        self.status_var.set(f"Aggregated {len(xml_files)} user XML files into aggregated_tracking.xml.")
        messagebox.showinfo("Aggregation Complete", f"Created file:\n{output_xml}")

    def build_tracking_xml(self):
        if not self._require_project():
            return
        try:
            output_xml = build_tracking_xml_from_dataset(
                dataset_dir=self.project_dir,
                include_otsu=False,
                launch_ui=False,
                phases=load_phases(self.project_dir),
            )
        except ValueError as exc:
            messagebox.showerror("Build Tracking XML", str(exc), parent=self.root)
            self.status_var.set("Tracking build blocked: aggregation is stale.")
            return
        self._refresh_project_status()
        self.status_var.set(f"Built tracking XML with original boxes: {output_xml}")
        messagebox.showinfo("Build Complete", f"Created file:\n{output_xml}")

    def _require_current_tracking(self):
        tracking_xml = os.path.join(self.project_dir, "user_selections", "tracking_review.xml")
        cell_xml = str(resolve_cell_regions_xml(self.project_dir).path)
        if not os.path.isfile(tracking_xml):
            messagebox.showerror("Tracking XML Missing", "Build tracking_review.xml first.", parent=self.root)
            return None
        if not tracking_is_current(tracking_xml, cell_xml):
            messagebox.showerror(
                "Tracking Data Stale",
                "Raw CellClicker tracks or phase selections changed. Aggregate selections and rebuild tracking review first.",
                parent=self.root,
            )
            return None
        return tracking_xml

    def apply_otsu(self):
        if not self._require_project():
            return
        tracking_xml = self._require_current_tracking()
        if not tracking_xml:
            return

        overwrite = messagebox.askyesno(
            "Replace Existing Otsu Boxes",
            "Replace existing `otsu` boxes if they already exist in tracking_review.xml?",
            parent=self.root,
        )
        self.status_var.set("Running Otsu box generation...")
        self.root.update_idletasks()

        try:
            stats = self._run_with_progress_dialog(
                "Applying Otsu",
                "Generating Otsu boxes...",
                lambda progress_callback: run_otsu_on_tracking_xml(
                    tracking_xml_path=tracking_xml,
                    prompt_box_type="original",
                    overwrite=overwrite,
                    progress_callback=progress_callback,
                ),
            )
        except Exception as exc:
            LOGGER.exception("Otsu generation failed for tracking XML `%s`.", tracking_xml)
            messagebox.showerror("Otsu Failed", str(exc), parent=self.root)
            self.status_var.set("Otsu generation failed.")
            return

        self.status_var.set(
            "Otsu completed. "
            f"Processed {stats['processed']} timepoints, created {stats['created']}, "
            f"updated {stats['updated']}, skipped {stats['skipped']}, failed {stats['failed']}."
        )
        messagebox.showinfo(
            "Otsu Complete",
            (
                f"Tracks: {stats['tracks']}\n"
                f"Timepoints: {stats['timepoints']}\n"
                f"Processed: {stats['processed']}\n"
                f"Created: {stats['created']}\n"
                f"Updated: {stats['updated']}\n"
                f"Skipped: {stats['skipped']}\n"
                f"Failed: {stats['failed']}"
            ),
            parent=self.root,
        )

    def open_tracking_review(self):
        if not self._require_project():
            return
        tracking_xml = self._require_current_tracking()
        if not tracking_xml:
            return

        if self.review_window and self.review_window.winfo_exists():
            self.review_window.destroy()

        self.review_window = tk.Toplevel(self.root)
        TrackingReviewUI(self.review_window, tracking_xml_path=tracking_xml)
        self.status_var.set("Tracking review opened.")

    def check_duplicate_tracks(self):
        """Display high-overlap track pairs and safely remove a chosen duplicate."""
        if not self._require_project():
            return
        tracking_xml = self._require_current_tracking()
        if not tracking_xml:
            return
        tracking_data = read_tracking_xml(tracking_xml)
        candidates = find_near_duplicate_tracks(tracking_data.get("tracks", []))
        if not candidates:
            messagebox.showinfo(
                "Check Duplicate Tracks",
                "No near-duplicate tracks found. Candidates require matching preferred labels on at least two frames with 90% box overlap.",
                parent=self.root,
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Possible Duplicate Tracks")
        dialog.geometry("760x380+220+220")
        dialog.transient(self.root)
        dialog.grab_set()
        tk.Label(
            dialog,
            text="Review these high-overlap candidates. Select a row, then delete only the duplicate track you do not want to keep.",
            anchor=tk.W, justify=tk.LEFT,
        ).pack(fill=tk.X, padx=12, pady=(12, 8))
        columns = ("first", "second", "shared", "iou")
        table = ttk.Treeview(dialog, columns=columns, show="headings", selectmode="browse")
        table.heading("first", text="Track / Series A")
        table.heading("second", text="Track / Series B")
        table.heading("shared", text="Matching frames")
        table.heading("iou", text="Mean IoU")
        table.column("first", width=220)
        table.column("second", width=220)
        table.column("shared", width=130, anchor=tk.CENTER)
        table.column("iou", width=100, anchor=tk.CENTER)
        for index, candidate in enumerate(candidates):
            first, second = candidate["first"], candidate["second"]
            table.insert(
                "", tk.END, iid=str(index),
                values=(
                    f"{first.get('track_id')} / {first.get('series_id')}",
                    f"{second.get('track_id')} / {second.get('series_id')}",
                    candidate["shared_frames"], f"{candidate['mean_iou']:.3f}",
                ),
            )
        table.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        button_frame = tk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        def delete_selected(which):
            selected = table.selection()
            if not selected:
                messagebox.showerror("Delete Duplicate", "Select a candidate pair first.", parent=dialog)
                return
            candidate = candidates[int(selected[0])]
            target = candidate["first"] if which == "first" else candidate["second"]
            if not messagebox.askyesno(
                "Delete Duplicate Track",
                f"Delete {target.get('track_id')} (series {target.get('series_id')})?\n\n"
                "This removes that track from CellClicker, selections, aggregate, and review. Exports must be rebuilt.",
                parent=dialog,
            ):
                return
            try:
                delete_track_from_project(self.project_dir, target["source_path"], target["series_id"])
            except Exception as exc:
                messagebox.showerror("Delete Duplicate", str(exc), parent=dialog)
                return
            dialog.destroy()
            self._refresh_project_status()
            self.status_var.set(f"Deleted duplicate track {target.get('track_id')}; rebuild exports before training.")
            messagebox.showinfo("Duplicate Deleted", "Track deleted. Existing exports are stale; rebuild exports before training.", parent=self.root)

        add_tooltip(
            tk.Button(button_frame, text="Delete Track A", command=lambda: delete_selected("first")),
            "Delete Track A from the project. Existing exports will become stale.",
        ).pack(side=tk.LEFT)
        add_tooltip(
            tk.Button(button_frame, text="Delete Track B", command=lambda: delete_selected("second")),
            "Delete Track B from the project. Existing exports will become stale.",
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)

    def export_yolo_labels(self):
        if not self._require_project():
            return
        tracking_xml = self._require_current_tracking()
        if not tracking_xml:
            return

        output_dir = os.path.join(self.project_dir, "user_selections", "exported_labels")
        labels_by_file = export_tracking_xml_to_yolo(
            tracking_xml_path=tracking_xml,
            output_dir=output_dir,
            box_type=self.export_box_type_var.get(),
        )
        self.status_var.set(f"Exported {len(labels_by_file)} YOLO label files to {output_dir}.")
        messagebox.showinfo("Export Complete", f"Exported {len(labels_by_file)} YOLO label files.")

    def view_exported_dataset(self):
        """Open a read-only preview of the current YOLO dataset export."""
        if not self._require_project():
            return
        if not exported_labels_are_current(self.project_dir):
            messagebox.showerror(
                "Export Required",
                "Export current YOLO labels before opening the final dataset viewer.",
            )
            return
        try:
            ExportedDatasetViewer(self.root, self.project_dir)
        except (FileNotFoundError, OSError, ValueError) as exc:
            messagebox.showerror("Cannot Open Exported Dataset", str(exc))

    def export_coco_labels(self):
        if not self._require_project():
            return
        tracking_xml = self._require_current_tracking()
        if not tracking_xml:
            return

        output_json_path = os.path.join(self.project_dir, "user_selections", "exported_coco", "annotations.json")
        coco_data = export_tracking_xml_to_coco(
            tracking_xml_path=tracking_xml,
            output_json_path=output_json_path,
            box_type=self.export_box_type_var.get(),
            replace_output_directory=True,
        )
        self.status_var.set(
            f"Exported COCO annotations with {len(coco_data['annotations'])} boxes to {output_json_path}."
        )
        messagebox.showinfo(
            "Export Complete",
            f"Exported {len(coco_data['annotations'])} COCO annotations.",
        )

    def export_miniseries(self):
        if not self._require_project():
            return
        tracking_xml = self._require_current_tracking()
        if not tracking_xml:
            return

        output_dir = os.path.join(self.project_dir, "miniseries")
        exported_images = export_tracking_xml_to_miniseries(
            tracking_xml_path=tracking_xml,
            output_dir=output_dir,
            box_type="preferred",
            padding_ratio=0.10,
        )
        self.status_var.set(f"Exported {len(exported_images)} miniseries crops to {output_dir}.")
        messagebox.showinfo(
            "Export Complete",
            f"Exported {len(exported_images)} miniseries crops.\n\nOutput folder:\n{output_dir}",
        )

    def run_sam2(self):
        if not self._require_project():
            return

        tracking_xml = self._require_current_tracking()
        if not tracking_xml:
            return

        model_name = simpledialog.askstring(
            "SAM2 Model",
            "Ultralytics SAM2 checkpoint name or path:",
            initialvalue=DEFAULT_SAM2_MODEL,
            parent=self.root,
        )
        if not model_name:
            return

        run_mode = messagebox.askyesnocancel(
            "SAM2 Progress Handling",
            "Yes: resume and keep existing SAM2 boxes.\n"
            "No: start from the beginning and overwrite existing SAM2 boxes.\n"
            "Cancel: abort.",
            parent=self.root,
        )
        if run_mode is None:
            return
        overwrite = not run_mode

        self.status_var.set(f"Running SAM2 with {model_name} on {DEFAULT_SAM2_DEVICE}...")
        self.root.update_idletasks()

        try:
            stats = self._run_with_progress_dialog(
                "Running SAM2",
                f"Generating SAM2 boxes with {model_name} on {DEFAULT_SAM2_DEVICE}...",
                lambda progress_callback: run_sam2_on_tracking_xml(
                    tracking_xml_path=tracking_xml,
                    model_name=model_name,
                    prompt_box_type="original",
                    overwrite=overwrite,
                    progress_callback=progress_callback,
                    device=DEFAULT_SAM2_DEVICE,
                    save_every=1,
                ),
            )
        except Exception as exc:
            LOGGER.exception("SAM2 run failed for tracking XML `%s`.", tracking_xml)
            messagebox.showerror("SAM2 Failed", str(exc), parent=self.root)
            self.status_var.set("SAM2 run failed.")
            return

        self.status_var.set(
            "SAM2 completed. "
            f"Processed {stats['processed']} timepoints, created {stats['created']}, "
            f"updated {stats['updated']}, skipped {stats['skipped']}, failed {stats['failed']}."
        )
        messagebox.showinfo(
            "SAM2 Complete",
            (
                f"Tracks: {stats['tracks']}\n"
                f"Timepoints: {stats['timepoints']}\n"
                f"Processed: {stats['processed']}\n"
                f"Created: {stats['created']}\n"
                f"Updated: {stats['updated']}\n"
                f"Skipped: {stats['skipped']}\n"
                f"Failed: {stats['failed']}"
            ),
            parent=self.root,
        )

    def configure_tightener(self):
        """Save the trained tightener checkpoint selected for this project."""
        if not self._require_project():
            return
        tracking_xml = self._require_current_tracking()
        if not tracking_xml:
            return
        weights_path = filedialog.askopenfilename(
            title="Select Cell Tightener best.pt", initialdir=DEFAULT_TIGHTENER_MODELS_ROOT,
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")], parent=self.root,
        )
        if not weights_path:
            return
        current_selection = DEFAULT_TIGHTENER_SELECTION
        try:
            current_selection = read_tracking_xml(tracking_xml).get("metadata", {}).get(
                TIGHTENER_SELECTION_METADATA_KEY, DEFAULT_TIGHTENER_SELECTION
            )
        except Exception:
            pass
        selection_strategy = simpledialog.askstring(
            "Tightener Prediction Selection",
            "Select `center_confidence` (recommended), `overlap`, or `confidence`:\n"
            "center_confidence chooses the highest-confidence box containing the original-box centre, then falls back to overlap.",
            initialvalue=current_selection,
            parent=self.root,
        )
        if selection_strategy is None:
            return
        selection_strategy = selection_strategy.strip().lower()
        try:
            configure_tightener_weights(tracking_xml, weights_path, selection_strategy=selection_strategy)
        except Exception as exc:
            messagebox.showerror("Configure Tightener", str(exc), parent=self.root)
            return
        self.status_var.set(f"Configured cell tightener ({selection_strategy} selection): {weights_path}")

    def run_tightener(self):
        if not self._require_project():
            return
        tracking_xml = self._require_current_tracking()
        if not tracking_xml:
            return
        try:
            stats = self._run_with_progress_dialog(
                "Running YOLO11 Cell Tightener", "Generating trained cell-tightener box variants...",
                lambda progress_callback: run_tightener_on_tracking_xml(tracking_xml, progress_callback=progress_callback),
            )
        except Exception as exc:
            LOGGER.exception("Cell tightener failed for `%s`.", tracking_xml)
            messagebox.showerror("YOLO11 Cell Tightener Failed", str(exc), parent=self.root)
            self.status_var.set("YOLO11 cell tightener failed.")
            return
        self.status_var.set(
            f"YOLO11 cell tightener completed. Created {stats['created']}, updated {stats['updated']}, "
            f"original fallbacks {stats['fallback_original']}, failed {stats['failed']}."
        )
        messagebox.showinfo("YOLO11 Tightener Complete", "\n".join(f"{key}: {value}" for key, value in stats.items()), parent=self.root)

    def open_yolo_training(self):
        window = tk.Toplevel(self.root)
        app = YOLOTrainingUI(window)
        if self.project_dir:
            normalized = os.path.normpath(self.project_dir)
            app.train_projects.append(normalized)
            app.train_listbox.insert(tk.END, normalized)
            app.refresh_counts()
            app.status_var.set("Loaded project added to the training list as a starting point.")

    def open_tightener_training(self):
        window = tk.Toplevel(self.root)
        app = MitoticTightenerTrainingUI(window)
        if self.project_dir:
            app.add_project_to_split("train", self.project_dir)
            app.refresh_counts()


def launch_project_gui():
    """Create the root Tk window, run the project GUI, and return its instance."""
    root = tk.Tk()
    app = ProjectGUI(root)
    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
    return app
