"""Tkinter editor for the repo-local CelFDrive prediction YAML config.

This module provides a small GUI for editing ``celfdrive_predict.yaml``.
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "celfdrive_predict.yaml"
CONFIG_DIR = REPO_ROOT / "Configs"
CONFIG_PATH = DEFAULT_CONFIG_PATH


class ScrollableFrame(ttk.Frame):
    """Frame with a vertical scrollbar for forms that exceed the window height.

    Args:
        parent (tk.Widget): Parent widget that owns the scrollable frame.

    Attributes:
        canvas (tk.Canvas): Canvas used to host the scrollable content.
        scrollbar (ttk.Scrollbar): Vertical scrollbar bound to the canvas.
        content (ttk.Frame): Child frame where form controls are placed.
    """

    def __init__(self, parent):
        """Create the canvas-backed scrollable frame.

        Args:
            parent (tk.Widget): Parent widget that owns the scrollable frame.

        Returns:
            None
        """
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)

        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._update_content_width)

    def _update_scroll_region(self, _event):
        """Resize the scrollable region when form content changes.

        Args:
            _event (tk.Event): Tkinter configure event. Unused.

        Returns:
            None
        """
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _update_content_width(self, event):
        """Keep the inner form width matched to the visible canvas width.

        Args:
            event (tk.Event): Canvas configure event containing the new width.

        Returns:
            None
        """
        self.canvas.itemconfigure(self.window_id, width=event.width)


class ConfigEditor:
    """GUI for editing celfdrive_predict.yaml without hand-editing YAML.

    Args:
        root (tk.Tk): Root Tkinter window.
        config_path (Path | str | None): Initial editable YAML config path.

    Attributes:
        root (tk.Tk): Root Tkinter window.
        config_path (Path): YAML config path.
        config (dict): In-memory YAML config data.
        vars (dict[tuple, tuple[tk.Variable, type]]): Form variables keyed by config path.
    """

    def __init__(self, root, config_path=None):
        """Load the config and build the editor window.

        Args:
            root (tk.Tk): Root Tkinter window.
            config_path (Path | str | None): Initial editable YAML config path. If ``None``, the
                first YAML file in ``Configs`` is used.

        Returns:
            None
        """
        self.root = root
        self.ensure_config_folder()
        self.config_path = Path(config_path) if config_path is not None else self.get_initial_config_path()
        self.root.title("CelFDrive Prediction Config")
        self.root.geometry("980x720")

        self.config = self.load_config()
        self.vars = {}

        self.build_ui()

    def ensure_config_folder(self):
        """Create the config folder and seed it from the active default if needed.

        Args:
            None

        Returns:
            None
        """
        CONFIG_DIR.mkdir(exist_ok=True)
        if not list(CONFIG_DIR.glob("*.yaml")) and DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
            with open(CONFIG_DIR / "default.yaml", "w", encoding="utf-8") as file:
                yaml.safe_dump(config_data, file, sort_keys=False)

    def get_config_files(self):
        """Return editable YAML config files in the repo config folder.

        Args:
            None

        Returns:
            list[Path]: Sorted list of ``*.yaml`` files in ``Configs``.
        """
        return sorted(CONFIG_DIR.glob("*.yaml"))

    def get_initial_config_path(self):
        """Choose the initial editable config path.

        Args:
            None

        Returns:
            Path: First YAML config file in ``Configs``.

        Raises:
            FileNotFoundError: If no editable configs are available.
        """
        config_files = self.get_config_files()
        if not config_files:
            raise FileNotFoundError(f"No YAML configs found in {CONFIG_DIR}")
        return config_files[0]

    def load_config(self):
        """Read the YAML config from disk.

        Args:
            None

        Returns:
            dict: Parsed YAML config.
        """
        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def build_ui(self):
        """Create the notebook tabs and bottom action bar.

        Args:
            None

        Returns:
            None
        """
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        config_bar = ttk.Frame(self.root)
        config_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        config_bar.columnconfigure(1, weight=1)

        config_names = [path.name for path in self.get_config_files()]
        self.config_name_var = tk.StringVar(value=self.config_path.name)
        ttk.Label(config_bar, text="Config").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.config_selector = ttk.Combobox(
            config_bar,
            textvariable=self.config_name_var,
            values=config_names,
            state="readonly",
        )
        self.config_selector.grid(row=0, column=1, sticky="ew")
        ttk.Button(config_bar, text="Load", command=self.load_selected_config).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(config_bar, text="Save As", command=self.save_as_config).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(config_bar, text="Set as default", command=self.set_default_config).grid(row=0, column=4, padx=(8, 0))

        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        general = self.add_tab(self.main_notebook, "General")
        preprocessing = self.add_tab(self.main_notebook, "Image")
        coordinates = self.add_tab(self.main_notebook, "Coordinates")
        profiles = self.add_tab(self.main_notebook, "Profiles")

        self.build_general_tab(general)
        self.build_image_tab(preprocessing)
        self.build_coordinates_tab(coordinates)
        self.build_profiles_tab(profiles)

        action_bar = ttk.Frame(self.root)
        action_bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        action_bar.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value=f"Editing {self.config_path}. Active default is {DEFAULT_CONFIG_PATH.name}.")
        ttk.Label(action_bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(action_bar, text="Reload", command=self.reload).grid(row=0, column=1, padx=4)
        ttk.Button(action_bar, text="Save", command=self.save).grid(row=0, column=2, padx=4)
        ttk.Button(action_bar, text="Close", command=self.root.destroy).grid(row=0, column=3, padx=4)

    def add_tab(self, notebook, title):
        """Add a scrollable tab to the main notebook.

        Args:
            notebook (ttk.Notebook): Notebook that receives the tab.
            title (str): Text displayed on the tab.

        Returns:
            ttk.Frame: Content frame where controls should be added.
        """
        frame = ScrollableFrame(notebook)
        notebook.add(frame, text=title)
        return frame.content

    def get_value(self, path):
        """Return a nested config value addressed by a list path.

        Args:
            path (list[str | int]): Keys used to walk through ``self.config``.

        Returns:
            object: Nested config value.
        """
        value = self.config
        for key in path:
            value = value[key]
        return value

    def set_value(self, path, value):
        """Set a nested config value addressed by a list path.

        Args:
            path (list[str | int]): Keys used to walk through ``self.config``.
            value (object): Replacement value to store.

        Returns:
            None
        """
        target = self.config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    def make_var(self, path, value_type):
        """Create and register a Tk variable for one config value.

        Args:
            path (list[str | int]): Config path for the value.
            value_type (type): Expected Python type for parsing on save.

        Returns:
            tk.BooleanVar | tk.StringVar: Tk variable bound to the form control.
        """
        value = self.get_value(path)
        if value_type == bool:
            var = tk.BooleanVar(value=bool(value))
        else:
            var = tk.StringVar(value=str(value))
        self.vars[tuple(path)] = (var, value_type)
        return var

    def add_section(self, parent, title, row):
        """Add a labelled section frame to a tab.

        Args:
            parent (tk.Widget): Parent container.
            title (str): Label frame title.
            row (int): Grid row in the parent.

        Returns:
            ttk.LabelFrame: Section frame.
        """
        section = ttk.LabelFrame(parent, text=title, padding=10)
        section.grid(row=row, column=0, sticky="ew", padx=8, pady=8)
        section.columnconfigure(1, weight=1)
        return section

    def add_field(self, parent, row, label, path, value_type=str, browse=None):
        """Add a labelled input bound to a config path.

        Args:
            parent (tk.Widget): Parent container.
            row (int): Grid row in the parent.
            label (str): Human-readable field label.
            path (list[str | int]): Config path for the value.
            value_type (type): Expected type. Supports ``str``, ``int``, ``float``, and ``bool``.
            browse (str | None): ``"dir"`` or ``"file"`` to add a browse button.

        Returns:
            None
        """
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        var = self.make_var(path, value_type)

        if value_type == bool:
            widget = ttk.Checkbutton(parent, variable=var)
            widget.grid(row=row, column=1, sticky="w", pady=4)
        else:
            widget = ttk.Entry(parent, textvariable=var)
            widget.grid(row=row, column=1, sticky="ew", pady=4)

        if browse:
            command = lambda: self.browse_path(var, browse)
            ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, padx=(8, 0), pady=4)

    def add_output_range_fields(self, parent, row):
        """Add paired min/max inputs for preprocessing.output_range.

        Args:
            parent (tk.Widget): Parent container.
            row (int): Grid row in the parent.

        Returns:
            None
        """
        output_range = self.get_value(["preprocessing", "output_range"])
        self.vars[("preprocessing", "output_range", 0)] = (tk.StringVar(value=str(output_range[0])), float)
        self.vars[("preprocessing", "output_range", 1)] = (tk.StringVar(value=str(output_range[1])), float)

        ttk.Label(parent, text="Output range").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        range_frame = ttk.Frame(parent)
        range_frame.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Entry(range_frame, width=10, textvariable=self.vars[("preprocessing", "output_range", 0)][0]).grid(row=0, column=0)
        ttk.Label(range_frame, text="to").grid(row=0, column=1, padx=6)
        ttk.Entry(range_frame, width=10, textvariable=self.vars[("preprocessing", "output_range", 1)][0]).grid(row=0, column=2)

    def browse_path(self, var, browse):
        """Open a file or directory picker and write the selected path to a variable.

        Args:
            var (tk.StringVar): Variable that receives the selected path.
            browse (str): ``"dir"`` to select a directory, otherwise selects a file.

        Returns:
            None
        """
        initial = var.get()
        if browse == "dir":
            selected = filedialog.askdirectory(initialdir=initial if initial else None)
        else:
            selected = filedialog.askopenfilename(initialdir=str(Path(initial).parent) if initial else None)
        if selected:
            var.set(selected.replace("\\", "/"))

    def get_selected_config_path(self):
        """Return the config path selected in the top combobox.

        Args:
            None

        Returns:
            Path: Selected config path inside ``Configs``.
        """
        return CONFIG_DIR / self.config_name_var.get()

    def load_selected_config(self):
        """Load the selected editable config from the ``Configs`` folder.

        Args:
            None

        Returns:
            None
        """
        selected_path = self.get_selected_config_path()
        if not selected_path.exists():
            messagebox.showerror("Load failed", f"Config does not exist: {selected_path}")
            return

        self.config_path = selected_path
        self.config = self.load_config()
        self.rebuild_ui()

    def write_config(self, path):
        """Write the in-memory config to a YAML file.

        Args:
            path (Path | str): Destination YAML path.

        Returns:
            None
        """
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file, sort_keys=False)

    def save_as_config(self):
        """Save the current form to a new editable config in ``Configs``.

        Args:
            None

        Returns:
            None
        """
        try:
            self.apply_vars_to_config()
        except Exception as exc:
            messagebox.showerror("Save As failed", str(exc))
            return

        selected = filedialog.asksaveasfilename(
            initialdir=CONFIG_DIR,
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
        )
        if not selected:
            return

        selected_path = Path(selected)
        if selected_path.suffix.lower() not in [".yaml", ".yml"]:
            selected_path = selected_path.with_suffix(".yaml")
        if selected_path.parent.resolve() != CONFIG_DIR.resolve():
            messagebox.showerror("Save As failed", f"Configs must be saved in {CONFIG_DIR}")
            return

        self.config_path = selected_path
        self.write_config(self.config_path)
        self.rebuild_ui()
        self.status_var.set(f"Saved {self.config_path}")

    def set_default_config(self):
        """Save the selected config and copy it to root celfdrive_predict.yaml.

        Args:
            None

        Returns:
            None
        """
        try:
            selected_path = self.get_selected_config_path()
            if selected_path != self.config_path:
                self.config_path = selected_path
                self.config = self.load_config()
            else:
                self.apply_vars_to_config()
                self.write_config(self.config_path)
            self.write_config(DEFAULT_CONFIG_PATH)
        except Exception as exc:
            messagebox.showerror("Set default failed", str(exc))
            return

        self.status_var.set(f"Set {self.config_path.name} as default")
        messagebox.showinfo("Default updated", f"Wrote {self.config_path.name} to {DEFAULT_CONFIG_PATH.name}")
        self.rebuild_ui()

    def build_general_tab(self, parent):
        """Build controls for project, model, and logging config.

        Args:
            parent (tk.Widget): Tab content frame.

        Returns:
            None
        """
        project = self.add_section(parent, "Project And Model", 0)
        self.add_field(project, 0, "Repo path", ["project", "repo_path"], browse="dir")
        self.add_field(project, 1, "Weights path", ["model", "weights_path"], browse="file")
        self.add_field(project, 2, "Backend", ["model", "backend"])
        self.add_field(project, 3, "Suppress model stdout", ["model", "suppress_stdout"], bool)

        logging = self.add_section(parent, "Logging", 1)
        self.add_field(logging, 0, "Root directory", ["logging", "root_dir"], browse="dir")
        self.add_field(logging, 1, "Use date subfolder", ["logging", "use_date_subfolder"], bool)
        self.add_field(logging, 2, "Date format", ["logging", "date_format"])
        self.add_field(logging, 3, "Experiment prefix", ["logging", "experiment_folder", "prefix"])
        self.add_field(logging, 4, "Experiment digits", ["logging", "experiment_folder", "digits"], int)
        self.add_field(logging, 5, "Output image prefix", ["logging", "output_image", "prefix"])
        self.add_field(logging, 6, "Output image digits", ["logging", "output_image", "digits"], int)
        self.add_field(logging, 7, "Output image extension", ["logging", "output_image", "extension"])

    def build_image_tab(self, parent):
        """Build controls for preprocessing, tiling, and plotting config.

        Args:
            parent (tk.Widget): Tab content frame.

        Returns:
            None
        """
        preprocessing = self.add_section(parent, "Preprocessing", 0)
        self.add_field(preprocessing, 0, "Input channel mode", ["preprocessing", "input_channel", "mode"])
        self.add_field(preprocessing, 1, "Top clip percentile", ["preprocessing", "top_clip_percentile"], float)
        self.add_field(preprocessing, 2, "Normalize min/max", ["preprocessing", "normalize_min_max"], bool)
        self.add_field(preprocessing, 3, "Output dtype", ["preprocessing", "output_dtype"])
        self.add_output_range_fields(preprocessing, 4)
        self.add_field(preprocessing, 5, "Repeat grayscale to RGB", ["preprocessing", "repeat_grayscale_to_rgb"], bool)

        tiling = self.add_section(parent, "Tiling", 1)
        self.add_field(tiling, 0, "Enabled", ["tiling", "enabled"], bool)
        self.add_field(tiling, 1, "Tile size px", ["tiling", "tile_size_px"], int)
        self.add_field(tiling, 2, "Edge mode", ["tiling", "edge_mode"])
        self.add_field(tiling, 3, "Overlap px", ["tiling", "overlap_px"], int)

        plotting = self.add_section(parent, "Plotting", 2)
        self.add_field(plotting, 0, "Enabled", ["plotting", "enabled"], bool)
        self.add_field(plotting, 1, "Color map", ["plotting", "cmap"])
        self.add_field(plotting, 2, "Box edge color", ["plotting", "bbox", "edge_color"])
        self.add_field(plotting, 3, "Box line width", ["plotting", "bbox", "line_width"], float)
        self.add_field(plotting, 4, "Label font size", ["plotting", "label", "font_size"], int)
        self.add_field(plotting, 5, "Label text color", ["plotting", "label", "text_color"])
        self.add_field(plotting, 6, "Label background color", ["plotting", "label", "background_color"])
        self.add_field(plotting, 7, "Label background alpha", ["plotting", "label", "background_alpha"], float)

    def build_coordinates_tab(self, parent):
        """Build controls for coordinate conversion and no-detection config.

        Args:
            parent (tk.Widget): Tab content frame.

        Returns:
            None
        """
        conversion = self.add_section(parent, "Coordinate Conversion", 0)
        self.add_field(conversion, 0, "Default z offset um", ["coordinate_conversion", "default_z_offset_um"], float)
        self.add_field(conversion, 1, "Merge tolerance um", ["coordinate_conversion", "merge_tolerance_um"], float)
        self.add_field(conversion, 2, "Stage direction x", ["coordinate_conversion", "stage_direction", "x"], int)
        self.add_field(conversion, 3, "Stage direction y", ["coordinate_conversion", "stage_direction", "y"], int)
        self.add_field(conversion, 4, "Stage direction z", ["coordinate_conversion", "stage_direction", "z"], int)
        self.add_field(conversion, 5, "LLSM invert y stage", ["coordinate_conversion", "llsm", "invert_y_stage_direction"], bool)

        no_detection = self.add_section(parent, "No Detection", 1)
        self.add_field(no_detection, 0, "Returned locations", ["no_detection", "n_returned_locations"], int)
        self.add_field(no_detection, 1, "Script", ["no_detection", "script"])
        self.add_field(no_detection, 2, "Name", ["no_detection", "name"])
        self.add_field(no_detection, 3, "Comment", ["no_detection", "comment"])
        self.add_field(no_detection, 4, "Return original first position", ["no_detection", "return_original_first_position"], bool)

    def build_profiles_tab(self, parent):
        """Build the profile notebook for all configured capture profiles.

        Args:
            parent (tk.Widget): Tab content frame.

        Returns:
            None
        """
        self.profile_notebook = ttk.Notebook(parent)
        self.profile_notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        parent.columnconfigure(0, weight=1)
        self.profile_tab_names = []

        for profile_name in self.config["profiles"]:
            frame = ttk.Frame(self.profile_notebook, padding=10)
            frame.columnconfigure(0, weight=1)
            self.profile_notebook.add(frame, text=profile_name)
            self.profile_tab_names.append(profile_name)
            self.build_profile(frame, profile_name)

    def build_profile(self, parent, profile_name):
        """Build controls for one capture profile and its class table.

        Args:
            parent (tk.Widget): Profile tab frame.
            profile_name (str): Key of the profile in ``config["profiles"]``.

        Returns:
            None
        """
        profile = self.add_section(parent, "Capture", 0)
        base = ["profiles", profile_name]
        self.add_field(profile, 0, "Description", base + ["description"])
        self.add_field(profile, 1, "LLSM profile", base + ["llsm"], bool)
        self.add_field(profile, 2, "Highres script", base + ["highres_script"])
        self.add_field(profile, 3, "Highres comment", base + ["highres_comment"])
        self.add_field(profile, 4, "Name template", base + ["name_template"])

        classes = self.add_section(parent, "Classes", 1)
        headings = ["ID", "Name", "Confidence threshold", "Priority rank", ""]
        for col, heading in enumerate(headings):
            ttk.Label(classes, text=heading).grid(row=0, column=col, sticky="w", padx=4, pady=(0, 6))
        classes.columnconfigure(1, weight=1)

        class_configs = self.config["profiles"][profile_name]["classes"]
        for row, class_id in enumerate(sorted(class_configs.keys(), key=int), start=1):
            ttk.Label(classes, text=str(class_id)).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            class_base = base + ["classes", class_id]
            name_var = self.make_var(class_base + ["name"], str)
            threshold_var = self.make_var(class_base + ["confidence_threshold"], float)
            rank_var = self.make_var(class_base + ["priority_rank"], int)
            ttk.Entry(classes, textvariable=name_var).grid(row=row, column=1, sticky="ew", padx=4, pady=3)
            ttk.Entry(classes, width=18, textvariable=threshold_var).grid(row=row, column=2, sticky="ew", padx=4, pady=3)
            ttk.Entry(classes, width=14, textvariable=rank_var).grid(row=row, column=3, sticky="ew", padx=4, pady=3)
            ttk.Button(
                classes,
                text="Remove",
                command=lambda profile_name=profile_name, class_id=class_id: self.remove_class(profile_name, class_id),
            ).grid(row=row, column=4, sticky="ew", padx=4, pady=3)

        add_row = len(class_configs) + 1
        ttk.Button(
            classes,
            text="Add class",
            command=lambda profile_name=profile_name: self.add_class(profile_name),
        ).grid(row=add_row, column=0, columnspan=5, sticky="w", padx=4, pady=(8, 0))

    def apply_vars_to_config(self):
        """Copy all current form values into the in-memory config dictionary.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: If a form value cannot be converted to its expected type.
        """
        for path, (var, value_type) in self.vars.items():
            if path[:2] == ("preprocessing", "output_range"):
                continue
            self.set_value(list(path), self.parse_var(var, value_type, path))

        output_min = self.parse_var(*self.vars[("preprocessing", "output_range", 0)], ("preprocessing", "output_range", 0))
        output_max = self.parse_var(*self.vars[("preprocessing", "output_range", 1)], ("preprocessing", "output_range", 1))
        self.config["preprocessing"]["output_range"] = [output_min, output_max]

    def rebuild_ui(self, selected_main_tab=None, selected_profile=None):
        """Recreate the full form from the current in-memory config.

        Args:
            selected_main_tab (str | None): Main tab title to select after rebuilding.
            selected_profile (str | None): Profile tab key to select after rebuilding.

        Returns:
            None
        """
        self.vars.clear()
        for child in self.root.winfo_children():
            child.destroy()
        self.build_ui()
        self.select_tabs(selected_main_tab, selected_profile)

    def select_tabs(self, selected_main_tab=None, selected_profile=None):
        """Select main/profile tabs after rebuilding the UI.

        Args:
            selected_main_tab (str | None): Main tab title to select.
            selected_profile (str | None): Profile tab key to select.

        Returns:
            None
        """
        if selected_main_tab is not None:
            for tab_id in self.main_notebook.tabs():
                if self.main_notebook.tab(tab_id, "text") == selected_main_tab:
                    self.main_notebook.select(tab_id)
                    break

        if selected_profile is not None and hasattr(self, "profile_notebook"):
            for tab_id in self.profile_notebook.tabs():
                if self.profile_notebook.tab(tab_id, "text") == selected_profile:
                    self.profile_notebook.select(tab_id)
                    break

    def add_class(self, profile_name):
        """Add a new class to a profile using the next available numeric class ID.

        Args:
            profile_name (str): Profile key under ``config["profiles"]``.

        Returns:
            None
        """
        try:
            self.apply_vars_to_config()
        except Exception as exc:
            messagebox.showerror("Cannot add class", str(exc))
            return

        classes = self.config["profiles"][profile_name]["classes"]
        numeric_ids = [int(class_id) for class_id in classes.keys()]
        next_class_id = max(numeric_ids, default=-1) + 1
        classes[next_class_id] = {
            "name": f"class_{next_class_id}",
            "confidence_threshold": 0.01,
            "priority_rank": next_class_id,
        }
        self.rebuild_ui(selected_main_tab="Profiles", selected_profile=profile_name)

    def remove_class(self, profile_name, class_id):
        """Remove one class from a profile after confirmation.

        Args:
            profile_name (str): Profile key under ``config["profiles"]``.
            class_id (int | str): Class ID key to remove.

        Returns:
            None
        """
        try:
            self.apply_vars_to_config()
        except Exception as exc:
            messagebox.showerror("Cannot remove class", str(exc))
            return

        classes = self.config["profiles"][profile_name]["classes"]
        if len(classes) <= 1:
            messagebox.showwarning("Cannot remove class", "Each profile must keep at least one class.")
            return

        class_name = classes[class_id].get("name", class_id)
        should_remove = messagebox.askyesno(
            "Remove class",
            f"Remove class {class_id}: {class_name} from the {profile_name} profile?",
        )
        if not should_remove:
            return

        del classes[class_id]
        self.rebuild_ui(selected_main_tab="Profiles", selected_profile=profile_name)

    def parse_var(self, var, value_type, path):
        """Convert a Tk variable into the expected Python type.

        Args:
            var (tk.Variable): Tk variable containing the raw form value.
            value_type (type): Expected Python type.
            path (tuple[str | int, ...]): Config path used for error messages.

        Returns:
            bool | int | float | str: Parsed value.

        Raises:
            ValueError: If conversion to ``value_type`` fails.
        """
        if value_type == bool:
            return bool(var.get())

        raw_value = var.get()
        try:
            if value_type == int:
                return int(raw_value)
            if value_type == float:
                return float(raw_value)
            return raw_value
        except ValueError as exc:
            label = ".".join(str(part) for part in path)
            raise ValueError(f"{label} must be a {value_type.__name__}") from exc

    def save(self):
        """Validate the form and write the YAML config back to disk.

        Args:
            None

        Returns:
            None
        """
        try:
            self.apply_vars_to_config()
            self.write_config(self.config_path)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.status_var.set(f"Saved {self.config_path}")
        messagebox.showinfo("Saved", f"Updated {self.config_path}")

    def reload(self):
        """Reload the YAML config from disk and rebuild the editor.

        Args:
            None

        Returns:
            None
        """
        self.config = self.load_config()
        self.rebuild_ui()


def main():
    """Launch the CelFDrive prediction config editor.

    Args:
        None

    Returns:
        None
    """
    root = tk.Tk()
    ConfigEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
