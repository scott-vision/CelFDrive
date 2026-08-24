"""Tkinter editor for the repo-local CelFDrive prediction YAML config.

This module provides a small GUI for editing ``celfdrive_predict.yaml``.
"""

from pathlib import Path
import math
from numbers import Real
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml

from predict import migrate_predict_config


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "celfdrive_predict.yaml"
CONFIG_DIR = REPO_ROOT / "Configs"
CONFIG_PATH = DEFAULT_CONFIG_PATH
SLIDEBOOK_SCRIPT_PATH = REPO_ROOT / "SlideBook" / "CelFDrive.sbs"


def render_slidebook_script(config):
    """Render the direct-Python SlideBook macro from a CelFDrive config."""
    slidebook = config["slidebook"]
    environment = _slidebook_string(slidebook["python_environment"], "slidebook.python_environment")
    search_objective = _optional_slidebook_string(
        slidebook["objective_before_target_search"], "slidebook.objective_before_target_search"
    )
    objective = _slidebook_string(slidebook["highres_objective"], "slidebook.highres_objective")
    repo_path = Path(config["project"]["repo_path"])
    if not repo_path.is_absolute():
        repo_path = REPO_ROOT / repo_path
    python_path = repo_path.resolve().as_posix()
    commands = [
        f'Python_SetEnvironment(Environment = "{environment}", UseThread = true)',
        'Python_RunCommand(Command="import sys")',
        f'Python_RunCommand(Command="sys.path.insert(0, r\'{python_path}\')")',
    ]
    if search_objective:
        commands.append(f'ChangeObjective(Objective = "{search_objective}")')
    commands.extend(
        [
            'Python_RunHierarchicalCaptureFunction(<current image>, Function = "find_locations_of_interest_montage.py")',
            f'ChangeObjective(Objective = "{objective}")',
            "Run6DCapture()",
            "",
        ]
    )
    return "\n".join(commands)


def _slidebook_string(value, name):
    _non_empty_string(value, name)
    if '"' in value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} cannot contain quotes or line breaks")
    return value


def _optional_slidebook_string(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value:
        return ""
    return _slidebook_string(value, name)


def validate_prediction_config(config):
    """Validate every persisted prediction setting before it is saved.

    Raises:
        ValueError: If a setting is incompatible with the prediction workflow.
    """
    if not isinstance(config, dict):
        raise ValueError("Prediction configuration must be a mapping")

    project = _mapping(config.get("project"), "project")
    _non_empty_string(project.get("repo_path"), "project.repo_path")

    model = _mapping(config.get("model"), "model")
    _non_empty_string(model.get("weights_path"), "model.weights_path")
    if model.get("backend") != "ultralytics_yolo":
        raise ValueError("model.backend must be 'ultralytics_yolo'")
    _boolean(model.get("suppress_stdout"), "model.suppress_stdout")

    logging = _mapping(config.get("logging"), "logging")
    _boolean(logging.get("enabled"), "logging.enabled")
    _non_empty_string(logging.get("root_dir"), "logging.root_dir")
    _boolean(logging.get("use_date_subfolder"), "logging.use_date_subfolder")
    _non_empty_string(logging.get("date_format"), "logging.date_format")
    experiment_folder = _mapping(logging.get("experiment_folder"), "logging.experiment_folder")
    _non_empty_string(experiment_folder.get("prefix"), "logging.experiment_folder.prefix")
    _positive_integer(experiment_folder.get("digits"), "logging.experiment_folder.digits")
    output_image = _mapping(logging.get("output_image"), "logging.output_image")
    _non_empty_string(output_image.get("prefix"), "logging.output_image.prefix")
    _positive_integer(output_image.get("digits"), "logging.output_image.digits")
    extension = output_image.get("extension")
    if not isinstance(extension, str) or not extension.startswith(".") or len(extension) == 1:
        raise ValueError("logging.output_image.extension must be a file extension beginning with '.'")

    preprocessing = _mapping(config.get("preprocessing"), "preprocessing")
    input_channel = _mapping(preprocessing.get("input_channel"), "preprocessing.input_channel")
    if input_channel.get("mode") != "first_channel_if_rgb":
        raise ValueError("preprocessing.input_channel.mode must be 'first_channel_if_rgb'")
    top_clip_percentile = _finite_float(
        preprocessing.get("top_clip_percentile"), "preprocessing.top_clip_percentile"
    )
    if not 0 <= top_clip_percentile < 100:
        raise ValueError("preprocessing.top_clip_percentile must be in [0, 100)")
    _boolean(preprocessing.get("normalize_min_max"), "preprocessing.normalize_min_max")

    inference = _mapping(config.get("inference"), "inference")
    if inference.get("mode") not in {"standard", "sahi"}:
        raise ValueError("inference.mode must be 'standard' or 'sahi'")
    sahi = _mapping(inference.get("sahi"), "inference.sahi")
    confidence_threshold = _finite_float(
        sahi.get("confidence_threshold"), "inference.sahi.confidence_threshold"
    )
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("inference.sahi.confidence_threshold must be between 0 and 1")
    _positive_integer(sahi.get("slice_size_px"), "inference.sahi.slice_size_px")
    overlap_ratio = _finite_float(sahi.get("overlap_ratio"), "inference.sahi.overlap_ratio")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("inference.sahi.overlap_ratio must be in [0, 1)")
    _positive_integer(sahi.get("tile_batch_size"), "inference.sahi.tile_batch_size")
    merge_iou_threshold = _finite_float(
        sahi.get("merge_iou_threshold"), "inference.sahi.merge_iou_threshold"
    )
    if not 0 <= merge_iou_threshold <= 1:
        raise ValueError("inference.sahi.merge_iou_threshold must be between 0 and 1")

    tiling = _mapping(config.get("tiling"), "tiling")
    _boolean(tiling.get("enabled"), "tiling.enabled")
    tile_size_px = _positive_integer(tiling.get("tile_size_px"), "tiling.tile_size_px")
    if tiling.get("edge_mode") != "shift_last_tile_inside_image":
        raise ValueError("tiling.edge_mode must be 'shift_last_tile_inside_image'")
    overlap_px = _non_negative_integer(tiling.get("overlap_px"), "tiling.overlap_px")
    if overlap_px >= tile_size_px:
        raise ValueError("tiling.overlap_px must be smaller than tiling.tile_size_px")
    deduplication_tolerance_px = _finite_float(
        tiling.get("deduplication_tolerance_px"), "tiling.deduplication_tolerance_px"
    )
    if deduplication_tolerance_px < 0:
        raise ValueError("tiling.deduplication_tolerance_px must be non-negative")

    coordinates = _mapping(config.get("coordinate_conversion"), "coordinate_conversion")
    if coordinates.get("mode") not in {"stage", "pixel"}:
        raise ValueError("coordinate_conversion.mode must be 'stage' or 'pixel'; callable mode is API-only")
    _finite_float(coordinates.get("default_z_offset_um"), "coordinate_conversion.default_z_offset_um")
    merge_tolerance_um = _finite_float(
        coordinates.get("merge_tolerance_um"), "coordinate_conversion.merge_tolerance_um"
    )
    if merge_tolerance_um < 0:
        raise ValueError("coordinate_conversion.merge_tolerance_um must be non-negative")
    stage_direction = _mapping(coordinates.get("stage_direction"), "coordinate_conversion.stage_direction")
    if set(stage_direction) != {"x", "y", "z"}:
        raise ValueError("coordinate_conversion.stage_direction must contain x, y, and z values of -1 or 1")
    if any(not _is_stage_direction(value) for value in stage_direction.values()):
        raise ValueError("coordinate_conversion.stage_direction must contain x, y, and z values of -1 or 1")
    llsm = _mapping(coordinates.get("llsm"), "coordinate_conversion.llsm")
    _boolean(llsm.get("invert_y_stage_direction"), "coordinate_conversion.llsm.invert_y_stage_direction")

    slidebook = _mapping(config.get("slidebook"), "slidebook")
    _slidebook_string(slidebook.get("python_environment"), "slidebook.python_environment")
    _optional_slidebook_string(
        slidebook.get("objective_before_target_search"), "slidebook.objective_before_target_search"
    )
    _slidebook_string(slidebook.get("highres_objective"), "slidebook.highres_objective")
    objective_offset = _mapping(slidebook.get("objective_offset_um"), "slidebook.objective_offset_um")
    if set(objective_offset) != {"x", "y", "z"}:
        raise ValueError("slidebook.objective_offset_um must contain x, y, and z values")
    for axis, value in objective_offset.items():
        _finite_float(value, f"slidebook.objective_offset_um.{axis}")

    no_detection = _mapping(config.get("no_detection"), "no_detection")
    if no_detection.get("mode") not in {"end_workflow", "empty_3i_capture_script"}:
        raise ValueError("no_detection.mode must be 'end_workflow' or 'empty_3i_capture_script'")
    if no_detection["mode"] == "empty_3i_capture_script":
        _non_empty_string(no_detection.get("empty_3i_capture_script"), "no_detection.empty_3i_capture_script")

    plotting = _mapping(config.get("plotting"), "plotting")
    _boolean(plotting.get("enabled"), "plotting.enabled")
    _non_empty_string(plotting.get("cmap"), "plotting.cmap")
    bbox = _mapping(plotting.get("bbox"), "plotting.bbox")
    _non_empty_string(bbox.get("edge_color"), "plotting.bbox.edge_color")
    if _finite_float(bbox.get("line_width"), "plotting.bbox.line_width") <= 0:
        raise ValueError("plotting.bbox.line_width must be positive")
    label = _mapping(plotting.get("label"), "plotting.label")
    _positive_integer(label.get("font_size"), "plotting.label.font_size")
    _non_empty_string(label.get("text_color"), "plotting.label.text_color")
    _non_empty_string(label.get("background_color"), "plotting.label.background_color")
    background_alpha = _finite_float(label.get("background_alpha"), "plotting.label.background_alpha")
    if not 0 <= background_alpha <= 1:
        raise ValueError("plotting.label.background_alpha must be between 0 and 1")

    profile = _mapping(config.get("profile"), "profile")
    if not isinstance(profile.get("description"), str):
        raise ValueError("profile.description must be a string")
    if not isinstance(profile.get("highres_comment"), str):
        raise ValueError("profile.highres_comment must be a string")
    highres_script = profile.get("highres_script", "")
    if not isinstance(highres_script, str) or not highres_script.strip():
        raise ValueError("profile.highres_script must name the SlideBook postscan script")

    classes = profile.get("classes", {})
    if not isinstance(classes, dict) or not classes:
        raise ValueError("At least one detection class is required")

    names = set()
    for class_id, class_config in classes.items():
        class_config = _mapping(class_config, f"Class {class_id}")
        try:
            int(class_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Class ID {class_id!r} must be an integer") from error

        name = class_config.get("name", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Class {class_id} must have a name")
        name_key = name.casefold()
        if name_key in names:
            raise ValueError(f"Class name {name!r} is duplicated")
        names.add(name_key)

        try:
            threshold = float(class_config["confidence_threshold"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Class {class_id} confidence threshold must be a number") from error
        if not 0 <= threshold <= 1:
            raise ValueError(f"Class {class_id} confidence threshold must be between 0 and 1")

        try:
            priority = _integer(class_config["priority_rank"], f"Class {class_id} capture priority")
        except (KeyError, ValueError) as error:
            raise ValueError(f"Class {class_id} capture priority must be an integer") from error
        if priority < -1:
            raise ValueError(f"Class {class_id} capture priority must be -1 or greater")

    name_template = profile.get("name_template")
    _non_empty_string(name_template, "profile.name_template")
    try:
        name_template.format(class_name="class", x=0, y=0, z=0)
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError("profile.name_template may only use {class_name}, {x}, {y}, and {z}") from error


def _non_empty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _is_stage_direction(value):
    return not isinstance(value, bool) and isinstance(value, Real) and value in {-1, 1}


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


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
        """Choose the default editable config path.

        Args:
            None

        Returns:
            Path: ``Configs/default.yaml`` when available, otherwise the first
                alphabetically sorted YAML file in ``Configs``.

        Raises:
            FileNotFoundError: If no editable configs are available.
        """
        config_files = self.get_config_files()
        if not config_files:
            raise FileNotFoundError(f"No YAML configs found in {CONFIG_DIR}")
        default_config = CONFIG_DIR / "default.yaml"
        if default_config in config_files:
            return default_config
        return config_files[0]

    def load_config(self):
        """Read the YAML config from disk.

        Args:
            None

        Returns:
            dict: Parsed YAML config.
        """
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.migrate_config(config)
        return config

    def migrate_config(self, config):
        """Update older config shapes to the current prediction schema.

        Args:
            config (dict): Parsed YAML config to migrate in place.

        Returns:
            None
        """
        migrate_predict_config(config)
        slidebook = config.setdefault("slidebook", {})
        slidebook.setdefault("python_environment", "celfdrive-windows")
        if "objective_before_target_search" not in slidebook:
            slidebook["objective_before_target_search"] = slidebook.pop("pre_callback_objective", "")
        slidebook.setdefault("highres_objective", "20x Air")
        slidebook.setdefault("objective_offset_um", {"x": 0.0, "y": 0.0, "z": 0.0})
        inference = config.setdefault("inference", {})
        inference.setdefault("mode", "standard")
        sahi = inference.setdefault("sahi", {})
        sahi.setdefault("confidence_threshold", 0.5)
        sahi.setdefault("slice_size_px", 640)
        sahi.setdefault("overlap_ratio", 0.25)
        sahi.setdefault("tile_batch_size", 6)
        sahi.setdefault("merge_iou_threshold", 0.1)

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
        high_resolution = self.add_tab(self.main_notebook, "High Resolution Imaging")
        advanced = self.add_tab(self.main_notebook, "Advanced")

        self.build_general_tab(general)
        self.build_image_tab(preprocessing)
        self.build_coordinates_tab(coordinates)
        self.build_high_resolution_tab(high_resolution)
        self.build_advanced_tab(advanced)

        action_bar = ttk.Frame(self.root)
        action_bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        action_bar.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(
            value=f"Editing {self.config_path.name}. CelFDrive uses {DEFAULT_CONFIG_PATH.name} by default."
        )
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
        frame.content.columnconfigure(0, weight=1)
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

    def add_hint(self, parent, row, text, columnspan=2):
        """Add a wrapped explanatory label below a form section heading."""
        ttk.Label(parent, text=text, justify="left", wraplength=820).grid(
            row=row,
            column=0,
            columnspan=columnspan,
            sticky="w",
            pady=(0, 8),
        )

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

    def add_dropdown(self, parent, row, label, path, options):
        """Add a labelled dropdown bound to a config path.

        Args:
            parent (tk.Widget): Parent container.
            row (int): Grid row in the parent.
            label (str): Human-readable field label.
            path (list[str | int]): Config path for the value.
            options (list[str]): Allowed string values.

        Returns:
            ttk.Combobox: Dropdown widget bound to the config value.
        """
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        var = self.make_var(path, str)
        widget = ttk.Combobox(parent, textvariable=var, values=options, state="readonly")
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        return widget

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

    def write_slidebook_script(self):
        """Write the direct-Python SlideBook macro from the current config."""
        validate_prediction_config(self.config)
        SLIDEBOOK_SCRIPT_PATH.write_text(render_slidebook_script(self.config), encoding="utf-8")

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
        self.write_slidebook_script()
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
            self.write_slidebook_script()
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
        self.add_hint(
            project,
            0,
            "Select the CelFDrive project root: the folder cloned from GitHub that contains "
            "celfdrive_predict.yaml. Logs are saved in its Logging folder by default.",
            columnspan=3,
        )
        self.add_field(project, 1, "CelFDrive project root", ["project", "repo_path"], browse="dir")
        self.add_field(project, 2, "Model weights", ["model", "weights_path"], browse="file")
        self.add_dropdown(project, 3, "Backend", ["model", "backend"], ["ultralytics_yolo"])
        self.add_field(project, 4, "Suppress model stdout", ["model", "suppress_stdout"], bool)

        logging = self.add_section(parent, "Logging", 1)
        self.add_field(logging, 0, "Enabled", ["logging", "enabled"], bool)

        plotting = self.add_section(parent, "Plotting", 2)
        self.add_field(plotting, 0, "Enabled", ["plotting", "enabled"], bool)

    def build_image_tab(self, parent):
        """Build controls for preprocessing, tiling, and plotting config.

        Args:
            parent (tk.Widget): Tab content frame.

        Returns:
            None
        """
        preprocessing = self.add_section(parent, "Preprocessing", 0)
        self.add_dropdown(
            preprocessing,
            0,
            "Input channel mode",
            ["preprocessing", "input_channel", "mode"],
            ["first_channel_if_rgb"],
        )
        self.add_hint(
            preprocessing,
            1,
            "For an RGB input, CelFDrive uses the first channel. Two-dimensional images are used directly.",
            columnspan=3,
        )
        self.add_field(preprocessing, 2, "Top clip percentile", ["preprocessing", "top_clip_percentile"], float)
        self.add_field(preprocessing, 3, "Normalize min/max", ["preprocessing", "normalize_min_max"], bool)

        inference = self.add_section(parent, "Inference", 1)
        self.add_dropdown(
            inference,
            0,
            "Mode",
            ["inference", "mode"],
            ["standard", "sahi"],
        )
        self.add_field(
            inference,
            1,
            "SAHI confidence threshold",
            ["inference", "sahi", "confidence_threshold"],
            float,
        )
        self.add_field(
            inference,
            2,
            "SAHI slice size px",
            ["inference", "sahi", "slice_size_px"],
            int,
        )
        self.add_field(
            inference,
            3,
            "SAHI overlap ratio",
            ["inference", "sahi", "overlap_ratio"],
            float,
        )
        self.add_field(
            inference,
            4,
            "SAHI tile batch size",
            ["inference", "sahi", "tile_batch_size"],
            int,
        )
        self.add_field(
            inference,
            5,
            "SAHI merge IOU threshold",
            ["inference", "sahi", "merge_iou_threshold"],
            float,
        )
        self.add_hint(
            inference,
            6,
            "SAHI settings apply only when inference mode is sahi. Merging is class-aware and uses IOU.",
            columnspan=3,
        )

        tiling = self.add_section(parent, "Standard Tiling", 2)
        self.add_field(tiling, 0, "Enabled", ["tiling", "enabled"], bool)
        self.add_field(tiling, 1, "Tile size px", ["tiling", "tile_size_px"], int)
        self.add_dropdown(
            tiling,
            2,
            "Edge mode",
            ["tiling", "edge_mode"],
            ["shift_last_tile_inside_image"],
        )
        self.add_field(tiling, 3, "Overlap px", ["tiling", "overlap_px"], int)
        self.add_field(tiling, 4, "De-duplication tolerance px", ["tiling", "deduplication_tolerance_px"], float)
        self.add_hint(
            tiling,
            5,
            "De-duplication compares detection centres in the overview image, before conversion to stage coordinates.",
            columnspan=3,
        )

    def build_advanced_tab(self, parent):
        """Build advanced logging and plotting detail controls.

        Args:
            parent (tk.Widget): Tab content frame.

        Returns:
            None
        """
        logging = self.add_section(parent, "Logging Details", 0)
        self.add_hint(
            logging,
            0,
            "Log paths are relative to the CelFDrive project root. The default Logging folder is suitable for most workflows.",
            columnspan=3,
        )
        self.add_field(logging, 1, "Log directory", ["logging", "root_dir"], browse="dir")
        self.add_field(logging, 2, "Use date subfolder", ["logging", "use_date_subfolder"], bool)
        self.add_field(logging, 3, "Date format", ["logging", "date_format"])
        self.add_field(logging, 4, "Experiment prefix", ["logging", "experiment_folder", "prefix"])
        self.add_field(logging, 5, "Experiment digits", ["logging", "experiment_folder", "digits"], int)
        self.add_field(logging, 6, "Output image prefix", ["logging", "output_image", "prefix"])
        self.add_field(logging, 7, "Output image digits", ["logging", "output_image", "digits"], int)
        self.add_field(logging, 8, "Output image extension", ["logging", "output_image", "extension"])

        plotting = self.add_section(parent, "Plotting Details", 1)
        self.add_field(plotting, 0, "Color map", ["plotting", "cmap"])
        self.add_field(plotting, 1, "Box edge color", ["plotting", "bbox", "edge_color"])
        self.add_field(plotting, 2, "Box line width", ["plotting", "bbox", "line_width"], float)
        self.add_field(plotting, 3, "Label font size", ["plotting", "label", "font_size"], int)
        self.add_field(plotting, 4, "Label text color", ["plotting", "label", "text_color"])
        self.add_field(plotting, 5, "Label background color", ["plotting", "label", "background_color"])
        self.add_field(plotting, 6, "Label background alpha", ["plotting", "label", "background_alpha"], float)

    def build_coordinates_tab(self, parent):
        """Build controls for coordinate conversion and no-detection config.

        Args:
            parent (tk.Widget): Tab content frame.

        Returns:
            None
        """
        conversion = self.add_section(parent, "Coordinate Conversion", 0)
        self.add_dropdown(conversion, 0, "Coordinate mode", ["coordinate_conversion", "mode"], ["stage", "pixel"])
        self.add_field(conversion, 1, "Default z offset um", ["coordinate_conversion", "default_z_offset_um"], float)
        self.add_field(conversion, 2, "Merge tolerance um", ["coordinate_conversion", "merge_tolerance_um"], float)
        self.add_field(conversion, 3, "Stage direction x", ["coordinate_conversion", "stage_direction", "x"], int)
        self.add_field(conversion, 4, "Stage direction y", ["coordinate_conversion", "stage_direction", "y"], int)
        self.add_field(conversion, 5, "Stage direction z", ["coordinate_conversion", "stage_direction", "z"], int)
        self.add_field(conversion, 6, "LLSM mode: invert Y stage direction", ["coordinate_conversion", "llsm", "invert_y_stage_direction"], bool)
        self.add_field(conversion, 7, "SlideBook objective X offset um", ["slidebook", "objective_offset_um", "x"], float)
        self.add_field(conversion, 8, "SlideBook objective Y offset um", ["slidebook", "objective_offset_um", "y"], float)
        self.add_field(conversion, 9, "SlideBook objective Z offset um", ["slidebook", "objective_offset_um", "z"], float)

        slidebook = self.add_section(parent, "SlideBook Python Macro", 2)
        self.add_hint(
            slidebook,
            0,
            "Saving writes SlideBook/CelFDrive.sbs from these settings. Copy that generated file and "
            "find_locations_of_interest_montage.py to SlideBook's scripts folder.",
            columnspan=3,
        )
        self.add_field(slidebook, 1, "Registered Python environment", ["slidebook", "python_environment"])
        self.add_field(
            slidebook,
            2,
            "Objective before target search (optional)",
            ["slidebook", "objective_before_target_search"],
        )
        self.add_field(slidebook, 3, "High-resolution objective", ["slidebook", "highres_objective"])

        no_detection = self.add_section(parent, "No Detection", 1)
        mode_selector = self.add_dropdown(
            no_detection,
            0,
            "Mode",
            ["no_detection", "mode"],
            ["end_workflow", "empty_3i_capture_script"],
        )
        self.no_detection_script_frame = ttk.Frame(no_detection)
        self.no_detection_script_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.no_detection_script_frame.columnconfigure(1, weight=1)
        self.add_field(
            self.no_detection_script_frame,
            0,
            "Empty 3i capture script",
            ["no_detection", "empty_3i_capture_script"],
        )
        mode_selector.bind("<<ComboboxSelected>>", self.update_no_detection_fields)
        self.update_no_detection_fields()

    def update_no_detection_fields(self, _event=None):
        """Show the fallback script only when the selected mode uses it."""
        mode_var, _ = self.vars[("no_detection", "mode")]
        if mode_var.get() == "empty_3i_capture_script":
            self.no_detection_script_frame.grid()
        else:
            self.no_detection_script_frame.grid_remove()

    def build_high_resolution_tab(self, parent):
        """Build controls for the high-resolution capture workflow.

        Args:
            parent (tk.Widget): Tab content frame.

        Returns:
            None
        """
        parent.columnconfigure(0, weight=1)
        self.build_high_resolution_capture(parent)

    def build_high_resolution_capture(self, parent):
        """Build controls for the postscan script and detection-class table.

        Args:
            parent (tk.Widget): Profile tab frame.

        Returns:
            None
        """
        profile = self.add_section(parent, "SlideBook high-resolution capture", 0)
        base = ["profile"]
        self.add_hint(
            profile,
            0,
            "CelFDrive returns each selected target to this named SlideBook postscan script. "
            "The script name must match the postscan script created in SlideBook.",
        )
        self.add_field(profile, 1, "Imaging description", base + ["description"])
        self.add_field(profile, 2, "SlideBook postscan script", base + ["highres_script"])
        self.add_field(profile, 3, "Capture comment", base + ["highres_comment"])
        self.add_field(profile, 4, "Result name format", base + ["name_template"])
        self.add_hint(
            profile,
            5,
            "Use {class_name}, {x}, {y}, and {z} in the result name format to include the detected class and target coordinates.",
        )

        classes = self.add_section(parent, "Detection classes and capture order", 1)
        self.add_hint(
            classes,
            0,
            "A detection must meet its minimum confidence to be used. Capture priority 0 runs first; "
            "higher values run later. Set a priority of -1 to disable a class without deleting it.",
            columnspan=5,
        )
        headings = ["ID", "Class name", "Minimum confidence", "Capture priority", ""]
        for col, heading in enumerate(headings):
            ttk.Label(classes, text=heading).grid(row=1, column=col, sticky="w", padx=4, pady=(0, 6))
        classes.columnconfigure(1, weight=1)

        class_configs = self.config["profile"]["classes"]
        for row, class_id in enumerate(sorted(class_configs.keys(), key=int), start=2):
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
                command=lambda class_id=class_id: self.remove_class(class_id),
            ).grid(row=row, column=4, sticky="ew", padx=4, pady=3)

        add_row = len(class_configs) + 2
        ttk.Button(
            classes,
            text="Add detection class",
            command=self.add_class,
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
            self.set_value(list(path), self.parse_var(var, value_type, path))
        validate_prediction_config(self.config)

    def rebuild_ui(self, selected_main_tab=None, selected_profile=None):
        """Recreate the full form from the current in-memory config.

        Args:
            selected_main_tab (str | None): Main tab title to select after rebuilding.
            selected_profile (str | None): Deprecated. Ignored because there is one profile.

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
            selected_profile (str | None): Deprecated. Ignored because there is one profile.

        Returns:
            None
        """
        if selected_main_tab is not None:
            for tab_id in self.main_notebook.tabs():
                if self.main_notebook.tab(tab_id, "text") == selected_main_tab:
                    self.main_notebook.select(tab_id)
                    break

    def add_class(self):
        """Add a new class to a profile using the next available numeric class ID.

        Args:
            None

        Returns:
            None
        """
        try:
            self.apply_vars_to_config()
        except Exception as exc:
            messagebox.showerror("Cannot add class", str(exc))
            return

        classes = self.config["profile"]["classes"]
        numeric_ids = [int(class_id) for class_id in classes.keys()]
        next_class_id = max(numeric_ids, default=-1) + 1
        classes[next_class_id] = {
            "name": f"class_{next_class_id}",
            "confidence_threshold": 0.01,
            "priority_rank": next_class_id,
        }
        self.rebuild_ui(selected_main_tab="High Resolution Imaging")

    def remove_class(self, class_id):
        """Remove one class from a profile after confirmation.

        Args:
            class_id (int | str): Class ID key to remove.

        Returns:
            None
        """
        try:
            self.apply_vars_to_config()
        except Exception as exc:
            messagebox.showerror("Cannot remove class", str(exc))
            return

        classes = self.config["profile"]["classes"]
        if len(classes) <= 1:
            messagebox.showwarning("Cannot remove class", "Each profile must keep at least one class.")
            return

        class_name = classes[class_id].get("name", class_id)
        should_remove = messagebox.askyesno(
            "Remove class",
            f"Remove class {class_id}: {class_name}?",
        )
        if not should_remove:
            return

        del classes[class_id]
        self.rebuild_ui(selected_main_tab="High Resolution Imaging")

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
            self.write_slidebook_script()
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
