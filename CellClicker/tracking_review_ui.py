"""Review, edit, and select tracking box variants in a Tk canvas interface."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .tracking_sam2 import DEFAULT_SAM2_DEVICE, DEFAULT_SAM2_MODEL, predict_sam2_merged_box_from_points
from .tracking_xml import read_tracking_xml, write_tracking_data


BOX_COLORS = {
    "original": "#33aa33",
    "otsu": "#ff9933",
    "sam2": "#3399ff",
    "yolo11_tightened": "#8844dd",
    "tightened": "#cc33cc",
}

DEFAULT_CLASS_NAMES = [
    "prophase",
    "earlyprometaphase",
    "prometaphase",
    "metaphase",
    "anaphase",
    "telophase",
]

PREFERRED_BORDER_COLOR = "#22cc22"
UNSELECTED_BORDER_COLOR = "#dd4444"
FOCUSED_BORDER_COLOR = "#3399ff"

TILE_WIDTH = 180
TILE_HEIGHT = 180
COLUMN_WIDTH = 210
HEADER_HEIGHT = 84
ROW_GAP = 24
COLUMN_GAP = 16
SIDE_PAD = 12
TOP_PAD = 12
CROP_MARGIN_FACTOR = 1.8
MIN_CROP_SIZE = 96
HANDLE_SIZE = 8
SAM2_POINT_MARKER_RADIUS = 5


class TrackingReviewUI:
    """Tk editor for inspecting and selecting normalized tracking box variants.

    It reads/writes tracking XML in place. Display geometry is pixel-space, but
    stored boxes are normalized YOLO centre boxes.
    """
    def __init__(self, root, tracking_xml_path=None):
        self.root = root
        self.root.title("Tracking Review")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{int(screen_width * 0.9)}x{int(screen_height * 0.85)}+40+40")

        self.tracking_xml_path = tracking_xml_path
        self.tracking_data = None
        self.tracks = []
        self.track_index = 0

        self.track_canvas = None
        self.track_scroll_x = None
        self.track_scroll_y = None
        self.track_label = None
        self.status_label = None
        self.bulk_button_frame = None
        self.filter_button_frame = None
        self.series_var = None
        self.series_entry = None
        self.show_box_type_vars = {}
        self.class_options = []

        self.tile_refs = []
        self.tile_regions = []
        self.header_widgets = []
        self.selected_tile = None
        self.focused_tile = None
        self.edit_drag = None
        self.live_edit_rect_id = None
        self.sam2_points_mode = False
        self.sam2_point_tile = None
        self.sam2_points = []
        self.zoom_scale = 1.0

        self._build_layout()
        self._bind_keys()

        if self.tracking_xml_path:
            self.load_tracking_xml(self.tracking_xml_path)

    def _build_layout(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Button(top_frame, text="Open Tracking XML", command=self.open_tracking_xml).pack(side=tk.LEFT)
        tk.Button(top_frame, text="Save", command=self.save_tracking_xml).pack(side=tk.LEFT, padx=4)
        self.sam2_points_btn = tk.Button(top_frame, text="SAM2 Points To Box", command=self.start_sam2_points_mode)
        self.sam2_points_btn.pack(side=tk.LEFT, padx=(16, 4))
        self.sam2_done_btn = tk.Button(top_frame, text="Done", command=self.finish_sam2_points_mode, state=tk.DISABLED)
        self.sam2_done_btn.pack(side=tk.LEFT, padx=4)
        self.sam2_cancel_btn = tk.Button(
            top_frame,
            text="Cancel",
            command=self.cancel_sam2_points_mode,
            state=tk.DISABLED,
        )
        self.sam2_cancel_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(top_frame, text="Prev Track", command=self.prev_track).pack(side=tk.LEFT, padx=(16, 4))
        tk.Button(top_frame, text="Next Track", command=self.next_track).pack(side=tk.LEFT)
        tk.Label(top_frame, text="Series:").pack(side=tk.LEFT, padx=(16, 4))
        self.series_var = tk.StringVar()
        self.series_entry = tk.Entry(top_frame, textvariable=self.series_var, width=8)
        self.series_entry.pack(side=tk.LEFT)
        self.series_entry.bind("<Return>", self._on_series_entry_return)
        self.series_entry.bind("<Escape>", self._focus_review_canvas)
        tk.Button(top_frame, text="Go To Series", command=self.go_to_series).pack(side=tk.LEFT, padx=4)
        tk.Button(top_frame, text="Jump To Next TODO", command=self.jump_to_next_todo).pack(side=tk.LEFT, padx=4)

        self.track_label = tk.Label(top_frame, text="No tracking XML loaded", anchor=tk.W)
        self.track_label.pack(side=tk.LEFT, padx=16)

        self.filter_button_frame = tk.Frame(self.root)
        self.filter_button_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(self.filter_button_frame, text="Show:").pack(side=tk.LEFT)
        self.show_box_type_vars["original"] = tk.BooleanVar(value=False)
        self.show_box_type_vars["otsu"] = tk.BooleanVar(value=True)
        self.show_box_type_vars["sam2"] = tk.BooleanVar(value=True)
        self.show_box_type_vars["yolo11_tightened"] = tk.BooleanVar(value=True)
        self.show_box_type_vars["tightened"] = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.filter_button_frame,
            text="Original",
            variable=self.show_box_type_vars["original"],
            command=self.render_current_track,
        ).pack(side=tk.LEFT, padx=(8, 4))
        tk.Checkbutton(
            self.filter_button_frame,
            text="Otsu",
            variable=self.show_box_type_vars["otsu"],
            command=self.render_current_track,
        ).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(
            self.filter_button_frame,
            text="YOLO11",
            variable=self.show_box_type_vars["yolo11_tightened"],
            command=self.render_current_track,
        ).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(
            self.filter_button_frame,
            text="SAM2",
            variable=self.show_box_type_vars["sam2"],
            command=self.render_current_track,
        ).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(
            self.filter_button_frame,
            text="Tightened",
            variable=self.show_box_type_vars["tightened"],
            command=self.render_current_track,
        ).pack(side=tk.LEFT, padx=4)

        self.bulk_button_frame = tk.Frame(self.root)
        self.bulk_button_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(self.bulk_button_frame, text="Bulk select for current track:").pack(side=tk.LEFT)
        tk.Button(
            self.bulk_button_frame,
            text="All Original",
            command=lambda: self.set_all_preferred_box_type("original"),
        ).pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(
            self.bulk_button_frame,
            text="All Otsu",
            command=lambda: self.set_all_preferred_box_type("otsu"),
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.bulk_button_frame,
            text="All YOLO11",
            command=lambda: self.set_all_preferred_box_type("yolo11_tightened"),
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.bulk_button_frame,
            text="All SAM2",
            command=lambda: self.set_all_preferred_box_type("sam2"),
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.bulk_button_frame,
            text="All Tightened",
            command=lambda: self.set_all_preferred_box_type("tightened"),
        ).pack(side=tk.LEFT, padx=4)

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.track_canvas = tk.Canvas(canvas_frame, bg="#111111", highlightthickness=0)
        self.track_scroll_x = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.track_canvas.xview)
        self.track_scroll_y = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.track_canvas.yview)
        self.track_canvas.configure(xscrollcommand=self.track_scroll_x.set, yscrollcommand=self.track_scroll_y.set)

        self.track_canvas.grid(row=0, column=0, sticky="nsew")
        self.track_scroll_y.grid(row=0, column=1, sticky="ns")
        self.track_scroll_x.grid(row=1, column=0, sticky="ew")

        self.status_label = tk.Label(
            self.root,
            text=(
                "Click a tile to choose the preferred box for that frame. "
                "Drag a corner or edge handle to edit that box."
            ),
            anchor=tk.W,
            justify=tk.LEFT,
        )
        self.status_label.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        self.track_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.track_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.track_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.track_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.track_canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)

    def _bind_keys(self):
        self.root.bind("<Left>", lambda _: self.move_focus_horizontal(-1))
        self.root.bind("<Right>", lambda _: self.move_focus_horizontal(1))
        self.root.bind("<Up>", lambda _: self.move_focus_vertical(-1))
        self.root.bind("<Down>", lambda _: self.move_focus_vertical(1))
        self.root.bind("<Return>", self._on_return_key)
        self.root.bind("<n>", lambda _: None if self._entry_has_focus() else self.next_track())
        self.root.bind("<o>", lambda _: None if self._entry_has_focus() else self.set_all_preferred_box_type("otsu"))
        self.root.bind("<y>", lambda _: None if self._entry_has_focus() else self.set_all_preferred_box_type("yolo11_tightened"))
        self.root.bind("<p>", lambda _: None if self._entry_has_focus() else self.prev_track())
        self.root.bind("<f>", lambda _: None if self._entry_has_focus() or self.sam2_points_mode else self.start_sam2_points_mode())
        self.root.bind("<s>", lambda _: None if self._entry_has_focus() else self.save_tracking_xml())
        self.root.bind("<d>", lambda _: None if self._entry_has_focus() or not self.sam2_points_mode else self.finish_sam2_points_mode())
        self.root.bind("<c>", lambda _: None if self._entry_has_focus() or not self.sam2_points_mode else self.cancel_sam2_points_mode())
        self.root.bind("<Prior>", lambda _: self.prev_track())
        self.root.bind("<Next>", lambda _: self.next_track())
        self.root.bind("<Control-s>", lambda _: self.save_tracking_xml())
        for class_id in range(10):
            self.root.bind(str(class_id), lambda _event, cid=class_id: self._assign_class_from_key(cid))

    def open_tracking_xml(self):
        xml_path = filedialog.askopenfilename(
            title="Select Tracking XML",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
        )
        if xml_path:
            self.load_tracking_xml(xml_path)

    def load_tracking_xml(self, xml_path):
        self.tracking_xml_path = os.path.normpath(xml_path)
        self.tracking_data = read_tracking_xml(self.tracking_xml_path)
        self.class_options = [
            (int(class_id), str(class_name))
            for class_id, class_name in sorted(
                self.tracking_data.get("classes", {}).items(),
                key=lambda item: int(item[0]),
            )
        ]
        self.tracks = self.tracking_data.get("tracks", [])
        self.track_index = 0
        self.selected_tile = None
        self.focused_tile = None
        self.edit_drag = None
        self.cancel_sam2_points_mode(update_status=False)
        self.render_current_track(reset_view=True)
        self._focus_review_canvas()

    def save_tracking_xml(self):
        if not self.tracking_data or not self.tracking_xml_path:
            return
        write_tracking_data(self.tracking_xml_path, self.tracking_data)
        self.status_label.config(text=f"Saved tracking XML: {self.tracking_xml_path}")

    def get_current_track(self):
        if not self.tracks:
            return None
        self.track_index = max(0, min(self.track_index, len(self.tracks) - 1))
        return self.tracks[self.track_index]

    def prev_track(self):
        if not self.tracks:
            return
        self.track_index = max(0, self.track_index - 1)
        self.selected_tile = None
        self.focused_tile = None
        self.edit_drag = None
        self.cancel_sam2_points_mode(update_status=False)
        self.render_current_track(reset_view=True)
        self._focus_review_canvas()

    def next_track(self):
        if not self.tracks:
            return
        current = self.get_current_track()
        if current and current.get("review_state") == "pending":
            current["review_state"] = "reviewed"
        self.track_index = min(len(self.tracks) - 1, self.track_index + 1)
        self.selected_tile = None
        self.focused_tile = None
        self.edit_drag = None
        self.cancel_sam2_points_mode(update_status=False)
        self.render_current_track(reset_view=True)
        self._focus_review_canvas()

    def _entry_has_focus(self):
        return self.series_entry is not None and self.root.focus_get() == self.series_entry

    def _focus_review_canvas(self, event=None):
        if self.track_canvas is not None:
            self.track_canvas.focus_set()
        return "break" if event is not None else None

    def _on_series_entry_return(self, event):
        self.go_to_series()
        return "break"

    def _on_return_key(self, event):
        if self._entry_has_focus():
            return None
        self.activate_focused_tile()
        return "break"

    def go_to_series(self):
        if not self.tracks:
            return

        series_text = self.series_var.get().strip() if self.series_var is not None else ""
        if not series_text:
            messagebox.showerror("Go To Series", "Enter a series number.")
            return

        try:
            target_series = int(series_text)
        except ValueError:
            messagebox.showerror("Go To Series", "Series must be a whole number.")
            return

        target_index = None
        for index, track in enumerate(self.tracks):
            try:
                if int(track.get("series_id")) == target_series:
                    target_index = index
                    break
            except (TypeError, ValueError):
                continue

        if target_index is None and 1 <= target_series <= len(self.tracks):
            target_index = target_series - 1

        if target_index is None:
            messagebox.showerror("Go To Series", f"Could not find series `{target_series}`.")
            return

        self.track_index = target_index
        self.selected_tile = None
        self.focused_tile = None
        self.edit_drag = None
        self.cancel_sam2_points_mode(update_status=False)
        self.render_current_track(reset_view=True)
        self._focus_review_canvas()

    def _track_has_todo(self, track):
        if track.get("review_state") == "pending":
            return True
        for timepoint in track.get("timepoints", []):
            if timepoint.get("preferred_box_type") == "original":
                return True
        return False

    def jump_to_next_todo(self):
        if not self.tracks:
            return

        total_tracks = len(self.tracks)
        for offset in range(1, total_tracks + 1):
            candidate_index = (self.track_index + offset) % total_tracks
            if self._track_has_todo(self.tracks[candidate_index]):
                self.track_index = candidate_index
                self.selected_tile = None
                self.focused_tile = None
                self.edit_drag = None
                self.cancel_sam2_points_mode(update_status=False)
                self.render_current_track(reset_view=True)
                self._focus_review_canvas()
                self.status_label.config(
                    text=(
                        f"Jumped to next TODO track: series "
                        f"{self.tracks[candidate_index].get('series_id')}."
                    )
                )
                return

        messagebox.showinfo(
            "Jump To Next TODO",
            "No TODO tracks found. No remaining tracks have any frame still set to `original`.",
        )

    def render_current_track(self, reset_view=False):
        prior_xview = self.track_canvas.xview()
        prior_yview = self.track_canvas.yview()
        for widget in self.header_widgets:
            widget.destroy()
        self.header_widgets = []
        self.track_canvas.delete("all")
        self.live_edit_rect_id = None
        self.tile_refs = []
        self.tile_regions = []

        track = self.get_current_track()
        if not track:
            self.track_label.config(text="No tracking XML loaded")
            self.track_canvas.config(scrollregion=(0, 0, 1, 1))
            return

        timepoints = track.get("timepoints", [])
        filtered_timepoints = [self._get_visible_boxes(tp) for tp in timepoints]
        max_rows = max((len(boxes) for boxes in filtered_timepoints), default=1)
        tile_height = self.get_tile_height()
        row_gap = self.get_row_gap()
        column_width = self.get_column_width()
        column_gap = self.get_column_gap()
        content_height = TOP_PAD + HEADER_HEIGHT + max_rows * (tile_height + row_gap) + 40

        for timepoint_index, timepoint in enumerate(timepoints):
            column_x = SIDE_PAD + timepoint_index * (column_width + column_gap)
            self._draw_timepoint_column(column_x, timepoint_index, timepoint, filtered_timepoints[timepoint_index])

        content_width = SIDE_PAD + len(timepoints) * (column_width + column_gap) + 40
        self.track_canvas.config(scrollregion=(0, 0, content_width, content_height))
        if reset_view:
            self.track_canvas.xview_moveto(0)
            self.track_canvas.yview_moveto(0)
        else:
            if prior_xview:
                self.track_canvas.xview_moveto(prior_xview[0])
            if prior_yview:
                self.track_canvas.yview_moveto(prior_yview[0])
        self._ensure_focused_tile()
        self._draw_sam2_point_markers()
        self.track_label.config(
            text=(
                f"Track {self.track_index + 1}/{len(self.tracks)} "
                f"({track['track_id']})  Series {track['series_id']}  Frames {len(timepoints)}  "
                f"Review: {track.get('review_state', 'reviewed')}"
            )
        )

    def _get_visible_boxes(self, timepoint):
        boxes = timepoint.get("boxes", [])
        visible_boxes = []
        non_original_box_exists = any(box.get("box_type") != "original" for box in boxes)

        for box in boxes:
            box_type = box.get("box_type")
            if box_type == "original":
                if not self.show_box_type_vars["original"].get() and non_original_box_exists:
                    continue
            elif not self.show_box_type_vars.get(box_type, tk.BooleanVar(value=True)).get():
                continue
            visible_boxes.append(box)

        return visible_boxes or boxes[:1]

    def _draw_timepoint_column(self, column_x, timepoint_index, timepoint, visible_boxes):
        self.track_canvas.create_text(
            column_x,
            TOP_PAD,
            text=f"t={timepoint_index}",
            fill="white",
            anchor=tk.NW,
            font=("Arial", 10, "bold"),
        )
        self._draw_class_selector(column_x, timepoint_index, timepoint)

        preferred_box_type = timepoint.get("preferred_box_type")
        for box_row_index, box in enumerate(visible_boxes):
            tile_y = TOP_PAD + HEADER_HEIGHT + box_row_index * (self.get_tile_height() + self.get_row_gap())
            tile_info = self._draw_box_tile(
                column_x=column_x,
                tile_y=tile_y,
                timepoint_index=timepoint_index,
                box_row_index=box_row_index,
                timepoint=timepoint,
                box=box,
                is_preferred=box["box_type"] == preferred_box_type,
            )
            self.tile_regions.append(tile_info)

    def _draw_class_selector(self, column_x, timepoint_index, timepoint):
        class_values = self._get_class_dropdown_values()
        class_label = tk.Label(
            self.track_canvas,
            text="Class",
            bg="#111111",
            fg="white",
            anchor="w",
            font=("Arial", 9),
        )
        self.header_widgets.append(class_label)
        self.track_canvas.create_window(column_x, TOP_PAD + 20, window=class_label, anchor=tk.NW)

        class_combo = ttk.Combobox(
            self.track_canvas,
            values=class_values,
            state="readonly",
            width=max(18, int(self.get_tile_width() / 11)),
        )
        current_value = self._get_current_class_value(timepoint)
        try:
            class_combo.current(class_values.index(current_value))
        except ValueError:
            class_combo.set(current_value)
        class_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event, tp_index=timepoint_index, combo=class_combo: self._on_class_changed(tp_index, combo.get()),
        )
        self.header_widgets.append(class_combo)
        self.track_canvas.create_window(column_x, TOP_PAD + 38, window=class_combo, anchor=tk.NW)

    def _get_class_dropdown_values(self):
        if self.class_options:
            return [self._format_class_value(class_id, class_name) for class_id, class_name in self.class_options]
        return [self._format_class_value(index, class_name) for index, class_name in enumerate(DEFAULT_CLASS_NAMES)]

    def _get_current_class_value(self, timepoint):
        class_id = int(timepoint["class_id"])
        if self.class_options:
            for option_class_id, option_class_name in self.class_options:
                if int(option_class_id) == class_id:
                    return self._format_class_value(option_class_id, option_class_name)
        phase_name = timepoint.get("phase_name") or DEFAULT_CLASS_NAMES[class_id] if 0 <= class_id < len(DEFAULT_CLASS_NAMES) else "unknown"
        return self._format_class_value(class_id, phase_name)

    def _format_class_value(self, class_id, class_name):
        return f"{int(class_id)}: {class_name}"

    def _parse_class_value(self, value):
        class_id_text, phase_name = value.split(":", 1)
        return int(class_id_text.strip()), phase_name.strip()

    def _on_class_changed(self, timepoint_index, value):
        track = self.get_current_track()
        if track is None:
            return
        class_id, phase_name = self._parse_class_value(value)
        timepoint = track["timepoints"][timepoint_index]
        timepoint["class_id"] = class_id
        timepoint["phase_name"] = phase_name
        self.status_label.config(
            text=(
                f"Updated frame {timepoint_index} in track {track['track_id']} "
                f"to class `{phase_name}` ({class_id}). Save to persist changes."
            )
        )

    def _assign_class_from_key(self, class_id):
        if self._entry_has_focus() or self.sam2_points_mode:
            return None
        self._ensure_focused_tile()
        if not self.focused_tile:
            return "break"

        class_value = None
        for option_class_id, option_class_name in self.class_options:
            if int(option_class_id) == int(class_id):
                class_value = self._format_class_value(option_class_id, option_class_name)
                break

        if class_value is None:
            return "break"

        self._on_class_changed(self.focused_tile["timepoint_index"], class_value)
        self.render_current_track()
        self._scroll_tile_into_view()
        return "break"

    def _draw_box_tile(self, column_x, tile_y, timepoint_index, box_row_index, timepoint, box, is_preferred):
        image = Image.open(timepoint["image_path"])
        crop_bounds = self._compute_crop_bounds(image.size, box)
        tile_width = self.get_tile_width()
        tile_height = self.get_tile_height()
        crop_image = image.crop(crop_bounds).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(crop_image)
        self.tile_refs.append(photo)

        box_type = box["box_type"]
        is_focused = (
            self.focused_tile
            and self.focused_tile.get("timepoint_index") == timepoint_index
            and self.focused_tile.get("box_type") == box_type
        )
        border_color = FOCUSED_BORDER_COLOR if is_focused else (PREFERRED_BORDER_COLOR if is_preferred else UNSELECTED_BORDER_COLOR)

        x1 = column_x
        y1 = tile_y
        x2 = column_x + tile_width
        y2 = tile_y + tile_height

        image_id = self.track_canvas.create_image(x1, y1, image=photo, anchor=tk.NW)
        rect_id = self.track_canvas.create_rectangle(x1, y1, x2, y2, outline=border_color, width=3)
        if is_focused:
            self.track_canvas.create_rectangle(
                x1 - 4,
                y1 - 4,
                x2 + 4,
                y2 + 4,
                outline=FOCUSED_BORDER_COLOR,
                width=3,
                dash=(6, 3),
            )
        label_id = self.track_canvas.create_text(
            x1 + 4,
            y1 + 4,
            text=box_type,
            fill=border_color,
            anchor=tk.NW,
            font=("Arial", 10, "bold"),
        )

        box_rect = self._draw_box_within_tile(x1, y1, crop_bounds, image.size, box, border_color)
        self._draw_predicted_class_warning(box_rect, timepoint, box)
        handles = self._draw_edit_handles(box_rect) if self._is_selected_tile(timepoint_index, box_type) else {}

        return {
            "timepoint_index": timepoint_index,
            "box_row_index": box_row_index,
            "box_type": box_type,
            "tile_rect": (x1, y1, x2, y2),
            "box_rect": box_rect,
            "handles": handles,
            "crop_bounds": crop_bounds,
            "image_size": image.size,
            "image_id": image_id,
            "rect_id": rect_id,
            "label_id": label_id,
        }

    @staticmethod
    def _predicted_class_warning(timepoint, box):
        """Return a non-interactive warning only for a differing multiclass prediction."""
        predicted_class_id = box.get("predicted_class_id")
        if box.get("box_type") != "yolo11_tightened" or predicted_class_id is None:
            return None
        if int(predicted_class_id) == int(timepoint["class_id"]):
            return None
        class_name = box.get("predicted_class_name") or str(predicted_class_id)
        return f"pred: {class_name}"

    def _draw_predicted_class_warning(self, box_rect, timepoint, box):
        warning = self._predicted_class_warning(timepoint, box)
        if warning is None:
            return None
        x1, _, _, y2 = box_rect
        return self.track_canvas.create_text(
            x1 + 3,
            y2 - 3,
            text=warning,
            fill="#ff3333",
            anchor=tk.SW,
            font=("Arial", 9, "bold"),
        )

    def _get_box_rect_within_tile(self, tile_x, tile_y, crop_bounds, image_size, box):
        tile_width = self.get_tile_width()
        tile_height = self.get_tile_height()
        image_width, image_height = image_size
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
        crop_width = max(1, crop_x2 - crop_x1)
        crop_height = max(1, crop_y2 - crop_y1)

        center_x = box["x_center"] * image_width
        center_y = box["y_center"] * image_height
        width = box["width"] * image_width
        height = box["height"] * image_height

        box_x1 = (center_x - width / 2 - crop_x1) * tile_width / crop_width
        box_y1 = (center_y - height / 2 - crop_y1) * tile_height / crop_height
        box_x2 = (center_x + width / 2 - crop_x1) * tile_width / crop_width
        box_y2 = (center_y + height / 2 - crop_y1) * tile_height / crop_height

        return (
            tile_x + box_x1,
            tile_y + box_y1,
            tile_x + box_x2,
            tile_y + box_y2,
        )

    def _draw_box_within_tile(self, tile_x, tile_y, crop_bounds, image_size, box, color):
        abs_x1, abs_y1, abs_x2, abs_y2 = self._get_box_rect_within_tile(tile_x, tile_y, crop_bounds, image_size, box)

        self.track_canvas.create_rectangle(
            abs_x1,
            abs_y1,
            abs_x2,
            abs_y2,
            outline=color,
            width=2,
        )
        return abs_x1, abs_y1, abs_x2, abs_y2

    def _draw_edit_handles(self, box_rect):
        x1, y1, x2, y2 = box_rect
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        half = HANDLE_SIZE / 2.0
        handle_points = {
            "nw": (x1, y1),
            "n": (mid_x, y1),
            "ne": (x2, y1),
            "e": (x2, mid_y),
            "se": (x2, y2),
            "s": (mid_x, y2),
            "sw": (x1, y2),
            "w": (x1, mid_y),
        }
        handles = {}
        for handle_name, (cx, cy) in handle_points.items():
            rect = (cx - half, cy - half, cx + half, cy + half)
            self.track_canvas.create_rectangle(
                rect[0],
                rect[1],
                rect[2],
                rect[3],
                fill="#ffffff",
                outline=FOCUSED_BORDER_COLOR,
                width=2,
            )
            handles[handle_name] = rect
        return handles

    def _is_selected_tile(self, timepoint_index, box_type):
        return (
            self.selected_tile is not None
            and self.selected_tile.get("timepoint_index") == timepoint_index
            and self.selected_tile.get("box_type") == box_type
        )

    def _find_handle(self, tile_info, canvas_x, canvas_y):
        for handle_name, rect in tile_info.get("handles", {}).items():
            x1, y1, x2, y2 = rect
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                return handle_name
        return None

    def _compute_crop_bounds(self, image_size, box):
        image_width, image_height = image_size
        center_x = box["x_center"] * image_width
        center_y = box["y_center"] * image_height
        width = max(MIN_CROP_SIZE, box["width"] * image_width * CROP_MARGIN_FACTOR)
        height = max(MIN_CROP_SIZE, box["height"] * image_height * CROP_MARGIN_FACTOR)

        x1 = max(0, int(round(center_x - width / 2)))
        y1 = max(0, int(round(center_y - height / 2)))
        x2 = min(image_width, int(round(center_x + width / 2)))
        y2 = min(image_height, int(round(center_y + height / 2)))

        if x2 <= x1:
            x2 = min(image_width, x1 + MIN_CROP_SIZE)
        if y2 <= y1:
            y2 = min(image_height, y1 + MIN_CROP_SIZE)
        return x1, y1, x2, y2

    def on_canvas_press(self, event):
        canvas_x = self.track_canvas.canvasx(event.x)
        canvas_y = self.track_canvas.canvasy(event.y)
        tile_info = self._find_tile(canvas_x, canvas_y)
        if tile_info is None:
            return

        if self.sam2_points_mode:
            self._handle_sam2_point_click(tile_info, canvas_x, canvas_y)
            return

        self.selected_tile = tile_info
        self.focused_tile = {
            "timepoint_index": tile_info["timepoint_index"],
            "box_row_index": tile_info["box_row_index"],
            "box_type": tile_info["box_type"],
        }
        self._set_preferred_box_type(tile_info["timepoint_index"], tile_info["box_type"])
        self.render_current_track()
        tile_info = self._find_tile(canvas_x, canvas_y)
        if tile_info is None:
            return
        self.selected_tile = tile_info

        handle_name = self._find_handle(tile_info, canvas_x, canvas_y)
        if handle_name:
            self.edit_drag = {
                "timepoint_index": tile_info["timepoint_index"],
                "box_type": tile_info["box_type"],
                "handle": handle_name,
                "start_x": canvas_x,
                "start_y": canvas_y,
                "current_x": canvas_x,
                "current_y": canvas_y,
                "initial_box_rect": tile_info["box_rect"],
            }
            self.live_edit_rect_id = self._draw_live_edit_rect()

    def on_canvas_drag(self, event):
        if not self.edit_drag:
            return
        self.edit_drag["current_x"] = self.track_canvas.canvasx(event.x)
        self.edit_drag["current_y"] = self.track_canvas.canvasy(event.y)
        self._update_live_edit_rect()

    def on_canvas_release(self, event):
        if not self.edit_drag or not self.selected_tile:
            return

        self.edit_drag["current_x"] = self.track_canvas.canvasx(event.x)
        self.edit_drag["current_y"] = self.track_canvas.canvasy(event.y)
        self._apply_edit_drag()
        self.edit_drag = None
        self.live_edit_rect_id = None
        self.render_current_track()

    def _draw_live_edit_rect(self):
        if not self.edit_drag:
            return None
        x1, y1, x2, y2 = self._get_dragged_box_rect()
        return self.track_canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline="#ff4444",
            width=2,
            dash=(4, 2),
        )

    def _update_live_edit_rect(self):
        if not self.edit_drag:
            return
        if self.live_edit_rect_id is None:
            self.live_edit_rect_id = self._draw_live_edit_rect()
            return
        self.track_canvas.coords(self.live_edit_rect_id, *self._get_dragged_box_rect())
        self.track_canvas.tag_raise(self.live_edit_rect_id)

    def _get_dragged_box_rect(self):
        x1, y1, x2, y2 = self.edit_drag["initial_box_rect"]
        current_x = self.edit_drag["current_x"]
        current_y = self.edit_drag["current_y"]
        handle = self.edit_drag["handle"]

        if "n" in handle:
            y1 = current_y
        if "s" in handle:
            y2 = current_y
        if "w" in handle:
            x1 = current_x
        if "e" in handle:
            x2 = current_x

        if handle == "n":
            x1, x2 = self.edit_drag["initial_box_rect"][0], self.edit_drag["initial_box_rect"][2]
        elif handle == "s":
            x1, x2 = self.edit_drag["initial_box_rect"][0], self.edit_drag["initial_box_rect"][2]
        elif handle == "w":
            y1, y2 = self.edit_drag["initial_box_rect"][1], self.edit_drag["initial_box_rect"][3]
        elif handle == "e":
            y1, y2 = self.edit_drag["initial_box_rect"][1], self.edit_drag["initial_box_rect"][3]

        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def _apply_edit_drag(self):
        tile_info = self.selected_tile
        drag = self.edit_drag
        x1, y1, x2, y2 = tile_info["tile_rect"]
        crop_x1, crop_y1, crop_x2, crop_y2 = tile_info["crop_bounds"]
        image_width, image_height = tile_info["image_size"]

        raw_x1, raw_y1, raw_x2, raw_y2 = self._get_dragged_box_rect()
        start_x = min(max(raw_x1, x1), x2)
        start_y = min(max(raw_y1, y1), y2)
        end_x = min(max(raw_x2, x1), x2)
        end_y = min(max(raw_y2, y1), y2)

        if abs(end_x - start_x) < 4 or abs(end_y - start_y) < 4:
            return

        rel_x1 = min(start_x, end_x) - x1
        rel_y1 = min(start_y, end_y) - y1
        rel_x2 = max(start_x, end_x) - x1
        rel_y2 = max(start_y, end_y) - y1

        crop_width = max(1, crop_x2 - crop_x1)
        crop_height = max(1, crop_y2 - crop_y1)

        tile_width = self.get_tile_width()
        tile_height = self.get_tile_height()
        abs_x1 = crop_x1 + rel_x1 * crop_width / tile_width
        abs_y1 = crop_y1 + rel_y1 * crop_height / tile_height
        abs_x2 = crop_x1 + rel_x2 * crop_width / tile_width
        abs_y2 = crop_y1 + rel_y2 * crop_height / tile_height

        new_center_x = ((abs_x1 + abs_x2) / 2.0) / image_width
        new_center_y = ((abs_y1 + abs_y2) / 2.0) / image_height
        new_width = abs(abs_x2 - abs_x1) / image_width
        new_height = abs(abs_y2 - abs_y1) / image_height

        track = self.get_current_track()
        timepoint = track["timepoints"][tile_info["timepoint_index"]]
        target_box_type = tile_info["box_type"] if tile_info["box_type"] == "tightened" else "tightened"
        edited_box = None
        for box in timepoint["boxes"]:
            if box["box_type"] == target_box_type:
                edited_box = box
                break

        if edited_box is None:
            edited_box = {
                "box_type": "tightened",
                "format": "yolo_xywh_norm",
                "x_center": 0.0,
                "y_center": 0.0,
                "width": 0.0,
                "height": 0.0,
                "source": f"tracking_review_ui:derived_from={tile_info['box_type']}",
            }
            timepoint["boxes"].append(edited_box)

        edited_box["x_center"] = max(0.0, min(1.0, new_center_x))
        edited_box["y_center"] = max(0.0, min(1.0, new_center_y))
        edited_box["width"] = max(1e-6, min(1.0, new_width))
        edited_box["height"] = max(1e-6, min(1.0, new_height))
        edited_box["source"] = (
            "tracking_review_ui"
            if tile_info["box_type"] == "tightened"
            else f"tracking_review_ui:derived_from={tile_info['box_type']}"
        )
        timepoint["preferred_box_type"] = "tightened"
        self.selected_tile = {**tile_info, "box_type": "tightened"}
        self.focused_tile = {
            "timepoint_index": tile_info["timepoint_index"],
            "box_row_index": tile_info["box_row_index"],
            "box_type": "tightened",
        }

        self.status_label.config(
            text=(
                f"Edited frame {tile_info['timepoint_index']} in track {track['track_id']}. "
                f"Saved the edit into `tightened` and set it as preferred. Save to persist changes."
            )
        )

    def _set_preferred_box_type(self, timepoint_index, box_type):
        track = self.get_current_track()
        if track is None:
            return
        timepoint = track["timepoints"][timepoint_index]
        timepoint["preferred_box_type"] = box_type
        self.status_label.config(
            text=f"Selected `{box_type}` as the preferred box for frame {timepoint_index}. Drag to edit it."
        )

    def _ensure_focused_tile(self):
        if not self.tile_regions:
            self.focused_tile = None
            return
        if self.focused_tile:
            for tile in self.tile_regions:
                if (
                    tile["timepoint_index"] == self.focused_tile.get("timepoint_index")
                    and tile["box_type"] == self.focused_tile.get("box_type")
                ):
                    self.focused_tile = {
                        "timepoint_index": tile["timepoint_index"],
                        "box_row_index": tile["box_row_index"],
                        "box_type": tile["box_type"],
                    }
                    return
        first_tile = self.tile_regions[0]
        self.focused_tile = {
            "timepoint_index": first_tile["timepoint_index"],
            "box_row_index": first_tile["box_row_index"],
            "box_type": first_tile["box_type"],
        }

    def _find_focus_match(self, timepoint_index, preferred_row_index):
        candidates = [tile for tile in self.tile_regions if tile["timepoint_index"] == timepoint_index]
        if not candidates:
            return None
        candidates.sort(key=lambda tile: abs(tile["box_row_index"] - preferred_row_index))
        return candidates[0]

    def _scroll_tile_into_view(self):
        self._ensure_focused_tile()
        if not self.focused_tile:
            return

        target_tile = None
        for tile in self.tile_regions:
            if (
                tile["timepoint_index"] == self.focused_tile["timepoint_index"]
                and tile["box_type"] == self.focused_tile["box_type"]
            ):
                target_tile = tile
                break

        if target_tile is None:
            return

        x1, y1, x2, y2 = target_tile["tile_rect"]
        canvas_width = max(1, self.track_canvas.winfo_width())
        canvas_height = max(1, self.track_canvas.winfo_height())
        view_left = self.track_canvas.canvasx(0)
        view_top = self.track_canvas.canvasy(0)
        view_right = view_left + canvas_width
        view_bottom = view_top + canvas_height
        margin = 24

        scrollregion = self.track_canvas.cget("scrollregion")
        if not scrollregion:
            return
        _, _, total_width, total_height = [float(value) for value in scrollregion.split()]
        max_left = max(0.0, total_width - canvas_width)
        max_top = max(0.0, total_height - canvas_height)

        target_left = view_left
        target_top = view_top

        if x1 < view_left:
            target_left = max(0.0, x1 - margin)
        elif x2 > view_right:
            target_left = min(max_left, x2 - canvas_width + margin)

        if y1 < view_top:
            target_top = max(0.0, y1 - margin)
        elif y2 > view_bottom:
            target_top = min(max_top, y2 - canvas_height + margin)

        if total_width > 0:
            self.track_canvas.xview_moveto(target_left / total_width)
        if total_height > 0:
            self.track_canvas.yview_moveto(target_top / total_height)

    def move_focus_horizontal(self, delta):
        self._ensure_focused_tile()
        if not self.focused_tile:
            return
        target = self._find_focus_match(
            self.focused_tile["timepoint_index"] + delta,
            self.focused_tile["box_row_index"],
        )
        if target is None:
            return
        self.focused_tile = {
            "timepoint_index": target["timepoint_index"],
            "box_row_index": target["box_row_index"],
            "box_type": target["box_type"],
        }
        self.render_current_track()
        self._scroll_tile_into_view()

    def move_focus_vertical(self, delta):
        self._ensure_focused_tile()
        if not self.focused_tile:
            return
        target = self._find_focus_match(
            self.focused_tile["timepoint_index"],
            self.focused_tile["box_row_index"] + delta,
        )
        if target is None:
            return
        self.focused_tile = {
            "timepoint_index": target["timepoint_index"],
            "box_row_index": target["box_row_index"],
            "box_type": target["box_type"],
        }
        self.render_current_track()
        self._scroll_tile_into_view()

    def activate_focused_tile(self):
        self._ensure_focused_tile()
        if not self.focused_tile:
            return
        self._set_preferred_box_type(
            self.focused_tile["timepoint_index"],
            self.focused_tile["box_type"],
        )
        self.selected_tile = dict(self.focused_tile)
        self.render_current_track()

    def set_all_preferred_box_type(self, box_type):
        track = self.get_current_track()
        if track is None:
            return

        updated = 0
        missing = 0
        for timepoint in track.get("timepoints", []):
            matching_box = None
            for box in timepoint.get("boxes", []):
                if box.get("box_type") == box_type:
                    matching_box = box
                    break

            if matching_box is None:
                missing += 1
                continue

            timepoint["preferred_box_type"] = box_type
            updated += 1

        self.selected_tile = None
        self.edit_drag = None
        self.render_current_track()
        self.status_label.config(
            text=(
                f"Set preferred box type to `{box_type}` for {updated} frames "
                f"in the current track. {missing} frames did not have that box type."
            )
        )

    def _find_tile(self, canvas_x, canvas_y):
        for tile_info in self.tile_regions:
            x1, y1, x2, y2 = tile_info["tile_rect"]
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                return tile_info
        return None

    def get_tile_width(self):
        return max(80, int(TILE_WIDTH * self.zoom_scale))

    def get_tile_height(self):
        return max(80, int(TILE_HEIGHT * self.zoom_scale))

    def get_column_width(self):
        return max(self.get_tile_width() + 30, int(COLUMN_WIDTH * self.zoom_scale))

    def get_row_gap(self):
        return max(8, int(ROW_GAP * self.zoom_scale))

    def get_column_gap(self):
        return max(8, int(COLUMN_GAP * self.zoom_scale))

    def adjust_zoom(self, delta):
        step = 0.1 if delta > 0 else -0.1
        new_zoom = min(2.5, max(0.5, round(self.zoom_scale + step, 2)))
        if new_zoom == self.zoom_scale:
            return
        self.zoom_scale = new_zoom
        self.render_current_track()
        self.status_label.config(text=f"Zoom: {int(self.zoom_scale * 100)}%")

    def on_mousewheel(self, event):
        delta = -1 * int(event.delta / 120) if event.delta else 0
        if delta:
            self.track_canvas.yview_scroll(delta, "units")

    def on_shift_mousewheel(self, event):
        delta = int(event.delta / 120) if event.delta else 0
        if delta:
            self.adjust_zoom(delta)

    def _set_sam2_points_controls(self):
        self.sam2_points_btn.config(state=tk.DISABLED if self.sam2_points_mode else tk.NORMAL)
        active_state = tk.NORMAL if self.sam2_points_mode else tk.DISABLED
        self.sam2_done_btn.config(state=active_state)
        self.sam2_cancel_btn.config(state=active_state)

    def start_sam2_points_mode(self):
        if not self.tracks:
            return
        self.sam2_points_mode = True
        self.sam2_point_tile = None
        self.sam2_points = []
        self.edit_drag = None
        self._set_sam2_points_controls()
        self.render_current_track()
        self.status_label.config(
            text="SAM2 point mode: click one tile, add any number of positive points, then click Done."
        )

    def cancel_sam2_points_mode(self, update_status=True):
        self.sam2_points_mode = False
        self.sam2_point_tile = None
        self.sam2_points = []
        self._set_sam2_points_controls()
        if self.track_canvas is not None:
            self.render_current_track()
        if update_status and self.status_label is not None:
            self.status_label.config(text="Cancelled SAM2 point mode.")

    def _handle_sam2_point_click(self, tile_info, canvas_x, canvas_y):
        if self.sam2_point_tile is None:
            self.sam2_point_tile = {
                "timepoint_index": tile_info["timepoint_index"],
                "box_type": tile_info["box_type"],
            }
        elif (
            tile_info["timepoint_index"] != self.sam2_point_tile["timepoint_index"]
            or tile_info["box_type"] != self.sam2_point_tile["box_type"]
        ):
            messagebox.showerror(
                "SAM2 Points To Box",
                "All SAM2 points for one action must be placed on the same sub image.",
            )
            return

        image_point = self._canvas_point_to_image_point(tile_info, canvas_x, canvas_y)
        self.sam2_points.append(image_point)
        self.selected_tile = tile_info
        self.focused_tile = {
            "timepoint_index": tile_info["timepoint_index"],
            "box_row_index": tile_info["box_row_index"],
            "box_type": tile_info["box_type"],
        }
        self.render_current_track()
        self.status_label.config(
            text=(
                f"SAM2 point mode: added {len(self.sam2_points)} point(s) on frame "
                f"{tile_info['timepoint_index']}. Click Done to create/update `tightened`."
            )
        )

    def _canvas_point_to_image_point(self, tile_info, canvas_x, canvas_y):
        tile_x1, tile_y1, tile_x2, tile_y2 = tile_info["tile_rect"]
        crop_x1, crop_y1, crop_x2, crop_y2 = tile_info["crop_bounds"]
        crop_width = max(1, crop_x2 - crop_x1)
        crop_height = max(1, crop_y2 - crop_y1)
        tile_width = self.get_tile_width()
        tile_height = self.get_tile_height()

        rel_x = min(max(canvas_x - tile_x1, 0.0), tile_x2 - tile_x1)
        rel_y = min(max(canvas_y - tile_y1, 0.0), tile_y2 - tile_y1)
        image_x = crop_x1 + rel_x * crop_width / tile_width
        image_y = crop_y1 + rel_y * crop_height / tile_height
        return float(image_x), float(image_y)

    def _image_point_to_canvas_point(self, tile_info, image_point):
        tile_x1, tile_y1, _, _ = tile_info["tile_rect"]
        crop_x1, crop_y1, crop_x2, crop_y2 = tile_info["crop_bounds"]
        crop_width = max(1, crop_x2 - crop_x1)
        crop_height = max(1, crop_y2 - crop_y1)
        tile_width = self.get_tile_width()
        tile_height = self.get_tile_height()

        image_x, image_y = image_point
        canvas_x = tile_x1 + (image_x - crop_x1) * tile_width / crop_width
        canvas_y = tile_y1 + (image_y - crop_y1) * tile_height / crop_height
        return canvas_x, canvas_y

    def _draw_sam2_point_markers(self):
        if not self.sam2_points_mode or not self.sam2_point_tile or not self.sam2_points:
            return
        target_tile = None
        for tile in self.tile_regions:
            if (
                tile["timepoint_index"] == self.sam2_point_tile["timepoint_index"]
                and tile["box_type"] == self.sam2_point_tile["box_type"]
            ):
                target_tile = tile
                break
        if target_tile is None:
            return

        for image_point in self.sam2_points:
            canvas_x, canvas_y = self._image_point_to_canvas_point(target_tile, image_point)
            self.track_canvas.create_oval(
                canvas_x - SAM2_POINT_MARKER_RADIUS,
                canvas_y - SAM2_POINT_MARKER_RADIUS,
                canvas_x + SAM2_POINT_MARKER_RADIUS,
                canvas_y + SAM2_POINT_MARKER_RADIUS,
                fill="#00ccff",
                outline="#ffffff",
                width=2,
            )

    def finish_sam2_points_mode(self):
        if not self.sam2_points_mode:
            return
        if self.sam2_point_tile is None or not self.sam2_points:
            messagebox.showerror("SAM2 Points To Box", "Add at least one point before clicking Done.")
            return

        target_tile = None
        for tile in self.tile_regions:
            if (
                tile["timepoint_index"] == self.sam2_point_tile["timepoint_index"]
                and tile["box_type"] == self.sam2_point_tile["box_type"]
            ):
                target_tile = tile
                break
        if target_tile is None:
            raise RuntimeError("SAM2 point mode failed: target tile is no longer visible.")

        track = self.get_current_track()
        timepoint = track["timepoints"][target_tile["timepoint_index"]]
        try:
            tightened_box = predict_sam2_merged_box_from_points(
                image_path=timepoint["image_path"],
                point_pixels=self.sam2_points,
                crop_xyxy=target_tile["crop_bounds"],
                image_size=target_tile["image_size"],
                model_name=DEFAULT_SAM2_MODEL,
                device=DEFAULT_SAM2_DEVICE,
            )
        except Exception as exc:
            messagebox.showerror("SAM2 Points To Box", str(exc))
            return

        existing_tightened = None
        for box in timepoint["boxes"]:
            if box.get("box_type") == "tightened":
                existing_tightened = box
                break
        if existing_tightened is None:
            timepoint["boxes"].append(tightened_box)
        else:
            existing_tightened.update(tightened_box)

        timepoint["preferred_box_type"] = "tightened"
        self.selected_tile = {
            "timepoint_index": target_tile["timepoint_index"],
            "box_row_index": target_tile["box_row_index"],
            "box_type": "tightened",
        }
        self.focused_tile = dict(self.selected_tile)
        point_count = len(self.sam2_points)
        self.cancel_sam2_points_mode(update_status=False)
        self.render_current_track()
        self.status_label.config(
            text=(
                f"SAM2 created/updated `tightened` from {point_count} point(s) on frame "
                f"{target_tile['timepoint_index']}. Save to persist changes."
            )
        )


def launch_tracking_review_ui(tracking_xml_path=None):
    """Launch the review window for a tracking XML path, optionally prompting."""
    root = tk.Tk()
    app = TrackingReviewUI(root, tracking_xml_path=tracking_xml_path)
    root.mainloop()
    return app
