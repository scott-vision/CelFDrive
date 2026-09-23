"""Interactive CellClicker GUI for drawing boxes across image series."""

import os
import re
import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Toplevel, Label, filedialog, messagebox
from PIL import Image, ImageTk
from CellClicker.manageXML import (
    append_cell_regions_xml, check_xml, find_series_anchor_for_image,
    get_next_series_id, get_series_extension_start, prepare_series_extension,
    remove_entry_from_xml,
)
from CellClicker.clicker_utils import clear_series_width_cache, get_previous_image_name, get_relative_image_name, yolov5_to_xywh
from CellClicker.tooltips import add_tooltip
from CellClicker.project_paths import resolve_cell_regions_xml
from CellClicker.image_series import UnsupportedFrameNamingError, discover_image_series, series_menu_labels


MINI_CLICKER_DISPLAY_SCALE = 3
# Pillow added ``Image.Resampling`` in 9.1.  The legacy constants remain
# available in older supported Conda environments, including the environment
# used to run CellClicker directly on this workstation.
PIL_RESAMPLING = getattr(Image, "Resampling", Image)

# Guidance for the two image canvases. The pointer rests on a canvas constantly,
# so these are shown as a hover hint only once per run and otherwise live in a
# status line that is always visible and never covers the image.
CLICKER_CANVAS_HINT = "Click the centre of the same cell. Each click records a box and moves to the preceding frame."
VIEWER_CANVAS_HINT = (
    "Drag around a cell to create a red box. Green boxes are existing tracks; "
    "right-click one to extend it earlier, or use its red X to delete the track."
)
IMAGE_VIEWER_HELP_TEXT = (
    "Keyboard shortcuts: Left Arrow = previous image; Right Arrow = next image; I = Inspect; "
    "U = Update Progress; F = Finished (while the mini-clicker is open).\n\n"
    "Navigate images with << and >> or by entering a frame number.\n\n"
    "To start a track, drag a red box around a cell, then select Inspect. In the mini-clicker, "
    "click the cell's position in each preceding image. Select Finished to stop early.\n\n"
    "Existing tracks are green. Right-click a green box to extend that track earlier. Select the red X "
    "on its final box to delete the entire track.\n\n"
    "Use Update Progress after changing annotations outside this window."
)


def display_coordinate_to_roi_coordinate(coordinate, display_scale):
    """Convert a mini-clicker display coordinate to an integer source ROI pixel."""
    if display_scale <= 0:
        raise ValueError("Mini-clicker display scale must be greater than zero.")
    return coordinate // display_scale


def centered_window_position(parent_x, parent_y, parent_width, parent_height, window_width, window_height):
    """Return the top-left position that centers a child window over its parent."""
    return (
        parent_x + (parent_width - window_width) // 2,
        parent_y + (parent_height - window_height) // 2,
    )


class ImageProcessor:
    """Load and normalize microscope image files for annotation display."""
    def __init__(
        self, master, image_path, bbox, xml_path, series_id, next_class_id=0,
        anchor_path=None, on_finished=None,
    ):
        self.master = master
        self.image_path = image_path
        self.first_label = anchor_path or image_path
        self.bbox = bbox
        self.xml_path = xml_path
        self.class_id = next_class_id
        self.series_id = series_id
        self.on_finished = on_finished
        self._ended = False
        
        # Create a new window for image processing
        self.image_window = Toplevel(self.master)
        self.image_window.title("Cell Clicker")
        self.image_window.transient(self.master)
        self.image_window.protocol("WM_DELETE_WINDOW", self.end_session)
        self.image_window.bind("<f>", self.finish_hotkey)
        self.image_window.bind("<F>", self.finish_hotkey)

        # Display area for images
        self.canvas = tk.Canvas(self.image_window, width=600, height=600)
        self.canvas.pack()
        add_tooltip(self.canvas, CLICKER_CANVAS_HINT, once=True)

        # Status label
        self.status_label = Label(self.image_window, text=CLICKER_CANVAS_HINT, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Button to manually end the session
        self.stop_button = Button(self.image_window, text="Finished", command=self.end_session)
        add_tooltip(
            self.stop_button,
            "Stop tracing when the cell is absent or the series is complete. Shortcut: F.",
        )
        self.stop_button.pack(side=tk.BOTTOM)

        # print("current series")
        # print(self.series_count)
        
        # Load the initial image and display it
        self.display_roi()
        self.center_over_master()
        self.image_window.after_idle(self.focus_clicker)

    def center_over_master(self):
        """Place the mini-clicker at the center of the main Cell Clicker window."""
        self.image_window.update_idletasks()
        x, y = centered_window_position(
            self.master.winfo_rootx(),
            self.master.winfo_rooty(),
            self.master.winfo_width(),
            self.master.winfo_height(),
            self.image_window.winfo_width(),
            self.image_window.winfo_height(),
        )
        self.image_window.geometry(f"+{x}+{y}")

    def focus_clicker(self):
        """Ensure the new mini-clicker receives keyboard focus immediately."""
        self.image_window.lift()
        self.image_window.focus_force()
        self.canvas.focus_set()

    def finish_hotkey(self, event):
        """End the current mini-clicker session when F is pressed."""
        self.end_session()
        return "break"

    def normalize_image(self, image):
        """ Normalizes an image to a range of [0, 255] and converts it to uint8 data type. """
        image = (image - image.min()) / (image.max() - image.min()) * 255
        return image.astype(np.uint8)
    

    # def normalize_image(self, image):
    #     """Applies CLAHE to an image to enhance contrast locally."""
    #     # Convert image to grayscale if it is in color
    #     if len(image.shape) == 3:
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    #     # Create a CLAHE object
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #     cl1 = clahe.apply(image)
    
    #     return cl1

    def display_roi(self):
        """Displays an ROI centered at a specified location from an image."""
        img = cv2.imread(self.image_path)
        if img is None:
            self.status_label.config(text="Failed to load image")
            return
        
        self.y_shape, self.x_shape = img.shape[:2]
        expand = 10  # Example expansion parameter
        # print(self.bbox)
        
        # current_x and current_y are the top left coords, expand them and recalculate expanded w and h, limit to image dims
        self.current_x, self.current_y = max(0, self.bbox['x'] - expand), max(0, self.bbox['y'] - expand)
        self.current_w = min(self.x_shape, self.current_x + self.bbox['width'] + 2*expand) - self.current_x
        self.current_h = min(self.y_shape, self.current_y + self.bbox['height'] + 2*expand) - self.current_y
        display_width = self.current_w * MINI_CLICKER_DISPLAY_SCALE
        display_height = self.current_h * MINI_CLICKER_DISPLAY_SCALE
        self.canvas.config(width=display_width, height=display_height)
        # print(self.current_x, self.current_y, self.current_w, self.current_h)
        roi = self.normalize_image(img[self.current_y:self.current_y+self.current_h, self.current_x:self.current_x+self.current_w])
        display_roi = Image.fromarray(roi).resize(
            (display_width, display_height), PIL_RESAMPLING.NEAREST
        )
        self.image = ImageTk.PhotoImage(image=display_roi)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)


        
        # param = {'x': x, 'y': y, 'width': w, 'height': h, 'img': img, 'image_name': self.image_path,
        #          'class_id': 0, 'series_count': series_count + 1, 'first_label': self.image_path, 'xml_path': self.xml_path}
        # self.canvas.bind("<Button-1>", lambda event, arg=param: self.click_event(event, param))
        self.canvas.bind("<Button-1>", lambda event : self.click_event(event))

    def click_event(self, event):
        """Handles mouse click events to calculate a new ROI centered on the clicked position."""
        x = display_coordinate_to_roi_coordinate(event.x, MINI_CLICKER_DISPLAY_SCALE)
        y = display_coordinate_to_roi_coordinate(event.y, MINI_CLICKER_DISPLAY_SCALE)
        # print(f"Clicked at: x={x}, y={y}")  # Placeholder for actual functionality

        # add the clicked x and y to the top left to get the global clicked val
        x_global = x + self.current_x
        y_global = y + self.current_y
        
        # check that this wont exceed the top left of the image if the width and height is applied
        x_start = max(0, x_global - self.current_w // 2)
        y_start = max(0, y_global - self.current_h // 2)
        x_end = x_start + self.current_w
        y_end = y_start + self.current_h

        # print(x_start, x_end, y_start, y_end)
        # print("saving series")
        # print(self.series_count+1)
        
        append_cell_regions_xml(self.xml_path, self.first_label, self.class_id,
                                (x_start + x_end) / 2, (y_start + y_end) / 2,
                                self.current_w, self.current_h, self.x_shape, self.y_shape, self.series_id)

        self.class_id += 1

        self.bbox['x'] = x_start
        self.bbox['y'] = y_start
        
        self.image_path = get_previous_image_name(self.image_path)
        if self.image_path:
            self.display_roi()
        else:
            self.end_session()



    def end_session(self):
        """Close the mini-clicker and notify the Image Viewer that annotation ended."""
        if self._ended:
            return
        self._ended = True
        self.image_window.destroy()
        if self.on_finished is not None:
            self.on_finished()
        self.master.after_idle(self.restore_master_focus)

    def restore_master_focus(self):
        """Return keyboard focus to the Image Viewer after closing the mini-clicker."""
        self.master.lift()
        self.master.focus_force()



class ImageViewer:
    """Tk annotation window for creating bounding boxes across image series."""
    def __init__(self, root, project_dir=None):
        self.root = root
        self.root.title("Image Viewer")
        self.project_dir = os.path.normpath(project_dir) if project_dir else None

        help_frame = tk.Frame(self.root)
        help_frame.pack(side=tk.TOP, fill=tk.X)
        self.help_button = tk.Button(help_frame, text="?", width=2, command=self.show_help)
        self.help_button.pack(side=tk.RIGHT, padx=8, pady=6)

        # Set up the frame for navigation buttons
        frame = tk.Frame(self.root)
        frame.pack(side=tk.BOTTOM, pady=20)

        # Numeric input for frame number
        self.series_var = tk.StringVar(value="")
        self.series_menu = tk.OptionMenu(frame, self.series_var, "")
        tk.Label(frame, text="Series:").pack(side=tk.LEFT)
        self.series_menu.pack(side=tk.LEFT)
        self.frame_number = tk.StringVar()
        self.frame_entry = tk.Entry(frame, textvariable=self.frame_number)
        self.frame_entry.pack(side=tk.LEFT)


        # Go to frame button
        self.btn_go_to_frame = tk.Button(frame, text="Go to Frame", command=self.go_to_frame)
        self.btn_go_to_frame.pack(side=tk.LEFT)

        # Buttons
        self.btn_update_progress = tk.Button(frame, text="Update Progress", command=self.update_progress)
        add_tooltip(
            self.btn_update_progress,
            "Reload existing track overlays after annotations change. Shortcut: U.",
        )
        self.btn_update_progress.pack(side=tk.LEFT)
        self.btn_back = tk.Button(frame, text="<<", command=self.prev_image, state=tk.DISABLED)
        self.btn_back.pack(side=tk.LEFT)
        self.btn_forward = tk.Button(frame, text=">>", command=self.next_image)
        self.btn_forward.pack(side=tk.LEFT)
        self.btn_inspect = tk.Button(frame, text="Inspect", command=self.inspect_bbox)
        add_tooltip(
            self.btn_inspect,
            "Trace the red-boxed cell through preceding frames. Shortcut: I.",
        )
        self.btn_inspect.pack(side=tk.RIGHT)

        # Label for image name
        self.label = tk.Label(self.root, text='', pady=10)
        self.label.pack(side=tk.BOTTOM)

        # Canvas guidance, kept visible rather than hovering over the image
        self.hint_label = tk.Label(
            self.root, text=VIEWER_CANVAS_HINT, fg="#555555",
            wraplength=900, justify=tk.LEFT, pady=2,
        )
        self.hint_label.pack(side=tk.BOTTOM)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root, cursor="cross")
        add_tooltip(self.canvas, VIEWER_CANVAS_HINT, once=True)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        

        # Binding mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_bbox)
        self.canvas.bind("<B1-Motion>", self.expand_bbox)
        self.canvas.bind("<ButtonRelease-1>", self.finish_bbox)
        self.canvas.bind("<Left>", self.left_arrow)
        self.canvas.bind("<Right>", self.right_arrow)

        # Bind the configure event for resizing images
        self.original_image = None
        self.canvas.bind("<Configure>", self.handle_resize)
        # self.canvas.focus_set()  # Set focus to the canvas

        self.start_x = None
        self.start_y = None
        self.rect = None
        self.bbox_details = None

        # Size the image is actually drawn at. The canvas is packed to fill the
        # window, so once the window stops matching the image's aspect ratio the
        # two differ and only this one may be used to convert coordinates.
        self.displayed_size = None

        # Load images
        self.images = []
        self.series_images = {}
        self.series_labels = {}
        self.current_image = 0
        self.load_images()

        # Initial image setup
        self.update_image()

        # Bind left and right arrow keys to root window
        self.root.bind("<Left>", self.left_arrow)
        self.root.bind("<Right>", self.right_arrow)
        self.root.bind("<i>", self.inspect_hotkey)
        self.root.bind("<I>", self.inspect_hotkey)
        self.root.bind("<u>", self.update_progress_hotkey)
        self.root.bind("<U>", self.update_progress_hotkey)
        self.root.after_idle(self.focus_viewer)

        # Keep the image canvas focused so navigation and annotation hotkeys work immediately.
        self.root.after(100, self.focus_canvas)


    def focus_canvas(self):
        """Focus the image canvas so keyboard shortcuts do not edit the frame field."""
        self.canvas.focus_set()

    def focus_viewer(self):
        """Bring a newly opened Image Viewer forward for immediate hotkey use."""
        self.root.lift()
        self.root.focus_force()

    def inspect_hotkey(self, event):
        """Open the mini-clicker for the currently drawn box when I is pressed."""
        if self._typing_a_frame_number():
            return None
        self.inspect_bbox()
        return "break"

    def update_progress_hotkey(self, event):
        """Refresh track annotations when U is pressed."""
        if self._typing_a_frame_number():
            return None
        self.update_progress()
        return "break"

    def show_help(self):
        """Display concise instructions for annotating and managing tracks."""
        messagebox.showinfo("Cell Clicker help", IMAGE_VIEWER_HELP_TEXT, parent=self.root)


    def focus_in_event(self, event):
        self.frame_entry.focus_set()

    def _typing_a_frame_number(self):
        """True when the frame-number box has focus and keys should edit text.

        Navigation and action hotkeys are bound on the whole window so they work
        wherever focus sits, which would otherwise make them fire while the user
        is typing in that box.
        """
        try:
            return self.root.focus_get() is self.frame_entry
        except (KeyError, tk.TclError):
            return False

    def left_arrow(self, event):
        """Step one frame back, once per keypress."""
        if self._typing_a_frame_number():
            return None
        self.prev_image()
        # Arrow keys are bound on both the canvas and the window so they work
        # whichever has focus. Stop the event there, or the focused canvas would
        # hand it on to the window binding and move two frames per keypress.
        return "break"

    def right_arrow(self, event):
        """Step one frame forward, once per keypress."""
        if self._typing_a_frame_number():
            return None
        self.next_image()
        return "break"

    def load_images(self):
        # Ask the user for the directory
        if self.project_dir:
            directory = self.project_dir
        else:
            directory = filedialog.askdirectory(title="Select Directory with Images")
        if not directory:
            return
        
        directory = os.path.join(directory, "images")
        directory = os.path.normpath(directory)
        # The project's images may have been renamed since it was last loaded.
        clear_series_width_cache()
        print('current image folder')
        print(directory)
        self.xml_path = str(resolve_cell_regions_xml(os.path.dirname(directory)).path)
        try:
            self.xml_df = check_xml(self.xml_path)
        except UnsupportedFrameNamingError as exc:
            self._report_unsupported_frame_naming(exc)
            return
        print(self.xml_df)
        if not self.xml_df.empty:
            self.original_image_folder = os.path.normpath(self.xml_df['PathName'][0].split("images")[0])

        else:
            self.original_image_folder = directory.split("images")[0]

        self.original_image_folder = os.path.join(self.original_image_folder, "images")
        self.original_image_folder = os.path.normpath(self.original_image_folder)

        print(f'original image folder: {self.original_image_folder}')




        try:
            discovered = discover_image_series(directory)
        except UnsupportedFrameNamingError as exc:
            self._report_unsupported_frame_naming(exc)
            return
        self.series_images = {
            name: [self.normalize_path(path) for path in paths]
            for name, paths in discovered.items()
        }
        if not self.series_images:
            self.label.config(text="No images found!")
            return
        self._configure_series_menu()

    def _report_unsupported_frame_naming(self, error):
        """Refuse a project whose timepoints cannot be tracked, and say why."""
        self.series_images, self.images = {}, []
        self.label.config(text="Project not loaded: a series writes its timepoint inconsistently.")
        messagebox.showerror("Unsupported Image Names", str(error), parent=self.root)

    def _configure_series_menu(self):
        """Populate the series selector and activate the first ordered series."""
        # Every series filename repeats the experiment name, so the selector
        # shows only the part that distinguishes them, usually the position.
        self.series_labels = series_menu_labels(self.series_images)
        menu = self.series_menu["menu"]
        menu.delete(0, "end")
        for name in self.series_images:
            menu.add_command(label=self.series_labels[name], command=lambda selected=name: self.select_series(selected))
        self.select_series(next(iter(self.series_images)))

    def select_series(self, series_name):
        """Switch navigation and annotation to one independent image series."""
        self.series_var.set(self.series_labels.get(series_name, series_name))
        self.images = self.series_images[series_name]
        self.current_image = 0
        self.frame_number.set("0")
        self.update_image()

    def norm_esc_str(self, path):
        path_normalized = os.path.normpath(path)
        path_escaped = re.escape(path_normalized)
        return path_escaped
    
    def normalize_path(self, path):
        """Normalize and convert all path separators to forward slashes for uniformity."""
        return os.path.normpath(path).replace(os.sep, '/')


    def update_image(self):
        if not self.images:
            return

        img_path = self.images[self.current_image]
        # print(img_path)

        self.original_image = Image.open(img_path)

        # get the actual image data without the path to directory
        img_path = os.path.normpath(img_path.split('images')[1][1:])
        print(img_path)
        
#     filter to current image
        if not self.xml_df.empty:
            # Match the literal path fragment: image names contain regex
            # metacharacters such as `.`, and Windows separators would
            # otherwise be read as escape sequences.
            filtered_df = self.xml_df[self.xml_df['PathName'].apply(lambda path: os.path.normpath(path)).str.contains(self.norm_esc_str(img_path))]

            if filtered_df.empty:
                self.existing_bboxes = []  # No bounding boxes found
            else:
                # Sort by series and class order, then group by series
                grouped = filtered_df.sort_values(by=["SeriesID", "ClassID"]).groupby("SeriesID")

                # Extract bounding boxes correctly
                self.existing_bboxes = [
                    (
                        *yolov5_to_xywh(
                            float(row[3]),  # XCenter
                            float(row[4]),  # YCenter
                            float(row[5]),  # Width
                            float(row[6]),  # Height
                            self.original_image.width,
                            self.original_image.height
                        ),
                        series_id,
                        int(row[2]) == 0  # `is_last` is True if Class ID is 0
                    )
                    for series_id, group in grouped
                    for row in group.itertuples(index=False, name=None)  # Access row values as tuple
                ]
        else:
            self.existing_bboxes = []
        
        self.display_image()
        print(self.existing_bboxes)
        self.draw_existing_bboxes()

    def handle_resize(self, event):
        """Ensure the image fits within the canvas while maintaining aspect ratio."""
        if self.original_image:
            screen_height = self.root.winfo_screenheight()
            max_canvas_height = int(screen_height * 0.75)  # Max height is 3/4 of screen height

            img_width, img_height = self.original_image.size
            aspect_ratio = img_width / img_height

            # Determine the new size while keeping aspect ratio
            new_height = min(event.height, max_canvas_height)
            new_width = int(new_height * aspect_ratio)

            # Ensure it fits within the new window width
            if new_width > event.width:
                new_width = event.width
                new_height = int(new_width / aspect_ratio)

            # Resize the image
            resized_image = self.original_image.resize((new_width, new_height), PIL_RESAMPLING.LANCZOS)
            self.photo_img = ImageTk.PhotoImage(resized_image)

            # Update canvas dimensions
            self.canvas.config(width=new_width, height=new_height)
            self.displayed_size = (new_width, new_height)

            # Clear and redraw the image
            self.canvas.delete("all")  # Remove previous image to prevent layering
            self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)
            self.draw_existing_bboxes()



    def resize_image(self, width, height):
        # Avoid resizing to zero to prevent PIL errors
        if width > 1 and height > 1:
            resized_image = self.original_image.resize((width, height), PIL_RESAMPLING.LANCZOS)
            self.photo_img = ImageTk.PhotoImage(resized_image)
            self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)

    def displayed_image_size(self):
        """Return the on-screen size of the image, which is not the canvas size.

        ``pack(fill=BOTH, expand=True)`` overrides the width and height the
        canvas is configured with, so a maximised window leaves letterbox space
        beside or below the image that must not be counted when scaling boxes.
        """
        if self.displayed_size and all(self.displayed_size):
            return self.displayed_size
        return self.canvas.winfo_width(), self.canvas.winfo_height()

    def image_to_canvas_scale(self):
        """Return the factors taking stored image pixels to canvas pixels."""
        width, height = self.displayed_image_size()
        return width / self.original_image.width, height / self.original_image.height

    def canvas_to_image_scale(self):
        """Return the factors taking canvas pixels back to stored image pixels."""
        width, height = self.displayed_image_size()
        return self.original_image.width / width, self.original_image.height / height

    def display_image(self):
        if self.original_image:
            # Get screen height and calculate 3/4 of it
            screen_height = self.root.winfo_screenheight()
            max_canvas_height = int(screen_height * 0.75)

            # Get the original image dimensions
            img_width, img_height = self.original_image.size

            # Calculate new width while maintaining aspect ratio
            aspect_ratio = img_width / img_height
            new_width = int(max_canvas_height * aspect_ratio)
            new_height = max_canvas_height

            # Resize the image to fit within the calculated dimensions
            self.resized_image = self.original_image.resize((new_width, new_height), PIL_RESAMPLING.LANCZOS)

            # Update the canvas size dynamically
            self.canvas.config(width=new_width, height=new_height)
            self.displayed_size = (new_width, new_height)

            # Display the resized image
            self.photo_img = ImageTk.PhotoImage(self.resized_image)
            self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)

            # Update label with filename
            self.label.config(text=os.path.basename(self.images[self.current_image]))

            # Ensure existing bounding boxes are redrawn
            self.draw_existing_bboxes()



    def start_bbox(self, event):
        # Remove previous bounding box if any
        if self.rect:
            self.canvas.delete(self.rect)
        # Save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def expand_bbox(self, event):
        # Modify the current rectangle's corner to new mouse position
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def finish_bbox(self, event):
        # Finalize the rectangle
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        self.bbox_details = {'x': x0, 'y': y0, 'width': x1 - x0, 'height': y1 - y0}

    def inspect_bbox(self):
        if self.bbox_details and self.original_image:
            # Calculate scale factors
            scale_x, scale_y = self.canvas_to_image_scale()

            # Adjust coordinates
            original_x = int(self.bbox_details['x'] * scale_x)
            original_y = int(self.bbox_details['y'] * scale_y)
            original_width = int(self.bbox_details['width'] * scale_x)
            original_height = int(self.bbox_details['height'] * scale_y)

            self.start_clicker({
                'x': original_x,
                'y': original_y,
                'width': original_width,
                'height': original_height
            })

    def complete_clicker_session(self):
        """Clear the inspected box and refresh track overlays after mini-clicker completion."""
        if self.rect is not None:
            self.canvas.delete(self.rect)
        self.rect = None
        self.bbox_details = None
        self.update_progress()
        self.focus_canvas()

    def draw_existing_bboxes(self):
        """Draw bounding boxes and place 'X' buttons only on the last bounding box of each series."""

        self.canvas.delete("bbox")  # Clear previous bounding boxes
        self.canvas.delete("delete_x")  # Clear previous delete markers
        self.delete_buttons = {}  # Store delete button references

        scale_x, scale_y = self.image_to_canvas_scale()

        for bbox in self.existing_bboxes:
            x, y, w, h, series_id, is_last = bbox  # Extract data

            # Calculate scaled coordinates
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_w = int(w * scale_x)
            scaled_h = int(h * scale_y)

            # Draw bounding box
            self.canvas.create_rectangle(
                scaled_x, scaled_y, scaled_x + scaled_w, scaled_y + scaled_h,
                outline="green", tags="bbox"
            )

            # Only draw "X" button if this is the last one in the series
            if is_last:
                button_size = 16  # Button size
                button_x1 = scaled_x + scaled_w - button_size - 2
                button_y1 = scaled_y + 2
                button_x2 = button_x1 + button_size
                button_y2 = button_y1 + button_size

                # Draw small square button
                button_id = self.canvas.create_rectangle(
                    button_x1, button_y1, button_x2, button_y2, fill="red", tags="delete_x"
                )

                # Draw "X" inside button
                text_id = self.canvas.create_text(
                    (button_x1 + button_x2) // 2, (button_y1 + button_y2) // 2,
                    text="X", fill="white", font=("Arial", 10, "bold"), tags="delete_x"
                )
                # Map the delete button to the Series ID
                self.delete_buttons[button_id] = series_id
                self.delete_buttons[text_id] = series_id  # Ensure both rectangle and text respond to clicks

        # Bind clicks to delete
        self.canvas.tag_bind("delete_x", "<Button-1>", self.handle_delete_click)
        self.canvas.tag_bind("bbox", "<Button-3>", self.handle_extend_click)




    def handle_delete_click(self, event):
        """Handles clicks on 'X' buttons, removes the series, and updates IDs."""

        clicked_x_id = self.canvas.find_closest(event.x, event.y)[0]

        if clicked_x_id in self.delete_buttons:
            series_id_to_remove = self.delete_buttons[clicked_x_id]
            image_path = self.images[self.current_image]

            if not messagebox.askyesno(
                "Delete CellClicker Track",
                "Delete this raw CellClicker series and remove its phase selections, aggregate entry, "
                "and tracking-review record? Existing exports will be marked stale and must be rebuilt.\n\n"
                "This cannot be undone.",
                parent=self.root,
            ):
                return
            from CellClicker.project_reconciliation import delete_track_from_project
            delete_track_from_project(self.project_dir, image_path, series_id_to_remove)

            # Refresh display
            self.update_progress()

    def handle_extend_click(self, event):
        """Offer non-destructive backward extension for the clicked raw series."""
        item_id = self.canvas.find_closest(event.x, event.y)[0]
        for bbox in self.existing_bboxes:
            x, y, width, height, series_id, _ = bbox
            scale_x, scale_y = self.image_to_canvas_scale()
            if int(x * scale_x) <= event.x <= int((x + width) * scale_x) and int(y * scale_y) <= event.y <= int((y + height) * scale_y):
                if not messagebox.askyesno(
                    "Extend Track Earlier",
                    "Continue this series into earlier raw timepoints? Its phase selection will become incomplete "
                    "and the reconciled track will require review. Existing reviewed boxes are kept.",
                    parent=self.root,
                ):
                    return
                try:
                    anchor_path = find_series_anchor_for_image(self.xml_path, self.images[self.current_image], series_id)
                    start = get_series_extension_start(self.xml_path, anchor_path, series_id)
                    prepare_series_extension(self.xml_path, anchor_path, series_id)
                    earliest_path = get_relative_image_name(anchor_path, start["class_id"])
                    if not earliest_path:
                        raise ValueError("This series is already at the first available timepoint.")
                    image = Image.open(earliest_path)
                    self.start_clicker(
                        {
                            "x": int((start["x_center"] - start["width"] / 2) * image.width),
                            "y": int((start["y_center"] - start["height"] / 2) * image.height),
                            "width": int(start["width"] * image.width),
                            "height": int(start["height"] * image.height),
                        },
                        series_id=series_id,
                        next_class_id=start["class_id"] + 1,
                        image_path=earliest_path,
                        anchor_path=anchor_path,
                    )
                except Exception as exc:
                    messagebox.showerror("Extend Track Earlier", str(exc), parent=self.root)
                return





    def update_progress(self):
        """Reloads the XML data and refreshes the bounding box display."""
        self.xml_df = check_xml(self.xml_path)  # Reload XML
        self.update_image()  # Refresh display


    def start_clicker(self, bbox, series_id=None, next_class_id=0, image_path=None, anchor_path=None):
        bbox['label'] = 'u-0'
        img_path = image_path or self.images[self.current_image]
        if series_id is None:
            series_id = get_next_series_id(self.xml_path, img_path)
        ImageProcessor(
            self.root, img_path, bbox, self.xml_path, series_id, next_class_id,
            anchor_path, on_finished=self.complete_clicker_session,
        )

    def next_image(self):
        if self.current_image < len(self.images) - 1:
            self.current_image += 1
            self.frame_number.set(str(self.current_image))
            self.update_image()

    def prev_image(self):
        if self.current_image > 0:
            self.current_image -= 1
            self.frame_number.set(str(self.current_image))
            self.update_image()

    def go_to_frame(self):
        frame_number = int(self.frame_number.get())
        if 0 <= frame_number < len(self.images):
            self.current_image = frame_number
            self.update_image()
            self.focus_canvas()
        else:
            messagebox.showerror("Error", "Invalid frame number")
