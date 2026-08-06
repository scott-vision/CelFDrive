"""Interactive CellClicker GUI for drawing boxes across image series."""

import os
import re
import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Toplevel, Label, filedialog, messagebox
from PIL import Image, ImageTk
from CellClicker.manageXML import get_series_count_for_label, append_cell_regions_xml, check_xml, remove_entry_from_xml
from CellClicker.clicker_utils import get_previous_image_name, yolov5_to_xywh


class ImageProcessor:
    """Load and normalize microscope image files for annotation display."""
    def __init__(self, master, image_path, bbox, xml_path):
        self.master = master
        self.image_path = image_path
        self.first_label = image_path
        self.bbox = bbox
        self.xml_path = xml_path
        self.class_id = 0
        
        # Create a new window for image processing
        self.image_window = Toplevel(self.master)
        self.image_window.title("Cell Clicker")

        # Display area for images
        self.canvas = tk.Canvas(self.image_window, width=200, height=200)
        self.canvas.pack()

        # Status label
        self.status_label = Label(self.image_window, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Button to manually end the session
        self.stop_button = Button(self.image_window, text="Finished", command=self.end_session)
        self.stop_button.pack(side=tk.BOTTOM)

        self.series_count = get_series_count_for_label(self.xml_path, self.first_label)
        # print("current series")
        # print(self.series_count)
        
        # Load the initial image and display it
        self.display_roi()

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
        self.canvas.config(width=self.current_w, height=self.current_h)
        # print(self.current_x, self.current_y, self.current_w, self.current_h)
        roi = self.normalize_image(img[self.current_y:self.current_y+self.current_h, self.current_x:self.current_x+self.current_w])
        self.image = ImageTk.PhotoImage(image=Image.fromarray(roi))
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)


        
        # param = {'x': x, 'y': y, 'width': w, 'height': h, 'img': img, 'image_name': self.image_path,
        #          'class_id': 0, 'series_count': series_count + 1, 'first_label': self.image_path, 'xml_path': self.xml_path}
        # self.canvas.bind("<Button-1>", lambda event, arg=param: self.click_event(event, param))
        self.canvas.bind("<Button-1>", lambda event : self.click_event(event))

    def click_event(self, event):
        """Handles mouse click events to calculate a new ROI centered on the clicked position."""
        x, y = event.x, event.y
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
                                self.current_w, self.current_h, self.x_shape, self.y_shape, self.series_count+1)

        self.class_id += 1

        self.bbox['x'] = x_start
        self.bbox['y'] = y_start
        
        self.image_path = get_previous_image_name(self.image_path)
        if self.image_path:
            self.display_roi()
        else:
            self.end_session()



    def end_session(self):
        """Closes all OpenCV windows and quits the Tkinter Toplevel."""
        self.image_window.destroy()



class ImageViewer:
    """Tk annotation window for creating bounding boxes across image series."""
    def __init__(self, root, project_dir=None):
        self.root = root
        self.root.title("Image Viewer")
        self.project_dir = os.path.normpath(project_dir) if project_dir else None
        

        # Set up the frame for navigation buttons
        frame = tk.Frame(self.root)
        frame.pack(side=tk.BOTTOM, pady=20)

         # Numeric input for frame number
        self.frame_number = tk.StringVar()
        self.frame_entry = tk.Entry(frame, textvariable=self.frame_number)
        self.frame_entry.pack(side=tk.LEFT)


        # Go to frame button
        self.btn_go_to_frame = tk.Button(frame, text="Go to Frame", command=self.go_to_frame)
        self.btn_go_to_frame.pack(side=tk.LEFT)

        # Buttons
        self.btn_inspect = tk.Button(frame, text="Update Progress", command=self.update_progress)
        self.btn_inspect.pack(side=tk.LEFT)
        self.btn_back = tk.Button(frame, text="<<", command=self.prev_image, state=tk.DISABLED)
        self.btn_back.pack(side=tk.LEFT)
        self.btn_forward = tk.Button(frame, text=">>", command=self.next_image)
        self.btn_forward.pack(side=tk.LEFT)
        self.btn_inspect = tk.Button(frame, text="Inspect", command=self.inspect_bbox)
        self.btn_inspect.pack(side=tk.RIGHT)

        # Label for image name
        self.label = tk.Label(self.root, text='', pady=10)
        self.label.pack(side=tk.BOTTOM)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root, cursor="cross")
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

        # Load images
        self.images = []
        self.current_image = 0
        self.load_images()

        # Initial image setup
        self.update_image()

        # Bind left and right arrow keys to root window
        self.root.bind("<Left>", self.left_arrow)
        self.root.bind("<Right>", self.right_arrow)

        # Set focus to entry box with a delay
        self.root.after(100, self.set_focus_to_entry_box)


    def set_focus_to_entry_box(self):
        self.frame_entry.focus_set()


    def focus_in_event(self, event):
        self.frame_entry.focus_set()

    def left_arrow(self, event):
        self.prev_image()
    def right_arrow(self, event):
        self.next_image()

    def load_images(self):
        # Ask the user for the directory
        if self.project_dir:
            directory = self.project_dir
        else:
            directory = filedialog.askdirectory(title="Select Directory with Images")
        if not directory:
            return
        
        # List all image files in the directory
        supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        directory = os.path.join(directory, "images")
        directory = os.path.normpath(directory)
        print('current image folder')
        print(directory)
        self.xml_path = os.path.join(directory, "cell_reigons.xml")
        self.xml_df = check_xml(self.xml_path)
        print(self.xml_df)
        if not self.xml_df.empty:
            self.original_image_folder = os.path.normpath(self.xml_df['PathName'][0].split("images")[0])

        else:
            self.original_image_folder = directory.split("images")[0]

        self.original_image_folder = os.path.join(self.original_image_folder, "images")
        self.original_image_folder = os.path.normpath(self.original_image_folder)

        print(f'original image folder: {self.original_image_folder}')




        self.images = [self.normalize_path(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith(supported_formats)]

        if not self.images:
            self.label.config(text="No images found!")
            return
        
        self.images.sort()

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
            filtered_df = self.xml_df[self.xml_df['PathName'].apply(lambda path: os.path.normpath(path)).str.contains(img_path)]

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
            resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.photo_img = ImageTk.PhotoImage(resized_image)

            # Update canvas dimensions
            self.canvas.config(width=new_width, height=new_height)

            # Clear and redraw the image
            self.canvas.delete("all")  # Remove previous image to prevent layering
            self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)
            self.draw_existing_bboxes()



    def resize_image(self, width, height):
        # Avoid resizing to zero to prevent PIL errors
        if width > 1 and height > 1:
            resized_image = self.original_image.resize((width, height), Image.Resampling.LANCZOS)
            self.photo_img = ImageTk.PhotoImage(resized_image)
            self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)

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
            self.resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Update the canvas size dynamically
            self.canvas.config(width=new_width, height=new_height)

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
            scale_x = self.original_image.width / self.canvas.winfo_width()
            scale_y = self.original_image.height / self.canvas.winfo_height()

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

    def draw_existing_bboxes(self):
        """Draw bounding boxes and place 'X' buttons only on the last bounding box of each series."""

        self.canvas.delete("bbox")  # Clear previous bounding boxes
        self.canvas.delete("delete_x")  # Clear previous delete markers
        self.delete_buttons = {}  # Store delete button references

        scale_x = self.canvas.winfo_width() / self.original_image.width
        scale_y = self.canvas.winfo_height() / self.original_image.height

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




    def handle_delete_click(self, event):
        """Handles clicks on 'X' buttons, removes the series, and updates IDs."""

        clicked_x_id = self.canvas.find_closest(event.x, event.y)[0]

        if clicked_x_id in self.delete_buttons:
            series_id_to_remove = self.delete_buttons[clicked_x_id]
            image_path = self.images[self.current_image]

            # Remove the series and adjust remaining IDs
            remove_entry_from_xml(self.xml_path, label_path=image_path, series_id=series_id_to_remove)

            # Refresh display
            self.update_progress()





    def update_progress(self):
        """Reloads the XML data and refreshes the bounding box display."""
        self.xml_df = check_xml(self.xml_path)  # Reload XML
        self.update_image()  # Refresh display


    def start_clicker(self, bbox):
        bbox['label'] = 'u-0'
        img_path = self.images[self.current_image]
        ImageProcessor(self.root, img_path, bbox, self.xml_path)

    def next_image(self):
        if self.current_image < len(self.images) - 1:
            self.current_image += 1
            self.update_image()

    def prev_image(self):
        if self.current_image > 0:
            self.current_image -= 1
            self.update_image()

    def go_to_frame(self):
        frame_number = int(self.frame_number.get())
        if 0 <= frame_number < len(self.images):
            self.current_image = frame_number
            self.update_image()
        else:
            messagebox.showerror("Error", "Invalid frame number")
