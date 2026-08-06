"""Path, image-series, and YOLO bounding-box utilities for CellClicker.

Pixel boxes use ``(x_min, y_min, x_max, y_max)`` or ``(x, y, width, height)``
as named by each function; YOLO values are normalized to image dimensions.
"""

import re
import os

def convert_path_format(input_path):
    """Return ``input_path`` with separators normalized for the current OS."""
    if os.name == 'posix':
        # Running on macOS or Linux, convert to macOS-style path
        return input_path.replace('\\', '/')
    else:
        # Running on Windows, convert to Windows-style path
        return input_path.replace('/', '\\')

def get_previous_image_name(image_name):
    """Decrease the timepoint of the image by one and return the new image name."""
    # Extract the timepoint using regular expression
    match = re.search(r't(\d{3})\.png$', image_name)
    if match:
        timepoint = int(match.group(1))
        # If timepoint would become 0, return  None
        
        timepoint -=1
        if timepoint == 0:
            return None
        # Replace the old timepoint with the new one in the image name
        new_image_name = re.sub(r't\d{3}\.png$', f't{timepoint:03}.png', image_name)
        return new_image_name
    else:
        return None  # Return None if the format doesn't match
    
def get_relative_image_name(image_name, stepback):
    """Return the image filename ``stepback`` frames before a series filename."""
    """Decrease the timepoint of the image by stepback and return the new image name."""
    # Extract the timepoint using regular expression
    match = re.search(r't(\d{3})\.png$', image_name)
    if match:
        timepoint = int(match.group(1))
        # If adjusted timepoint is less than 1 return None
        timepoint -= stepback
        if timepoint < 1:
            return None
        # Replace the old timepoint with the new one in the image name
        new_image_name = re.sub(r't\d{3}\.png$', f't{timepoint:03}.png', image_name)
        return new_image_name
    else:
        return None  # Return None if the format doesn't match
    
def get_relative_label_name(image_name, stepback):
    """Decrease the timepoint of the image by stepback and return the new image name."""
    # Extract the timepoint using regular expression
    match = re.search(r't(\d{3})\.txt$', image_name)
    if match:
        timepoint = int(match.group(1))
        # If timepoint is already 0, keep it as is (or handle accordingly)
        timepoint = max(0, timepoint - stepback)
        # Replace the old timepoint with the new one in the image name
        new_image_name = re.sub(r't\d{3}\.txt$', f't{timepoint:03}.txt', image_name)
        return new_image_name
    else:
        return None  # Return None if the format doesn't match
    
    
def append_yolov5_label(label_path, x_center, y_center, width, height, img_width, img_height, class_id):
    """Append one pixel-space box as a normalized YOLO label line.

    ``x_center`` and ``y_center`` are image pixels; width/height are pixel
    extents; ``img_width``/``img_height`` define the normalization dimensions.
    """
    """Append a new YOLOv5 label to a label file, removing any existing newline characters."""
    # Normalize the coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    # Read the existing content of the file
    with open(label_path, 'r') as f:
        existing_content = f.read()

    # Remove newline characters if they exist
    existing_content = existing_content.replace("    \n", "")

    # Append the new label and add a newline
    new_label = f"{class_id} {x_center} {y_center} {width} {height}\n"

    # Append the new label to the file
    with open(label_path, 'a') as f:
        f.write(new_label)
        
def yolov5_to_xyxy(x_center, y_center, width, height, image_width, image_height):
    """Convert a normalized YOLO centre box to pixel ``(x_min, y_min, x_max, y_max)``."""
    # Convert YOLOv5 coordinates to real coordinates
    x_center_real = x_center * image_width
    y_center_real = y_center * image_height
    box_width = width * image_width
    box_height = height * image_height
    
    # Calculate top-left corner coordinates
    x1 = x_center_real - (box_width / 2)
    y1 = y_center_real - (box_height / 2)
    
    # Calculate bottom-right corner coordinates
    x2 = x_center_real + (box_width / 2)
    y2 = y_center_real + (box_height / 2)
    
    return [x1, y1, x2, y2]

def yolov5_to_xywh(x_center, y_center, width, height, image_width, image_height):
    """Convert a normalized YOLO centre box to pixel ``(x, y, width, height)``."""
    # Convert YOLOv5 coordinates to real coordinates
    x_center_real = x_center * image_width
    y_center_real = y_center * image_height
    box_width = width * image_width
    box_height = height * image_height
    
    # Calculate top-left corner coordinates
    x1 = x_center_real - (box_width / 2)
    y1 = y_center_real - (box_height / 2)
    
    # Return the bounding box in xywh format
    return [x1, y1, box_width, box_height]
