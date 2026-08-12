"""Aggregate phase selections and generate phase-specific YOLO labels.

Image coordinates are pixels until converted to normalized YOLO
``(class_id, x_center, y_center, width, height)`` records.
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import os
from .clicker_utils import get_relative_image_name
from .workflow_state import format_stale_selection_report, raw_revision_fingerprint, stale_selection_report


def xyxy_to_yolov5(x_min, y_min, x_max, y_max, img_width, img_height):
    """Convert pixel ``(x_min, y_min, x_max, y_max)`` bounds to normalized YOLO."""
    # Calculate bounding box center
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    # Calculate bounding box width and height
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    # Normalize coordinates
    center_x_normalized = center_x / img_width
    center_y_normalized = center_y / img_height
    box_width_normalized = box_width / img_width
    box_height_normalized = box_height / img_height
    
    return center_x_normalized, center_y_normalized, box_width_normalized, box_height_normalized


def adjust_bbox_via_threshold(image_path, label, x_center_norm, y_center_norm, width_norm, height_norm):
    """Tighten a normalized YOLO box using Otsu thresholding in its image ROI.

    Returns ``(class_id, x_center, y_center, width, height)`` with normalized
    values. Image X/Y coordinates are columns/rows respectively.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at the specified path: {image_path}")

    img_h, img_w = image.shape[:2]
    print(f"Image dimensions: height={img_h}, width={img_w}")

    # Convert YOLO format to absolute bounding box coordinates
    x_center = x_center_norm * img_w
    y_center = y_center_norm * img_h
    width = width_norm * img_w
    height = height_norm * img_h
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    print(f"Initial Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

    # Expand the bounding box by 10%
    expansion_factor = 0.1
    width_expand = int(width * expansion_factor)
    height_expand = int(height * expansion_factor)
    x_min = max(x_min - width_expand, 0)
    y_min = max(y_min - height_expand, 0)
    x_max = min(x_max + width_expand, img_w)
    y_max = min(y_max + height_expand, img_h)

    print(f"Expanded Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

    # Ensure the bounding box remains within the image boundaries
    x_min = max(0, min(x_min, img_w))
    y_min = max(0, min(y_min, img_h))
    x_max = max(0, min(x_max, img_w))
    y_max = max(0, min(y_max, img_h))

    # Crop the relevant part of the image
    crop_img = image[y_min:y_max, x_min:x_max]

    # Convert to grayscale
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return label, x_center_norm, y_center_norm, width_norm, height_norm

    # Find the largest and possibly the second largest contour based on area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    if label in [4, 5] and len(contours) > 1 and cv2.contourArea(contours[1]) >= 0.5 * cv2.contourArea(largest_contour):
        x, y, w, h = cv2.boundingRect(np.vstack([largest_contour, contours[1]]))
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)

    print(f"Contour Bounding Box: x={x}, y={y}, w={w}, h={h}")

    # Adjust coordinates based on the original crop
    x_min_new = max(x_min + x, 0)
    y_min_new = max(y_min + y, 0)
    x_max_new = min(x_min_new + w, img_w)
    y_max_new = min(y_min_new + h, img_h)

    print(f"New Bounding Box Coordinates: x_min_new={x_min_new}, y_min_new={y_min_new}, x_max_new={x_max_new}, y_max_new={y_max_new}")

    x_center_new, y_center_new, width_new, height_new = xyxy_to_yolov5(x_min_new, y_min_new, x_max_new, y_max_new, img_w, img_h)
    print(w, h)

    print(f"Normalized Output: x_center_new={x_center_new}, y_center_new={y_center_new}, width_new={width_new}, height_new={height_new}")

    return label, x_center_new, y_center_new, width_new, height_new

# def xyxy_to_yolov5(x_min, y_min, x_max, y_max, img_w, img_h):
#     """Convert bounding box from (x_min, y_min, x_max, y_max) format to YOLO format."""
#     x_center = (x_min + x_max) / 2 / img_w
#     y_center = (y_min + y_max) / 2 / img_h
#     width = (x_max - x_min) / img_w
#     height = (y_max - y_min) / img_h
#     return x_center, y_center, width, height


def parse_xml_for_phases(xml_file):
    """Read selected phase indices from user XML keyed by image path and series."""
    """ Parse XML and get phase indices for each image and series. """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_data = {}
    for entry in root.findall('DataEntry'):
        path = entry.find('PathName').text
        series_id = entry.find('SeriesID').text
        indices = {
            phase.tag: int(phase.text) for phase in entry
            if phase.tag in ['prophase','earlyprometaphase','prometaphase', 'metaphase', 'anaphase', 'telophase'] and phase.text.isdigit()
        }
        image_data[(path, series_id)] = indices
    return image_data

def parse_xml_for_phases_resume(xml_file):
    """ Parse XML and get phase indices for each image and series. """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_data = {}
    for entry in root.findall('DataEntry'):
        path = entry.find('PathName').text
        series_id = entry.find('SeriesID').text
        indices = {
            phase.tag: int(phase.text) for phase in entry
            if phase.tag in ['prophase','earlyprometaphase','prometaphase', 'metaphase', 'anaphase', 'telophase'] 
        }
        image_data[(path, series_id)] = indices
    return image_data

def parse_xml_for_labels(xml_file):
    """Read CellClicker region XML into label records grouped by image and series."""
    """ Parse annotation XML and return structured label data. """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels_data = {}
    for path_element in root.findall('path'):
        image_path = path_element.find('name').text
        for series in path_element.findall('series'):
            series_id = series.get('id')
            labels = []
            for label in series.findall('label'):
                label_info = {
                    'class_id': int(label.find('class_id').text),
                    'x_center': float(label.find('x_center').text),
                    'y_center': float(label.find('y_center').text),
                    'width': float(label.find('width').text),
                    'height': float(label.find('height').text)
                }
                labels.append(label_info)
            labels_data[(image_path, series_id)] = labels
    return labels_data

def generate_filename(base_path, offset):
    """ Generate a new filename based on an offset from the base filename """
    base_name = os.path.basename(base_path)
    prefix, number, ext = base_name.rsplit('_', 2)
    new_number = int(number[1:]) + offset
    new_filename = f"{prefix}_t{str(new_number).zfill(3)}.{ext}"
    return new_filename

def get_phase(new_index, phases):
    """Return the phase active at a chronological image index."""
    # Iterate over the phases
    for phase, value in phases.items():
        # Check if the new index is larger than the corresponding value
        if new_index >= value:
            return phase
    # If the new index is less than or equal to all values, return None
    return None

def create_yolo_labels(phase_data, labels_data, output_dir, user, imgpath):
    """Write phase-classified normalized YOLO labels from parsed XML mappings."""
    """ Generate YOLO label files considering class_id as filename reference. """
    os.makedirs(output_dir, exist_ok=True)
    
    class_dict = {'prophase': 0, 'earlyprometaphase': 1, 'prometaphase': 2, 'metaphase': 3, 'anaphase': 4, 'telophase': 5}
    # limits number of the final class that can be shown
    telophase_limit = 4
    offset = telophase_limit -1

    for (path, series_id), indices in labels_data.items():
        total_series_length = len(indices)
        phases = get_filtered_phases(phase_data, path, series_id, imgpath)
        
        # skip if nothing returned
        if not phases:
            continue
        
        print(len(indices))
        telophase_max_index = get_telophase_max_index(phases, offset, total_series_length)
        
        print(f"telophase_max_index: {telophase_max_index}")
        sorted_phases = sort_phases(phases)
        
        for label in indices[offset:]:
            print("label: ")
            print(label)
            process_label(label, indices[offset:], sorted_phases, class_dict, path, user, total_series_length)

def get_filtered_phases(phase_data, path, series_id, imgpath):
    """Return valid phase indices for one image path and series."""
    # remove the phase if skipped
    phases = phase_data.get((path, series_id), {})
    path = imgpath + "/images/" + path.split("/images/")[1]
    return {k: v for k, v in phases.items() if v != -1}

def get_telophase_max_index(phases, telophase_limit, total_indices):
    """Calculate the final usable chronological index for telophase labels."""
    #  as the list of phases is reversed and the loop counts down we need total indicies - where we want to stop, in order to determine where to start, it also must substract 1 in order to adjust for slicing
    telophase_max_index = total_indices
    if 'telophase' in phases:

        telophase_max_index = min(int(phases['telophase']) + telophase_limit, total_indices)
    return telophase_max_index

def sort_phases(phases):
    """Return phase indices ordered from earliest to latest selected phase."""
    # reverse the phases dict
    return dict(sorted(phases.items(), key=lambda item: item[1], reverse=True))

def process_label(label, indices, sorted_phases, class_dict, path, user, total_series_length):
    """Assign one source label to a phase class and adjusted output frame."""
    # as the set is backwards - class id to get new class id
    new_index = total_series_length - int(label['class_id']) -1

    selected_phase = get_phase(new_index, sorted_phases)
    
    if selected_phase:
        new_class_id = class_dict[selected_phase]
        filename = get_relative_image_name(path, int(label['class_id']))
        adjusted_label = adjust_bbox_via_threshold(filename, new_class_id, label['x_center'], label['y_center'], label['width'], label['height'])
        write_yolo_label(filename, user, new_class_id, adjusted_label)

def write_yolo_label(filename, user, new_class_id, adjusted_label):
    """Append a normalized YOLO box to the selected user's label file."""
    yolo_label = f"{new_class_id} {adjusted_label[1]} {adjusted_label[2]} {adjusted_label[3]} {adjusted_label[4]}"
    filename = filename.replace('.png', '.txt').replace('images', f"{user}_labels")
    with open(filename, 'a') as file:
        file.write(yolo_label + '\n')



def convert_selections_multiphase(user_xml, cell_reigons_xml, new_label_folder, user, imgpath):
    """Convert one user's phase XML and region XML into YOLO label files.

    The output contains normalized class-ID centre boxes and is written beneath
    ``new_label_folder``.
    """

    phase_data = parse_xml_for_phases(user_xml)
    # print(phase_data)
    labels_data = parse_xml_for_labels(cell_reigons_xml)
    # print(labels_data)
    create_yolo_labels(phase_data, labels_data, new_label_folder, user, imgpath)


def calculate_median_handling_negatives(group):
    """ Calculate median, if median is -1, recalculate ignoring -1 values. """
    median_value = group.median()
    if median_value == -1:
        valid_values = group[group != -1]
        if not valid_values.empty:
            return valid_values.median()
    return median_value


def parse_xml_for_phases_df(xml_file):
    """ Parse XML and get phase indices for each image and series. """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract the username from the file path
    user = os.path.basename(xml_file).split('.')[0]
    
    data = []
    for entry in root.findall('DataEntry'):
        path = entry.find('PathName').text
        series_id = entry.find('SeriesID').text
        for phase in entry:
            if phase.tag in ['prophase', 'earlyprometaphase', 'prometaphase', 'metaphase', 'anaphase', 'telophase']:
                data.append({
                    'User': user,
                    'PathName': path,
                    'SeriesID': series_id,
                    'Phase': phase.tag,
                    'Index': int(phase.text)
                })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    return df

def calculate_median_handling_negatives(group):
    """ Calculate median, if median is -1, recalculate ignoring -1 values. """
    median_value = group.median()
    if median_value == -1:
        valid_values = group[group != -1]
        if not valid_values.empty:
            return valid_values.median()
    return median_value

def aggregate_phase_data(xml_files):
    """ Aggregate phase data from multiple XML files, calculating the median index for each group. """
    all_data = pd.DataFrame()

    # Parse each XML file and concatenate the results
    for xml_file in xml_files:
        df = parse_xml_for_phases_df(xml_file)
        all_data = pd.concat([all_data, df], ignore_index=True)

    # Group by PathName, SeriesID, and Phase, and calculate the median index
    aggregated_data = all_data.groupby(['PathName', 'SeriesID', 'Phase'])['Index'].apply(calculate_median_handling_negatives).reset_index()
    
    return aggregated_data

def adjust_phase_indices(aggregated_data):
    """ Adjust phase indices to ensure each phase is later than the previous one. """
    phase_order = ['prophase', 'earlyprometaphase', 'prometaphase', 'metaphase', 'anaphase', 'telophase']
    adjusted_data = []

    grouped = aggregated_data.groupby(['PathName', 'SeriesID'])

    for (path, series_id), group in grouped:
        group = group.set_index('Phase').reindex(phase_order).reset_index()
        previous_index = -1

        for _, row in group.iterrows():
            if row['Index'] != -1:
                new_index = max(previous_index + 1, int(np.floor(row['Index'])))
                previous_index = new_index
                row['Index'] = new_index
            adjusted_data.append(row)

    return pd.DataFrame(adjusted_data)

def create_new_xml_file(aggregated_data, output_file, raw_fingerprint=None):
    """ Create a new XML file with the aggregated and adjusted median data. """
    phase_order = ['prophase', 'earlyprometaphase', 'prometaphase', 'metaphase', 'anaphase', 'telophase']

    new_root = ET.Element('Data')
    if raw_fingerprint is not None:
        new_root.set("raw_revision_fingerprint", raw_fingerprint)

    grouped = aggregated_data.groupby(['PathName', 'SeriesID'])

    for (path, series_id), group in grouped:
        new_entry = ET.Element('DataEntry')
        ET.SubElement(new_entry, 'PathName').text = path
        ET.SubElement(new_entry, 'SeriesID').text = series_id

        for phase in phase_order:
            phase_elem = ET.SubElement(new_entry, phase)
            matching_row = group[group['Phase'] == phase]
            if not matching_row.empty:
                phase_elem.text = str(int(matching_row['Index'].values[0]))
            else:
                phase_elem.text = '-1'  # Default to -1 if no matching phase found
        
        new_root.append(new_entry)
    
    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_file)

def aggregate_xml(xml_files, output_file, cell_regions_xml=None):
    """Aggregate complete current-revision user selections and write ``output_file``."""
    if cell_regions_xml is not None:
        stale = stale_selection_report(cell_regions_xml, xml_files)
        if stale:
            raise ValueError(format_stale_selection_report(stale))
    aggregated_data = aggregate_phase_data(xml_files)
    adjusted_data = adjust_phase_indices(aggregated_data)
    create_new_xml_file(
        adjusted_data, output_file,
        raw_fingerprint=raw_revision_fingerprint(cell_regions_xml) if cell_regions_xml else None,
    )


