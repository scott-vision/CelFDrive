"""Read and update CellClicker's ``cell_regions.xml`` annotation database."""

import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
from .clicker_utils import get_previous_image_name, convert_path_format, get_relative_image_name
import pandas as pd

import os
import xml.etree.ElementTree as ET

def check_xml(xml_path):
    """Create an empty CellClicker region XML document when ``xml_path`` is absent."""
    if not os.path.exists(xml_path):
        # Create the root element
        root = ET.Element("annotations")
        # Initialize the tree structure from the root
        tree = ET.ElementTree(root)
        # Write the tree to a file
        tree.write(xml_path)
    else:
        print(f"XML file '{xml_path}' already exists.")
        
    return cell_xml_to_dataframe_absfilenames(xml_path)

# suspect label_name not needed anymore, currently used to find out if something exists
def find_labels_and_extract_rois(xml_path, label_name, image_path):
    """Return normalized YOLO regions for one image and label from region XML."""
    labels = {}
    first_label_name = label_name
    first_image_path = image_path
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # print(convert_path_format(label_name))
    # print("searching...")

    # Find the parent element with the specified label name
    parent_elem = None
    for elem in root.findall("path"):
        print(convert_path_format(elem.find("name").text))
        if convert_path_format(elem.find("name").text) == convert_path_format(label_name):
            parent_elem = elem
            # print("foundname")
            break

    if parent_elem is not None:
        for series_elem in parent_elem.findall("series"):
            series_id = series_elem.get("id")
            # print("foundseries")
            labels[series_id] = []
            label_name = first_label_name
            image_path = first_image_path 

            for label_elem in series_elem.findall("label"):
                class_id = int(label_elem.find("class_id").text)
                x_center = float(label_elem.find("x_center").text)
                y_center = float(label_elem.find("y_center").text)
                width = float(label_elem.find("width").text)
                height = float(label_elem.find("height").text)
#                 print(class_id)
                # Load the image
                image = cv2.imread(image_path)

                # Calculate ROI coordinates
                x_center_pixel = int(x_center * image.shape[1])
                y_center_pixel = int(y_center * image.shape[0])
                half_width = int(width * image.shape[1] / 2)
                half_height = int(height * image.shape[0] / 2)
#                 print(x_center_pixel, y_center_pixel)
                # Crop the ROI from the image
                roi = image[y_center_pixel - half_height:y_center_pixel + half_height,
                            x_center_pixel - half_width:x_center_pixel + half_width]

                image_path = get_previous_image_name(image_path)
#                 print(image_path)
                labels[series_id].append(roi)

            labels[series_id].reverse()


    return labels

def append_cell_regions_xml(xml_path, label_path, class_id, x_center, y_center, width, height, img_width, img_height, series):
    """Append one pixel-space annotation to ``cell_regions.xml``.

    The pixel centre box is normalized by ``img_width`` and ``img_height``
    before persistence; ``series`` identifies the time-series annotation.
    """
    # Check if the XML file already exists; if not, create it
    if not os.path.exists(xml_path):
        root = ET.Element("annotations")
    else:
        # If the XML file exists, parse it and get the root element
        tree = ET.parse(xml_path)
        root = tree.getroot()

    # Create a unique identifier based on the label file name
    identifier = os.path.splitext(os.path.basename(label_path))[0]
    identifier = label_path

    # Find or create a parent element for the label path with the same name
    parent_elem = None
    for elem in root.findall("path"):
        if elem.find("name").text == label_path:
            parent_elem = elem
            break

    if parent_elem is None:
        parent_elem = ET.Element("path")
        ET.SubElement(parent_elem, "name").text = label_path
        root.append(parent_elem)
        
        
    # Find or create a parent element for the series with the same name
    series_elem = None
    for elem in parent_elem.findall("series"):
        if elem.get("id") == str(series):
            series_elem = elem
            # print("foundseries!")
            break
       
    
    if series_elem is None:
        # print("series not found!")
        series_elem = ET.Element("series", id=str(series))
        series_elem.set("revision", "0")
        parent_elem.append(series_elem)

#     # Create a <series> element for the specified series
#     series_elem = ET.Element("series", id=str(series))
    # print("before label")
    # Normalize the coordinates
    x_center_normalized = x_center / img_width
    y_center_normalized = y_center / img_height
    width_normalized = width / img_width
    height_normalized = height / img_height

    # Create a <label> element for the YOLOv5 label
    label_elem = ET.Element("label")
    ET.SubElement(label_elem, "class_id").text = str(class_id)
    ET.SubElement(label_elem, "x_center").text = str(x_center_normalized)
    ET.SubElement(label_elem, "y_center").text = str(y_center_normalized)
    ET.SubElement(label_elem, "width").text = str(width_normalized)
    ET.SubElement(label_elem, "height").text = str(height_normalized)

    # Append the <label> element to the <series> element
    series_elem.append(label_elem)
    # print("label append")
    # Append the <series> element to the parent element
#     parent_elem.append(series_elem)
#     print("series append")

    # Save the updated XML file
    tree = ET.ElementTree(root)
    # print("end")

    tree.write(xml_path)




def get_all_label_names(xml_path):
    """List unique annotation image paths stored in region XML."""
    label_names = []

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Iterate through path elements to collect label names
    for path_elem in root.findall("path"):
        label_name = path_elem.find("name").text
        # label_name = convert_path_format(label_name)
        label_names.append(label_name)

    return label_names


def get_series_count_for_label(xml_path, label_name):
    """Return the number of annotated series for one image path."""
    if not os.path.exists(xml_path):
        return 0  # Return 0 if the XML file doesn't exist

    series_count = 0
    # label_name = convert_path_format(label_name)
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the parent element with the specified label name
    parent_elem = None
    for elem in root.findall("path"):
        if elem.find("name").text == label_name:
            parent_elem = elem
            break

    if parent_elem is not None:
        # Count the number of <series> elements within the parent element
        series_count = len(parent_elem.findall("series"))

    return series_count


def get_next_series_id(xml_path, label_name):
    """Return an unused numeric series ID without renumbering existing tracks."""
    if not os.path.exists(xml_path):
        return 1
    root = ET.parse(xml_path).getroot()
    for elem in root.findall("path"):
        if elem.findtext("name") == label_name:
            ids = [int(series.get("id")) for series in elem.findall("series") if series.get("id", "").isdigit()]
            return max(ids, default=0) + 1
    return 1


def get_all_images(xml_path, progress_callback=None):
    """Load images referenced by region XML, grouped by path and series.

    ``progress_callback`` receives ``(current, total, image_path)`` while
    images are read. Returned arrays retain image row/column axis order.
    """
    if not os.path.exists(xml_path):
        return 0  # Return 0 if the XML file doesn't exist
    
    # here we need to make this run across devices, so if the location has changed we connect via the folder name
    print(xml_path)

#     print("running")
    label_names = get_all_label_names(xml_path)
#     print(label_names)
#     print("anmes are there")
    image_names = [x.replace(".txt", ".png").replace("labels", "images") for x in label_names]
    allimages = {}
    total_items = len(label_names)
    for index, (label_path, image_path) in enumerate(zip(label_names, image_names), start=1):
#         print(label_path, image_path)
        image_path_parts = image_path.replace("\\", "/").split("/images/", 1)
        if len(image_path_parts) == 2:
            image_path = os.path.join(os.path.dirname(os.path.abspath(xml_path)), image_path_parts[1])
        print("finding: " +image_path)
        image_dict = find_labels_and_extract_rois(xml_path, label_path, image_path)
#         print(image)
        for series_id in image_dict.keys():
            allimages[(label_path, series_id)] = image_dict[series_id]
        if progress_callback is not None:
            progress_callback(index, total_items, image_path)

#     result_dict = {(text, key): value for text, inner_dict in allimages.items() for key, value in inner_dict.items()}


    return allimages



def cell_xml_to_dataframe(xml_file):
    """Read region XML into a dataframe of normalized boxes and source metadata."""
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Prepare a list to hold the data
    data_entries = []

    # Iterate through each 'path' in the XML
    for path in root.findall('path'):
        path_name = path.find('name').text
        
        for series in path.findall('series'):
            series_id = series.get('id')
            print(path_name)
            print(series_id)
        # Iterate through each 'label' within 'series'
            for label in series.findall('./label'):
                class_id = label.find('class_id').text
                print(class_id)
                x_center = label.find('x_center').text
                y_center = label.find('y_center').text
                width = label.find('width').text
                height = label.find('height').text

                # Append this entry as a dict to the list
                data_entries.append({
                    'PathName': path_name,
                    'SeriesID': series_id,
                    'ClassID': class_id,
                    'XCenter': x_center,
                    'YCenter': y_center,
                    'Width': width,
                    'Height': height
                })

    # Create a DataFrame from the list of dicts
    df = pd.DataFrame(data_entries)
    return df

def cell_xml_to_dataframe_absfilenames(xml_file):
    """Parse XML into a DataFrame with absolute file paths and series IDs."""

    tree = ET.parse(xml_file)
    root = tree.getroot()

    data_entries = []

    for path in root.findall('path'):
        path_name = path.find('name').text
        path_name_series = path.find('name').text  # Copy for decrementing over series
        
        for series in path.findall('series'):
            series_id = series.get('id')
            path_name_series = path_name

            for label in series.findall('label'):
                class_id = label.find('class_id').text
                x_center = label.find('x_center').text
                y_center = label.find('y_center').text
                width = label.find('width').text
                height = label.find('height').text

                # Append data including series ID
                data_entries.append({
                    'PathName': path_name_series,
                    'SeriesID': series_id,  # Now we track series
                    'ClassID': class_id,
                    'XCenter': x_center,
                    'YCenter': y_center,
                    'Width': width,
                    'Height': height
                })
                path_name_series = get_previous_image_name(str(path_name_series))

    return pd.DataFrame(data_entries)



def modify_class_ids(df, selected_indices, target_class_id = 2):
    """Apply a target class ID to dataframe rows identified by selected indices."""
    # selected_indices is a dict with keys as (PathName, SeriesID) and values as the selected index
    # Example: selected_indices = {('path1', '1'): 5, ('path2', '2'): 3, ...}

    modified_data = []

    for (path_name, series_id), group in df.groupby(['PathName', 'SeriesID']):
        if (path_name, series_id) in selected_indices:
            # Get the selected index for this group
            selected_index = selected_indices[(path_name, series_id)]

            # Sort the group by ClassID (if not already sorted)
            group = group.sort_values(by='ClassID')

            # Calculate new ClassIDs
            new_class_ids = list(range(len(group) - 1, selected_index - 1, -1)) + \
                            [target_class_id] + \
                            list(range(1, len(group) - selected_index))

            # Drop the last label as per your requirement
            new_class_ids = new_class_ids[:-1]

            # Update the group with new class IDs
            group = group.iloc[:-1]  # Drop the last label
            group['ClassID'] = new_class_ids
            modified_data.append(group)

    # Concatenate all modified groups into a new DataFrame
    modified_df = pd.concat(modified_data)

    return modified_df

# Example usage
# df is your original DataFrame
# selected_indices is your dictionary of selected indices for each (PathName, SeriesID)
# selected_indices = {('finder-camera4_3/labels/20220329_4_3_P1 - 1t027.txt', '1'): 5, ...}
# target_class_id is the class ID to be set for the selected index


# # Example usage
# xml_path = "cell_regions.xml"
# label_name = "example1.txt"

# series_count = get_series_count_for_label(xml_path, label_name)
# print(f"Number of series for '{label_name}': {series_count}")

# # Example usage
# xml_path = "cell_regions.xml"
# label_name = "example1.txt"

# series_count = get_series_count_for_label(xml_path, label_name)
# print(f"Number of series for '{label_name}': {series_count}")

# # Example usage
# xml_path = "cell_regions.xml"
# label_name = "example1.txt"
# image_path = "example1.jpg"

# result = find_labels_and_extract_rois(xml_path, label_name, image_path)
# print(result)

def remove_entry_from_xml(xml_path, label_path, series_id):
    """
    Removes a series from the XML and updates remaining series IDs to keep them consecutive.

    Parameters:
    - xml_path (str): Path to the XML file.
    - label_path (str): Image path corresponding to the series.
    - series_id (str or int): ID of the series to remove.
    """
    if not os.path.exists(xml_path):
        print(f"XML file '{xml_path}' does not exist.")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Locate the <path> element
    parent_elem = None
    for elem in root.findall("path"):
        if elem.find("name").text == label_path:
            parent_elem = elem
            break

    if parent_elem is None:
        print(f"No entry found for {label_path}.")
        return

    # Find and remove the target series
    series_to_remove = None
    series_list = []

    for series_elem in parent_elem.findall("series"):
        series_list.append(series_elem)  # Collect all series
        if series_elem.get("id") == str(series_id):
            series_to_remove = series_elem

    if series_to_remove:
        parent_elem.remove(series_to_remove)
        print(f"Series {series_id} removed from {label_path}.")

        # Remove <path> if there are no series left
        if not parent_elem.findall("series"):
            root.remove(parent_elem)
            print(f"Removed {label_path} as no series remain.")

        # Save updated XML
        tree.write(xml_path)
        print("XML updated successfully.")
    else:
        print(f"Series {series_id} not found in {label_path}.")


def get_series_extension_start(xml_path, label_path, series_id):
    """Return the earliest raw box and source index for an existing series."""
    root = ET.parse(xml_path).getroot()
    for path_elem in root.findall("path"):
        if path_elem.findtext("name") != label_path:
            continue
        for series in path_elem.findall("series"):
            if series.get("id") != str(series_id):
                continue
            labels = series.findall("label")
            if not labels:
                raise ValueError(f"CellClicker series {series_id} has no labels to extend.")
            label = max(labels, key=lambda item: int(item.findtext("class_id")))
            return {
                "class_id": int(label.findtext("class_id")),
                "x_center": float(label.findtext("x_center")),
                "y_center": float(label.findtext("y_center")),
                "width": float(label.findtext("width")),
                "height": float(label.findtext("height")),
            }
    raise ValueError(f"CellClicker series {series_id} was not found.")


def prepare_series_extension(xml_path, label_path, series_id):
    """Increment a series revision and move it to the end of its source path."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for path_elem in root.findall("path"):
        if path_elem.findtext("name") != label_path:
            continue
        for series in path_elem.findall("series"):
            if series.get("id") != str(series_id):
                continue
            current_revision = int(series.get("revision", "0"))
            series.set("revision", str(current_revision + 1))
            path_elem.remove(series)
            path_elem.append(series)
            tree.write(xml_path)
            return current_revision + 1
    raise ValueError(f"CellClicker series {series_id} was not found for extension.")


def find_series_anchor_for_image(xml_path, image_path, series_id):
    """Find the anchor path for a displayed frame belonging to one raw series."""
    normalized_image = os.path.normpath(image_path).replace("\\", "/")
    root = ET.parse(xml_path).getroot()
    for path_elem in root.findall("path"):
        anchor = path_elem.findtext("name")
        for series in path_elem.findall("series"):
            if series.get("id") != str(series_id):
                continue
            for label in series.findall("label"):
                source_index = int(label.findtext("class_id"))
                frame = get_relative_image_name(anchor, source_index)
                if frame and os.path.normpath(frame).replace("\\", "/") == normalized_image:
                    return anchor
    raise ValueError(f"Could not identify raw series {series_id} for displayed image {image_path!r}.")
