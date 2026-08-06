"""Legacy Tk interface for selecting one image from each annotated series."""

import tkinter as tk
import os
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from .name_selector import run_name_selector
from .manageXML import append_cell_regions_xml, find_labels_and_extract_rois, get_all_label_names, get_series_count_for_label, get_all_images, cell_xml_to_dataframe
from .user_xml import store_results, read_xml_to_dataframe
from .convert_selections import modify_class_ids, append_modified_labels
import numpy as np


def load_selector(image_dict, set_index):
    """Display the legacy selector for image series and return selected indices."""
    selected_indices = []
    image_sets =[]
    
    for (image_name, series_id) in image_dict.keys():
        image_sets.append(image_dict[image_name, series_id])

#     for image_dict in image_dict:
#         for value in image_dict.values():
#             image_sets.append(value)
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    print("loading "+str(len(image_sets))+ "image sets")
    display_set(image_sets, set_index, selected_indices, root)
    root.mainloop() 
    return selected_indices 
    


def normalize_image(image):
    """Normalize a two-dimensional image array to displayable ``uint8`` values."""

    # Normalize the image to the range [0, 1]
    # Normalize the image to the range [0, 255]
    image = (image - image.min()) / (image.max() - image.min()) * 255

    # Convert the image to uint8 data type
    image = image.astype(np.uint8)

    return image
    
def display_set(image_sets, set_index,selected_indices, root):
    """Render one image series and its selection controls."""
    
    
    window = tk.Toplevel()
    window.title(f"Select Image from Set {set_index + 1}")
    series = image_sets[set_index]
    set_len = len(series)
    for i, img_array in enumerate(series):
        img_array = normalize_image(img_array)
        img = Image.fromarray(img_array)
        img.thumbnail((100, 100))  # Resize to thumbnail
        img_tk = ImageTk.PhotoImage(img)
        
        btn = tk.Button(window, image=img_tk, command=lambda i=i, win=window: on_selection_clicked(i, win, image_sets, selected_indices, root, set_len))
        btn.image = img_tk  # Keep a reference, prevent GC
        btn.grid(row=0, column=i)

    # Blurry button
    blurry_btn = tk.Button(window, text="Blurry", command=lambda win=window: on_blurry_clicked(win, image_sets, selected_indices, root))
    blurry_btn.grid(row=1, columnspan=len(series))

    window.protocol("WM_DELETE_WINDOW", lambda win=window: on_window_close(win, root))
    return selected_indices 


def on_window_close(window, root):
    """Close the current selector window and stop its Tk event loop."""
    window.destroy()
    root.quit()  # Quit the Tkinter main loop

def on_blurry_clicked(window, image_sets, selected_indices, root):
    """Store a blurry-series marker and advance to the next image set."""
    selected_indices.append(-1)
    window.destroy()
    next_set_index = len(selected_indices)
    if next_set_index < len(image_sets):
        display_set(image_sets, next_set_index, selected_indices, root)
    else:
        messagebox.showinfo("Completed", "Selection completed.")
        print("Selected indices:", selected_indices)
        root.quit()

def on_selection_clicked(index, window, image_sets, selected_indices, root,set_len):
    """Store the selected reverse-time index and advance the selector."""
    selected_indices.append(set_len-1-index)
    window.destroy()
    next_set_index = len(selected_indices)
    if next_set_index < len(image_sets):
        display_set(image_sets,  next_set_index, selected_indices, root)
    else:
        messagebox.showinfo("Completed", "Selection completed.")
        print("Selected indices:", selected_indices)
        window.destroy()
        root.quit()
        return selected_indices 
        


def load_ui(cell_xml):
    """Load region XML, select an output XML, and launch the legacy selector."""
    # Call the UI function to run the name selector
    name_xml = run_name_selector("select_xmls")
    print(f"Selected XML: {name_xml}")
    images_dict = get_all_images(cell_xml)
    selected_indicies = load_selector(images_dict, 0)
    store_results(images_dict, selected_indicies, name_xml)

def load_ui_from_folder():
    """Prompt for a project folder and launch the legacy selector workflow."""
    # Call the UI function to run the name selector

    directory = filedialog.askdirectory(title="Select Directory with Images")
    if not directory:
        return
    selections_folder = os.path.join(directory, "user_selections")
    os.makedirs(selections_folder, exist_ok=True)
    
    cell_xml = os.path.join(os.path.join(directory, "images"), "cell_reigons.xml")
    cell_xml = os.path.normpath(cell_xml)

    name_xml = run_name_selector(selections_folder)
    print(f"Selected XML: {name_xml}")
    images_dict = get_all_images(cell_xml)
    selected_indicies = load_selector(images_dict, 0)
    store_results(images_dict, selected_indicies, name_xml)
    
def xml_to_labels(name_xml, cell_xml):
    """Convert legacy selection XML and region XML to modified labels."""
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    
    target_class_id = 2
    modified_df = modify_class_ids(cell_df, user_df, target_class_id)
    append_modified_labels(modified_df)
    
    
def debug_xml_to_labels(name_xml, cell_xml):
    """Return legacy user and region XML dataframes for inspection."""
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    return user_df, cell_df

def debug_xml_to_labels2(name_xml, cell_xml):
    """Return the modified legacy label dataframe without writing label files."""
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    
    target_class_id = 2
    modified_df = modify_class_ids(cell_df, user_df, target_class_id)
    return modified_df
#     target_class_id = 2
#     modified_df = modify_class_ids(cell_df, user_df, target_class_id)
#     append_modified_labels(modified_df)
        
