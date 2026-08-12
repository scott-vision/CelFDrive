"""Tk interface for assigning ordered cell-cycle phases to image series.

Selections are persisted as phase indices in user XML; image arrays are shown
as two-dimensional grayscale data with rows first and columns second.
"""

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .name_selector import run_name_selector
from .manageXML import append_cell_regions_xml, find_labels_and_extract_rois, get_all_label_names, get_series_count_for_label, get_all_images, cell_xml_to_dataframe
from .user_xml import store_results, store_results_multiclass, read_xml_to_dataframe
from .convert_selections import modify_class_ids, append_modified_labels
from .convert_selections_multiphase import parse_xml_for_phases, parse_xml_for_phases_resume
from .workflow_state import entry_is_current, raw_track_revisions, selection_entries_by_track


def _finish_selector(root):
    owns_root = bool(getattr(root, "_selector_owns_root", False))
    if owns_root:
        root.quit()
        root.destroy()


def _load_selected_indices_from_xml(image_keys, name_xml, revisions=None):
    if not name_xml or not os.path.exists(name_xml):
        return []

    stored_selections = parse_xml_for_phases_resume(name_xml)
    entries = selection_entries_by_track(name_xml)
    selected_indices = []
    for image_key in image_keys:
        revision = (revisions or {}).get((image_key[0], str(image_key[1])), 0)
        entry = entries.get((image_key[0], str(image_key[1])))
        selected_indices.append(
            dict(stored_selections.get(image_key, {})) if entry_is_current(entry, revision) else {}
        )
    return selected_indices


def _load_images_with_progress(cell_xml, root):
    loading_window = tk.Toplevel(root)
    loading_window.title("Loading Image Data")
    loading_window.geometry("420x120+200+200")
    loading_window.resizable(False, False)
    loading_window.transient(root)
    loading_window.lift()

    loading_label = tk.Label(
        loading_window,
        text="Loading tracked image sets from cell_reigons.xml...",
        anchor=tk.W,
        justify=tk.LEFT,
    )
    loading_label.pack(fill=tk.X, padx=12, pady=(12, 8))

    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(
        loading_window,
        orient="horizontal",
        mode="determinate",
        maximum=1,
        variable=progress_var,
        length=380,
    )
    progress_bar.pack(fill=tk.X, padx=12, pady=(0, 8))

    progress_text = tk.Label(
        loading_window,
        text="0 / 0 image groups loaded",
        anchor=tk.W,
        justify=tk.LEFT,
    )
    progress_text.pack(fill=tk.X, padx=12, pady=(0, 12))
    loading_window.update()

    result = {}
    progress_state = {
        "current": 0,
        "total": 1,
        "image_path": "",
    }

    def report_progress(current, total, image_path):
        progress_state["current"] = current
        progress_state["total"] = max(1, total)
        progress_state["image_path"] = image_path

    def worker():
        try:
            result["images_dict"] = get_all_images(cell_xml, progress_callback=report_progress)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while thread.is_alive():
        progress_bar.configure(maximum=progress_state["total"])
        progress_var.set(progress_state["current"])
        if progress_state["image_path"]:
            loading_label.config(text=f"Loading {os.path.basename(progress_state['image_path'])}")
        progress_text.config(
            text=f"{progress_state['current']} / {progress_state['total']} image groups loaded"
        )
        root.update()
        time.sleep(0.05)

    progress_bar.configure(maximum=progress_state["total"])
    progress_var.set(progress_state["current"])
    loading_window.destroy()

    if "error" in result:
        raise result["error"]

    return result.get("images_dict", {})


def load_selector(image_dict, set_index, phases, name_xml, parent=None, revisions=None):
    """Open the phase selector for image series and return selected indices.

    ``image_dict`` maps ``(image_path, series_id)`` to ordered image arrays.
    ``phases`` gives the chronological phase vocabulary; selections are saved to
    ``name_xml`` by the GUI callbacks.
    """

    selected_indices = []
    image_sets = []
    image_keys = []

    if parent is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root._selector_owns_root = True
    else:
        root = parent
        root._selector_owns_root = False

    loading_window = tk.Toplevel(root)
    loading_window.title("Loading Image Selector")
    loading_window.geometry("420x120+200+200")
    loading_window.resizable(False, False)
    loading_window.transient(root)
    loading_window.lift()

    loading_label = tk.Label(loading_window, text="Preparing image sets...", anchor=tk.W, justify=tk.LEFT)
    loading_label.pack(fill=tk.X, padx=12, pady=(12, 8))

    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(
        loading_window,
        orient="horizontal",
        mode="determinate",
        maximum=max(1, len(image_dict)),
        variable=progress_var,
        length=380,
    )
    progress_bar.pack(fill=tk.X, padx=12, pady=(0, 8))

    progress_text = tk.Label(loading_window, text=f"0 / {len(image_dict)} image sets", anchor=tk.W, justify=tk.LEFT)
    progress_text.pack(fill=tk.X, padx=12, pady=(0, 12))
    loading_window.update()

    set_count = 0

    for (image_name, series_id) in image_dict.keys():
        set_count+=1
        print(f'{set_count}: {image_name}')
        image_keys.append((image_name, str(series_id)))
        image_sets.append(image_dict[(image_name, series_id)])
        loading_label.config(text=f"Loading {os.path.basename(image_name)}")
        progress_var.set(set_count)
        progress_text.config(text=f"{set_count} / {len(image_dict)} image sets")
        loading_window.update()

    print(f"Loading {len(image_sets)} image sets")
    loading_window.destroy()
    selected_indices = _load_selected_indices_from_xml(image_keys, name_xml, revisions=revisions)
    next_set_index, next_phase = _find_resume_position(selected_indices, phases)
    if next_set_index >= len(image_sets):
        # Keep the ordered list available for inspection and correction even
        # when every track is currently complete.
        next_set_index, next_phase = min(set_index, max(0, len(image_sets) - 1)), phases[0]
    display_set(image_sets, image_keys, next_set_index, selected_indices, root, next_phase, phases, name_xml)
    if getattr(root, "_selector_owns_root", False):
        root.mainloop()
    return selected_indices

def normalize_image(image):
    """Scale one 2-D image array to displayable ``uint8`` grayscale values."""
    minimum = image.min()
    maximum = image.max()
    if maximum == minimum:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - minimum) / (maximum - minimum) * 255).astype(np.uint8)

# def normalize_image(image):
#         """Applies CLAHE to an image to enhance contrast locally."""
#         # Convert image to grayscale if it is in color
#         if len(image.shape) == 3:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Create a CLAHE object
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         cl1 = clahe.apply(image)

#         return cl1

# need to add pick up where we left off
def display_set(image_sets, image_keys, set_index, selected_indices, root, phase, phases, name_xml, window=None):
    """Render one series and controls for assigning its current phase index."""
    if set_index < 0 or set_index >= len(image_sets):
        messagebox.showinfo("Completed", "All selections completed.")
        _finish_selector(root)
        return

    if window is None or not window.winfo_exists():
        window = tk.Toplevel(root)
    else:
        for child in window.winfo_children():
            child.destroy()
    window.title(f"Select First frame visible for {phase.capitalize()} - Set {set_index + 1} of {len(image_sets)}")

    series = image_sets[set_index]
    source_image, series_id = image_keys[set_index]
    set_len = len(series)
    max_images_per_row = 13 
    min_window_width = 100*max_images_per_row  # Calculate minimum width based on max images per row and their thumbnail size
    window.minsize(min_window_width, 0) 

    

    tk.Label(
        window,
        text=f"Track {set_index + 1}/{len(image_sets)}  |  Series {series_id}  |  Source image: {source_image}",
        anchor=tk.W,
        justify=tk.LEFT,
    ).grid(row=0, column=0, columnspan=max_images_per_row, sticky='w', padx=4, pady=(0, 2))

    photo_images = []  # To store PhotoImage references and prevent garbage collection
    selected_index_for_phase = None
    selected_text = "Current selected: none"
    if len(selected_indices) > set_index:
        phase_value = selected_indices[set_index].get(phase)
        if isinstance(phase_value, int):
            selected_index_for_phase = phase_value
            selected_text = f"Current selected: {phase_value}"
        elif phase_value == 'skipped':
            selected_text = "Current selected: skipped"
        elif phase_value == 'blurry':
            selected_text = "Current selected: blurry"

    selected_label = tk.Label(
        window,
        text=selected_text,
        anchor=tk.W,
        justify=tk.LEFT,
        fg="#22cc22" if selected_index_for_phase is not None else "#cccccc",
    )
    selected_label.grid(row=1, column=0, columnspan=max_images_per_row, sticky='w', padx=4, pady=(0, 6))

    window.bind(
        "<Left>",
        lambda event: go_back(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml),
    )
    window.bind(
        "<Right>",
        lambda event: on_next_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml),
    )

    for i, img_array in enumerate(series):
        row = i // max_images_per_row + 2  # Determine which row to place the image
        column = i % max_images_per_row  # Determine which column to place the image

        img_array = normalize_image(img_array)
        img = Image.fromarray(img_array).convert("RGB")
        img.thumbnail((100, 100))
        if i == selected_index_for_phase:
            draw = ImageDraw.Draw(img)
            outline_color = "#22cc22"
            inset = 2
            line_width = 3
            for offset in range(line_width):
                draw.rectangle(
                    (
                        inset + offset,
                        inset + offset,
                        max(inset + offset, img.width - 1 - inset - offset),
                        max(inset + offset, img.height - 1 - inset - offset),
                    ),
                    outline=outline_color,
                )
        img_tk = ImageTk.PhotoImage(img)
        photo_images.append(img_tk)

        button_frame = tk.Frame(window, bd=0)
        btn = tk.Button(
            button_frame,
            image=img_tk,
            command=lambda i=i: on_selection_clicked(i, window, image_sets, image_keys, selected_indices, root, set_len, phase, set_index, phases, name_xml),
            relief=tk.FLAT,
            bd=0,
        )
        btn.image = img_tk  # Keep a reference
        btn.pack()
        button_frame.grid(row=row, column=column, padx=1, pady=1)

    window.geometry("+100+100")  # Optional: Position the window at a specific location

    # Buttons for skipping phase and marking blurry
    button_row = (set_len - 1) // max_images_per_row + 3
    back_btn = tk.Button(window, text="Back", command=lambda: go_back(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml))
    back_btn.grid(row=button_row, column=0, sticky='ew')

    next_btn = tk.Button(window, text="Next (No Change)", command=lambda: on_next_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml))
    next_btn.grid(row=button_row, column=1, sticky='ew')

    skip_btn = tk.Button(window, text="Skip Phase", command=lambda: on_skip_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml))
    skip_btn.grid(row=button_row, column=2, sticky='ew')

    blurry_btn = tk.Button(window, text="Mark as Blurry", command=lambda: on_blurry_clicked(window, image_sets, image_keys, selected_indices, root, set_index, phases, name_xml))
    blurry_btn.grid(row=button_row, column=3, sticky='ew')

    resume_btn = tk.Button(window, text="Resume", command=lambda: on_resume_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml))
    resume_btn.grid(row=button_row, column=4, sticky='ew')

    todo_btn = tk.Button(
        window,
        text="Jump to Next TODO",
        command=lambda: jump_to_next_todo(
            window, image_sets, image_keys, selected_indices, root, set_index, phases, name_xml,
        ),
    )
    todo_btn.grid(row=button_row, column=5, sticky='ew')

    tk.Label(window, text="Track:").grid(row=button_row, column=6, sticky='e', padx=(8, 2))
    track_number_var = tk.StringVar(value=str(set_index + 1))
    track_number_entry = tk.Entry(window, textvariable=track_number_var, width=6)
    track_number_entry.grid(row=button_row, column=7, sticky='w')
    track_number_entry.bind(
        "<Return>",
        lambda _event: go_to_track(
            track_number_var.get(), window, image_sets, image_keys, selected_indices,
            root, phase, phases, name_xml,
        ),
    )
    tk.Button(
        window,
        text="Go to Track",
        command=lambda: go_to_track(
            track_number_var.get(), window, image_sets, image_keys, selected_indices,
            root, phase, phases, name_xml,
        ),
    ).grid(row=button_row, column=8, sticky='ew', padx=(2, 0))
    # print(set_index)

    save_btn = tk.Button(window, text="Save", command=lambda: on_save_clicked(selected_indices, phases, name_xml))
    save_btn.grid(row=button_row, column=9, sticky='ew')
    tk.Button(
        window,
        text="Close Selector",
        command=lambda: close_selector(window, root),
    ).grid(row=button_row, column=10, sticky='ew', padx=(4, 0))

def _find_resume_position(selected_indices, phases):
    for current_set_index, phase_selection in enumerate(selected_indices):
        for phase_name in phases:
            if phase_name not in phase_selection:
                return current_set_index, phase_name
    return len(selected_indices), phases[0]


def jump_to_next_todo(window, image_sets, image_keys, selected_indices, root, set_index, phases, name_xml):
    """Navigate to the next track with a missing or revision-stale phase selection."""
    for offset in range(1, len(image_sets) + 1):
        candidate_index = (set_index + offset) % len(image_sets)
        selections = selected_indices[candidate_index]
        for phase in phases:
            if phase not in selections:
                display_set(
                    image_sets, image_keys, candidate_index, selected_indices,
                    root, phase, phases, name_xml, window=window,
                )
                return
    messagebox.showinfo("Jump to Next TODO", "No incomplete tracks remain.")


def go_to_track(track_number, window, image_sets, image_keys, selected_indices, root, phase, phases, name_xml):
    """Open a one-based track number without changing its stored selection."""
    try:
        set_index = int(track_number) - 1
    except (TypeError, ValueError):
        messagebox.showerror("Go to Track", "Enter a whole track number.")
        return
    if not 0 <= set_index < len(image_sets):
        messagebox.showerror("Go to Track", f"Track number must be from 1 to {len(image_sets)}.")
        return
    display_set(
        image_sets, image_keys, set_index, selected_indices, root,
        phase, phases, name_xml, window=window,
    )


def close_selector(window, root):
    """Close Phase Selector only when the user explicitly chooses to do so."""
    window.destroy()
    _finish_selector(root)


def on_resume_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml):
    """Resume selection at the next unfinished phase or image series."""
    next_set_index, next_phase = _find_resume_position(selected_indices, phases)
    window.destroy()

    if next_set_index >= len(image_sets):
        choice = messagebox.askyesnocancel(
            "Selections Complete",
            "This set is already completed.\n\n"
            "Yes: reopen the last completed entry.\n"
            "No: start again from the beginning.\n"
            "Cancel: keep the selector closed."
        )
        if choice is None:
            return
        if choice:
            display_set(image_sets, image_keys, len(image_sets) - 1, selected_indices, root, phases[-1], phases, name_xml)
            return

        selected_indices.clear()
        display_set(image_sets, image_keys, 0, selected_indices, root, phases[0], phases, name_xml)
        return

    display_set(image_sets, image_keys, next_set_index, selected_indices, root, next_phase, phases, name_xml)

def on_save_clicked(selected_indices, phases, name_xml):
    """Persist the current phase selections to user XML."""
    
    store_results_multiclass(images_dict, selected_indices, name_xml, phases)


def on_next_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml):
    """Advance after saving the current multiphase selection state."""
    handle_next_phase_or_set(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml)

# clicks and sets output
def on_selection_clicked(index, window, image_sets, image_keys, selected_indices, root, set_len, phase, set_index, phases, name_xml):
    """Record a selected frame for a phase and advance the selector."""
    #  creates new dict if not at the end
    if len(selected_indices) <= set_index:
        selected_indices.append({})

    selected_indices[set_index][phase] = index
    print(f"Selected {phase} in set {set_index + 1}: {index}")
    
    handle_next_phase_or_set(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml)

def handle_next_phase_or_set(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml):
    """Move the GUI to the next phase or the next image series."""
    
    next_index = phases.index(phase) + 1
    if next_index < len(phases):
        display_set(image_sets, image_keys, set_index, selected_indices, root, phases[next_index], phases, name_xml, window=window)
    else:
        for offset in range(1, len(image_sets) + 1):
            candidate_index = (set_index + offset) % len(image_sets)
            for candidate_phase in phases:
                if candidate_phase not in selected_indices[candidate_index]:
                    window.destroy()
                    display_set(
                        image_sets, image_keys, candidate_index, selected_indices,
                        root, candidate_phase, phases, name_xml,
                    )
                    return
        messagebox.showinfo("Completed", "All selections completed.")
        print("Final selections:", selected_indices)
        display_set(
            image_sets, image_keys, set_index, selected_indices, root,
            phase, phases, name_xml, window=window,
        )

def on_blurry_clicked(window, image_sets, image_keys, selected_indices, root, set_index, phases, name_xml):
    """Record the current phase as unavailable because the image is blurry."""
    if len(selected_indices) <= set_index:
        selected_indices.append({phase_name: 'blurry' for phase_name in phases})

    window.destroy()
    if set_index + 1 < len(image_sets):
        display_set(image_sets, image_keys, set_index + 1, selected_indices, root, phases[0], phases, name_xml)
    else:
        messagebox.showinfo("Completed", "All selections completed.")
        display_set(
            image_sets, image_keys, set_index, selected_indices, root,
            phases[0], phases, name_xml,
        )

def on_skip_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml):
    """Record an explicit skipped phase and advance the selector."""
    if len(selected_indices) <= set_index:
        selected_indices.append({})

    selected_indices[set_index][phase] = 'skipped'
    print(f"Skipped {phase} in set {set_index + 1}")
    handle_next_phase_or_set(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml)

def go_back(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml):
    """Return to the preceding phase or image series without changing files."""
    if set_index > 0 or (set_index == 0 and len(selected_indices[set_index]) > 1):
        # Revert to previous phase or image set
        previous_phase_index = phases.index(phase) - 1
        if previous_phase_index >= 0:
            # Go back within the same set
            display_set(image_sets, image_keys, set_index, selected_indices, root, phases[previous_phase_index], phases, name_xml, window=window)
        else:
            window.destroy()
            # Go back to the previous set
            if set_index > 0:
                display_set(image_sets, image_keys, set_index - 1, selected_indices, root, phases[-1], phases, name_xml)




def load_ui(cell_xml):
    """Launch phase selection after loading an explicit region XML file."""
    # Call the UI function to run the name selector
    
    name_xml = run_name_selector("select_xmls")
    print(f"Selected XML: {name_xml}")
    images_dict = get_all_images(cell_xml)
    selected_indicies = load_selector(images_dict, 0)
    store_results(images_dict, selected_indicies, name_xml)

def load_ui_from_folder(phases=None, verbose=False):
    """Ask for a CellClicker XML file and launch the multiphase selector."""
    # Call the UI function to run the name selector
    phases = phases or ['prophase','earlyprometaphase', 'prometaphase', 'metaphase', 'anaphase', 'telophase']

    directory = filedialog.askdirectory(title="Select Directory with Images")
    if not directory:
        return
    return load_ui_for_project(directory, phases=phases)

def load_ui_for_project(directory, phases=None, parent=None):
    """Launch phase selection for a standard project directory.

    Reads ``images/cell_reigons.xml`` and writes the chosen user XML beneath
    ``user_selections``. ``parent`` allows the selector to be a child window.
    """
    phases = phases or ['prophase','earlyprometaphase', 'prometaphase', 'metaphase', 'anaphase', 'telophase']
    selections_folder = os.path.join(directory, "user_selections")
    os.makedirs(selections_folder, exist_ok=True)
    
    cell_xml = os.path.join(os.path.join(directory, "images"), "cell_reigons.xml")
    cell_xml = os.path.normpath(cell_xml)

    if parent is None:
        progress_root = tk.Tk()
        progress_root.withdraw()
        progress_root._selector_owns_root = True
    else:
        progress_root = parent

    name_xml = run_name_selector(selections_folder)
    images_dict = _load_images_with_progress(cell_xml, progress_root)
    revisions = raw_track_revisions(cell_xml)
    selected_indices = load_selector(images_dict, 0, phases, name_xml, parent=parent, revisions=revisions)
    store_results_multiclass(images_dict, selected_indices, name_xml, phases, revisions=revisions)
    if parent is None and getattr(progress_root, "_selector_owns_root", False):
        try:
            progress_root.destroy()
        except tk.TclError:
            pass
    return name_xml
    
def xml_to_labels(name_xml, cell_xml):
    """Convert selected phase XML and region XML into adjusted labels."""
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    
    target_class_id = 2
    modified_df = modify_class_ids(cell_df, user_df, target_class_id)
    append_modified_labels(modified_df)
    
    
def debug_xml_to_labels(name_xml, cell_xml):
    """Return selection and region dataframes for interactive inspection."""
    
    user_df = read_xml_to_dataframe(name_xml)
    
    cell_df = cell_xml_to_dataframe(cell_xml)
    return user_df, cell_df

def debug_xml_to_labels2(name_xml, cell_xml):
    """Return modified phase-label dataframe without writing output files."""

    user_df = read_xml_to_dataframe(name_xml)

    cell_df = cell_xml_to_dataframe(cell_xml)

    target_class_id = 2
    modified_df = modify_class_ids(cell_df, user_df, target_class_id)
    return modified_df
#     target_class_id = 2
#     modified_df = modify_class_ids(cell_df, user_df, target_class_id)
#     append_modified_labels(modified_df)
