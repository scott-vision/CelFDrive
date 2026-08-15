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
from .phase_settings import DEFAULT_PHASES, load_phases, phase_signature, settings_path
from .tooltips import add_tooltip


PHASE_SELECTOR_HELP_TEXT = (
    "Each track is one cell series. For the phase named in the window title, select the first thumbnail "
    "where that phase is visible. The selector then advances to the next phase or track.\n\n"
    "Navigation: Left Arrow or Back returns to the preceding phase; Right Arrow or Next moves on without "
    "changing the current selection. Next Track moves to another track without changing it.\n\n"
    "S or Skip Phase records that the current phase is not present. Mark as Blurry marks every phase in this track "
    "as unavailable. Abstain From Track records no opinion, allowing other reviewers' selections to be used.\n\n"
    "Use Jump to Next TODO to find unfinished tracks. Save writes your selections; Close Selector exits the selector."
)


def show_selector_help(window):
    """Display guidance for the persistent phase-selector controls."""
    messagebox.showinfo("Phase Selector Help", PHASE_SELECTOR_HELP_TEXT, parent=window)


def _run_selector_hotkey(action):
    """Run a selector shortcut action and prevent Tk from processing the key again."""
    action()
    return "break"


def _finish_selector(root):
    owns_root = bool(getattr(root, "_selector_owns_root", False))
    if owns_root:
        root.quit()
        root.destroy()


def _load_selected_indices_from_xml(image_keys, name_xml, revisions=None, phases=None, selection_phase_signature=None):
    if not name_xml or not os.path.exists(name_xml):
        return []

    stored_selections = parse_xml_for_phases_resume(name_xml, phases=phases)
    entries = selection_entries_by_track(name_xml)
    selected_indices = []
    for image_key in image_keys:
        revision = (revisions or {}).get((image_key[0], str(image_key[1])), 0)
        entry = entries.get((image_key[0], str(image_key[1])))
        selected_indices.append(
            dict(stored_selections.get(image_key, {}))
            if entry_is_current(entry, revision, phase_signature=selection_phase_signature) else {}
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


def load_selector(image_dict, set_index, phases, name_xml, parent=None, revisions=None, selection_phase_signature=None):
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
    root._phase_selector_revisions = revisions or {}

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
    selected_indices = _load_selected_indices_from_xml(
        image_keys, name_xml, revisions=revisions,
        phases=phases,
        selection_phase_signature=selection_phase_signature,
    )
    next_set_index, next_phase = _find_resume_position(selected_indices, phases)
    if next_set_index >= len(image_sets):
        # Keep the ordered list available for inspection and correction even
        # when every track is currently complete.
        next_set_index, next_phase = min(set_index, max(0, len(image_sets) - 1)), phases[0]
    selector_window = display_set(
        image_sets, image_keys, next_set_index, selected_indices, root, next_phase, phases, name_xml,
    )
    if getattr(root, "_selector_owns_root", False):
        root.mainloop()
    else:
        root.wait_window(selector_window)
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

def _create_selector_layout(window):
    """Create the persistent controls used while moving through selector tracks."""
    window._selector_header = tk.StringVar()
    window._selector_selection = tk.StringVar()
    window._selector_track_number = tk.StringVar()

    tk.Label(window, textvariable=window._selector_header, anchor=tk.W, justify=tk.LEFT).grid(
        row=0, column=0, columnspan=13, sticky="ew", padx=4, pady=(0, 2)
    )
    window._selector_selection_label = tk.Label(
        window, textvariable=window._selector_selection, anchor=tk.W, justify=tk.LEFT
    )
    window._selector_selection_label.grid(row=1, column=0, columnspan=13, sticky="ew", padx=4, pady=(0, 6))

    window._selector_images = tk.Frame(window)
    window._selector_images.grid(row=2, column=0, columnspan=13, sticky="nw")

    controls = tk.Frame(window)
    controls.grid(row=3, column=0, columnspan=13, sticky="ew", padx=4, pady=(6, 4))
    window._selector_controls = {
        "back": tk.Button(controls, text="Back"),
        "next": tk.Button(controls, text="Next (No Change)"),
        "next_track": tk.Button(controls, text="Next Track (No Change)"),
        "skip": tk.Button(controls, text="Skip Phase"),
        "blurry": tk.Button(controls, text="Mark as Blurry"),
        "abstain": tk.Button(controls, text="Abstain From Track"),
        "resume": tk.Button(controls, text="Resume"),
        "todo": tk.Button(controls, text="Jump to Next TODO"),
        "go_to_track": tk.Button(controls, text="Go to Track"),
        "save": tk.Button(controls, text="Save"),
        "close": tk.Button(controls, text="Close Selector"),
        "help": tk.Button(controls, text="? Help"),
    }
    tooltip_text = {
        "next": "Move forward without modifying the saved selection. Shortcut: Right Arrow.",
        "next_track": "Move to the next cell track without changing this track.",
        "skip": "Record that this phase is not present in the current track. Shortcut: S.",
        "blurry": "Mark every phase in this track unavailable because the images cannot be judged.",
        "abstain": "Record no opinion so another reviewer can supply the selections.",
        "todo": "Find the next track with missing or stale phase selections.",
        "save": "Write the current selections to this reviewer's XML file.",
        "close": "Close the selector. Save first to preserve recent changes.",
    }
    for control_name, text in tooltip_text.items():
        add_tooltip(window._selector_controls[control_name], text)
    control_positions = (
        ("back", 0), ("next", 1), ("next_track", 2), ("skip", 3), ("blurry", 4),
        ("abstain", 5), ("resume", 6), ("todo", 7),
    )
    for name, column in control_positions:
        window._selector_controls[name].grid(row=0, column=column, sticky="ew")
    tk.Label(controls, text="Track:").grid(row=0, column=8, sticky="e", padx=(8, 2))
    track_entry = tk.Entry(controls, textvariable=window._selector_track_number, width=6)
    track_entry.grid(row=0, column=9, sticky="w")
    window._selector_controls["track_entry"] = track_entry
    window._selector_controls["go_to_track"].grid(row=0, column=10, sticky="ew", padx=(2, 0))
    window._selector_controls["save"].grid(row=0, column=11, sticky="ew")
    window._selector_controls["close"].grid(row=0, column=12, sticky="ew", padx=(4, 0))
    window._selector_controls["help"].grid(row=0, column=13, sticky="ew", padx=(4, 0))
    for column in range(14):
        controls.grid_columnconfigure(column, weight=1)


def display_set(image_sets, image_keys, set_index, selected_indices, root, phase, phases, name_xml, window=None):
    """Update the persistent selector window for one series and phase."""
    if set_index < 0 or set_index >= len(image_sets):
        messagebox.showinfo("Completed", "All selections completed.")
        _finish_selector(root)
        return

    is_new_window = window is None or not window.winfo_exists()
    if is_new_window:
        window = tk.Toplevel(root)
        _create_selector_layout(window)
        window.minsize(1300, 0)
        window.geometry("+100+100")

    series = image_sets[set_index]
    source_image, series_id = image_keys[set_index]
    set_len = len(series)
    max_images_per_row = 13
    window.title(f"Select First frame visible for {phase.capitalize()} - Set {set_index + 1} of {len(image_sets)}")
    window._selector_header.set(
        f"Track {set_index + 1}/{len(image_sets)}  |  Series {series_id}  |  Source image: {source_image}"
    )
    window._selector_track_number.set(str(set_index + 1))

    selected_index_for_phase = None
    selected_text = "Current selected: none"
    if len(selected_indices) > set_index:
        phase_value = selected_indices[set_index].get(phase)
        if isinstance(phase_value, int):
            selected_index_for_phase = phase_value
            selected_text = f"Current selected: {phase_value}"
        elif phase_value == "skipped":
            selected_text = "Current selected: skipped"
        elif phase_value == "blurry":
            selected_text = "Current selected: blurry"
    window._selector_selection.set(selected_text)
    window._selector_selection_label.configure(
        fg="#22cc22" if selected_index_for_phase is not None else "#cccccc"
    )

    controls = window._selector_controls
    controls["back"].configure(command=lambda: go_back(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml))
    controls["next"].configure(command=lambda: on_next_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml))
    controls["next_track"].configure(command=lambda: next_track_no_change(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml))
    controls["skip"].configure(command=lambda: on_skip_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml))
    controls["blurry"].configure(command=lambda: on_blurry_clicked(window, image_sets, image_keys, selected_indices, root, set_index, phases, name_xml))
    controls["abstain"].configure(command=lambda: on_abstain_clicked(window, image_sets, image_keys, selected_indices, root, set_index, phases, name_xml))
    controls["resume"].configure(command=lambda: on_resume_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml))
    controls["todo"].configure(command=lambda: jump_to_next_todo(window, image_sets, image_keys, selected_indices, root, set_index, phases, name_xml))
    controls["go_to_track"].configure(command=lambda: go_to_track(window._selector_track_number.get(), window, image_sets, image_keys, selected_indices, root, phase, phases, name_xml))
    controls["save"].configure(command=lambda: on_save_clicked(image_sets, image_keys, selected_indices, root, phases, name_xml))
    controls["close"].configure(command=lambda: close_selector(window, root))
    controls["help"].configure(command=lambda: show_selector_help(window))
    controls["track_entry"].bind("<Return>", lambda _event: go_to_track(window._selector_track_number.get(), window, image_sets, image_keys, selected_indices, root, phase, phases, name_xml))
    window.bind(
        "<Left>",
        lambda _event: _run_selector_hotkey(
            lambda: go_back(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml)
        ),
    )
    window.bind(
        "<Right>",
        lambda _event: _run_selector_hotkey(
            lambda: on_next_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml)
        ),
    )
    window.bind(
        "<s>",
        lambda _event: _run_selector_hotkey(
            lambda: on_skip_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml)
        ),
    )
    window.bind(
        "<S>",
        lambda _event: _run_selector_hotkey(
            lambda: on_skip_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml)
        ),
    )

    for child in window._selector_images.winfo_children():
        child.destroy()
    photo_images = []
    for index, image_array in enumerate(series):
        row = index // max_images_per_row
        column = index % max_images_per_row
        image = Image.fromarray(normalize_image(image_array)).convert("RGB")
        image.thumbnail((100, 100))
        if index == selected_index_for_phase:
            draw = ImageDraw.Draw(image)
            for offset in range(3):
                draw.rectangle(
                    (2 + offset, 2 + offset, max(2 + offset, image.width - 3 - offset), max(2 + offset, image.height - 3 - offset)),
                    outline="#22cc22",
                )
        photo = ImageTk.PhotoImage(image)
        photo_images.append(photo)
        button = tk.Button(
            window._selector_images, image=photo,
            command=lambda index=index: on_selection_clicked(index, window, image_sets, image_keys, selected_indices, root, set_len, phase, set_index, phases, name_xml),
            relief=tk.FLAT, bd=0,
        )
        button.image = photo
        add_tooltip(
            button,
            "Select this as the first frame where the current phase is visible.",
            delay_ms=350,
        )
        button.grid(row=row, column=column, padx=1, pady=1)
    window._selector_photos = photo_images
    return window

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
            display_set(
                image_sets, image_keys, len(image_sets) - 1, selected_indices,
                root, phases[-1], phases, name_xml, window=window,
            )
            return

        selected_indices.clear()
        display_set(image_sets, image_keys, 0, selected_indices, root, phases[0], phases, name_xml, window=window)
        return

    display_set(
        image_sets, image_keys, next_set_index, selected_indices,
        root, next_phase, phases, name_xml, window=window,
    )

def on_save_clicked(image_sets, image_keys, selected_indices, root, phases, name_xml):
    """Persist the current phase selections to user XML."""
    images_dict = dict(zip(image_keys, image_sets))
    store_results_multiclass(
        images_dict, selected_indices, name_xml, phases,
        revisions=getattr(root, "_phase_selector_revisions", {}),
        phase_signature=phase_signature(phases),
    )


def on_next_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml):
    """Advance after saving the current multiphase selection state."""
    handle_next_phase_or_set(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml)


def next_track_no_change(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml):
    """Move to the next track without modifying the current track's phases."""
    if not image_sets:
        return
    next_index = (set_index + 1) % len(image_sets)
    next_phase = next(
        (candidate for candidate in phases if candidate not in selected_indices[next_index]),
        phases[0],
    )
    display_set(
        image_sets, image_keys, next_index, selected_indices, root,
        next_phase, phases, name_xml, window=window,
    )

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
                    display_set(
                        image_sets, image_keys, candidate_index, selected_indices,
                        root, candidate_phase, phases, name_xml, window=window,
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
    selected_indices[set_index] = {phase_name: 'blurry' for phase_name in phases}

    if set_index + 1 < len(image_sets):
        display_set(
            image_sets, image_keys, set_index + 1, selected_indices,
            root, phases[0], phases, name_xml, window=window,
        )
    else:
        messagebox.showinfo("Completed", "All selections completed.")
        display_set(
            image_sets, image_keys, set_index, selected_indices, root,
            phases[0], phases, name_xml, window=window,
        )


def on_abstain_clicked(window, image_sets, image_keys, selected_indices, root, set_index, phases, name_xml):
    """Record no opinion for every phase in the current track."""
    if not messagebox.askyesno(
        "Abstain From Track",
        "Record no opinion for every phase in this track?\n\n"
        "The other reviewers' selected frames will be aggregated unless a strict majority skips a phase.",
        parent=window,
    ):
        return
    selected_indices[set_index] = {phase_name: 'skipped' for phase_name in phases}
    handle_next_phase_or_set(
        window, image_sets, image_keys, selected_indices, root,
        phases[-1], set_index, phases, name_xml,
    )

def on_skip_clicked(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml):
    """Record an explicit skipped phase and advance the selector."""
    if len(selected_indices) <= set_index:
        selected_indices.append({})

    selected_indices[set_index][phase] = 'skipped'
    print(f"Skipped {phase} in set {set_index + 1}")
    handle_next_phase_or_set(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml)

def go_back(window, image_sets, image_keys, selected_indices, root, phase, set_index, phases, name_xml):
    """Return to the preceding phase or track, regardless of saved selections."""
    previous_phase_index = phases.index(phase) - 1
    if previous_phase_index >= 0:
        display_set(
            image_sets, image_keys, set_index, selected_indices, root,
            phases[previous_phase_index], phases, name_xml, window=window,
        )
    elif set_index > 0:
        display_set(
            image_sets, image_keys, set_index - 1, selected_indices,
            root, phases[-1], phases, name_xml, window=window,
        )




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
    phases = phases or list(DEFAULT_PHASES)

    directory = filedialog.askdirectory(title="Select Directory with Images")
    if not directory:
        return
    return load_ui_for_project(directory, phases=phases)

def load_ui_for_project(directory, phases=None, parent=None):
    """Launch phase selection for a standard project directory.

    Reads ``images/cell_reigons.xml`` and writes the chosen user XML beneath
    ``user_selections``. ``parent`` allows the selector to be a child window.
    """
    phases = phases or load_phases(directory)
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
    progress_root._phase_selector_revisions = revisions
    selected_indices = load_selector(
        images_dict, 0, phases, name_xml, parent=parent, revisions=revisions,
        selection_phase_signature=phase_signature(phases) if os.path.exists(settings_path(directory)) else None,
    )
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
