"""Create or select the user XML file used to store phase annotations."""

# import os
# import xml.etree.ElementTree as ET
# import tkinter as tk
# from tkinter import filedialog

# xml_path = ""

# def create_xml_file(name):
#     # Create an XML file with the given name
#     file_name = name + ".xml"
#     with open(os.path.join(xml_path, file_name), "wb") as file:
#         # Create a root element and an ElementTree
#         root = ET.Element("Data")
#         tree = ET.ElementTree(root)
        
#         # Write the root element to the file
#         tree.write(file, encoding="utf-8", xml_declaration=True)

# def select_name():
#     global selected_xml_path
#     selected_name = name_listbox.get(tk.ACTIVE)
#     if selected_name:
#         # Set the selected XML path and close the UI
#         selected_xml_path = os.path.join(xml_path, selected_name + ".xml")
#         window.destroy()
#         window.quit()
#     else:
#         print("Please select a name.")

# def add_new_name():
#     new_name = new_name_var.get()
#     print(new_name)
#     print("add_new_name")
#     if new_name:
#         create_xml_file(new_name)
#         name_var.set("")  # Clear the selection
#         update_name_list()
#     else:
#         print("Please enter a new name.")

# def update_name_list():
#     # Get a list of all XML files in the folder
#     xml_files = [file.split(".")[0] for file in os.listdir(xml_path) if file.endswith(".xml")]
#     name_listbox.delete(0, tk.END)
#     for name in xml_files:
#         name_listbox.insert(tk.END, name)

# def run_name_selector(xml_folder_path):
#     global xml_path
#     xml_path = xml_folder_path
#     # Create the main window
#     global window
#     window = tk.Tk()
#     window.title("XML File Selector")

#     # Create variables for selected name and new name
#     global name_var, new_name_var
#     name_var = tk.StringVar()
#     new_name_var = tk.StringVar()

#     # Label for selecting an existing name
#     select_label = tk.Label(window, text="Select an existing name:")
#     select_label.pack()

#     # Listbox to display existing names
#     global name_listbox
#     name_listbox = tk.Listbox(window)
#     name_listbox.pack()

#     # Button to select an existing name
#     select_button = tk.Button(window, text="Select", command=select_name)
#     select_button.pack()

#     # Entry widget for adding a new name
#     new_name_entry = tk.Entry(window, textvariable=new_name_var)
#     new_name_entry.pack()

#     # Button to add a new name
#     add_button = tk.Button(window, text="Add New Name", command=add_new_name)
#     add_button.pack()

#     # Update the list of names
#     update_name_list()

#     # Start the tkinter main loop
#     window.mainloop()
    
#     return selected_xml_path

import os
import xml.etree.ElementTree as ET
import tkinter as tk

class NameSelector:
    """Tk dialog that selects or creates the XML file for one annotator."""
    def __init__(self, xml_folder_path):
        self.xml_path = xml_folder_path
        self.selected_xml_path = None  # Initialize with None
        self.window = tk.Tk()
        self.window.title("XML File Selector")

        self.setup_ui()

    def setup_ui(self):
        select_label = tk.Label(self.window, text="Select an existing name:")
        select_label.pack()

        self.name_listbox = tk.Listbox(self.window)
        self.name_listbox.pack()

        select_button = tk.Button(self.window, text="Select", command=self.select_name)
        select_button.pack()

        new_name_entry = tk.Entry(self.window, textvariable=tk.StringVar())
        new_name_entry.pack()

        add_button = tk.Button(self.window, text="Add New Name", command=lambda: self.add_new_name(new_name_entry.get()))
        add_button.pack()

        self.update_name_list()

    def create_xml_file(self, name):
        file_name = f"{name}.xml"
        with open(os.path.join(self.xml_path, file_name), "wb") as file:
            root = ET.Element("Data")
            tree = ET.ElementTree(root)
            tree.write(file, encoding="utf-8", xml_declaration=True)

    def select_name(self):
        selected_name = self.name_listbox.get(tk.ACTIVE)
        if selected_name:
            self.selected_xml_path = os.path.join(self.xml_path, f"{selected_name}.xml")
            self.window.quit()  # This now only quits the main loop

    def add_new_name(self, new_name):
        print(new_name)
        if new_name:
            self.create_xml_file(new_name)
            self.update_name_list()
        else:
            print("Please enter a new name.")

    def update_name_list(self):
        xml_files = [file.split(".")[0] for file in os.listdir(self.xml_path) if file.endswith(".xml")]
        self.name_listbox.delete(0, tk.END)
        for name in xml_files:
            self.name_listbox.insert(tk.END, name)

    def run(self):
        self.window.mainloop()
        self.window.destroy()  # Ensure the window is destroyed after exiting mainloop
        return self.selected_xml_path  # Return the path after the loop is closed




def run_name_selector(xml_folder_path):
    """Open the name-selection dialog and return the selected XML path."""
    app = NameSelector(xml_folder_path)
    selected_xml_file = app.run()
    print("Selected XML File:", selected_xml_file)
    return selected_xml_file
