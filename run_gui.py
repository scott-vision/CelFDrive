"""Launch the unified CellClicker project workflow GUI."""

import tkinter as tk

from CellClicker.project_gui import ProjectGUI


def main():
    """Create the project GUI and prompt immediately for a project directory."""
    root = tk.Tk()
    app = ProjectGUI(root)
    app.load_project()
    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
