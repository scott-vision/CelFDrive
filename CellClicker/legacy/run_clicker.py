"""Launch the legacy CellClicker bounding-box annotation GUI."""

from CellClicker.cell_clicker import ImageViewer
import tkinter as tk
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
