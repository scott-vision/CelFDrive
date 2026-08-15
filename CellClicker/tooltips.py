"""Small, reusable hover tooltips for the Tk interfaces."""

import tkinter as tk


class Tooltip:
    """Display explanatory text after the pointer rests over a widget."""

    def __init__(self, widget, text, delay_ms=500, wraplength=360):
        if not text or not text.strip():
            raise ValueError("Tooltip text must not be empty.")
        if delay_ms < 0:
            raise ValueError("Tooltip delay must not be negative.")

        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._after_id = None
        self._window = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._destroy, add="+")

    def _schedule(self, _event=None):
        self._cancel_scheduled()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_scheduled(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._window is not None or not self.widget.winfo_exists():
            return

        self._window = tk.Toplevel(self.widget)
        self._window.wm_overrideredirect(True)
        self._window.wm_geometry(
            f"+{self.widget.winfo_pointerx() + 14}+{self.widget.winfo_pointery() + 12}"
        )
        tk.Label(
            self._window,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            background="#ffffe0",
            foreground="#111111",
            padx=6,
            pady=4,
            wraplength=self.wraplength,
        ).pack()

    def _hide(self, _event=None):
        self._cancel_scheduled()
        if self._window is not None:
            self._window.destroy()
            self._window = None

    def _destroy(self, _event=None):
        self._cancel_scheduled()
        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
        self._window = None


def add_tooltip(widget, text, **options):
    """Attach a tooltip and return ``widget`` for concise UI construction."""
    widget._cellclicker_tooltip = Tooltip(widget, text, **options)
    return widget
