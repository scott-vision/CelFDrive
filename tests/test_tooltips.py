import pytest

from CellClicker.tooltips import Tooltip, add_tooltip


class FakeWidget:
    def __init__(self):
        self.bindings = {}
        self.cancelled = []
        self.scheduled = []

    def bind(self, event, callback, add=None):
        self.bindings[event] = (callback, add)

    def after(self, delay, callback):
        self.scheduled.append((delay, callback))
        return "after-1"

    def after_cancel(self, after_id):
        self.cancelled.append(after_id)


def test_add_tooltip_binds_hover_cleanup_and_returns_widget():
    widget = FakeWidget()

    returned = add_tooltip(widget, "Helpful text")

    assert returned is widget
    assert isinstance(widget._cellclicker_tooltip, Tooltip)
    assert set(widget.bindings) == {"<Enter>", "<Leave>", "<ButtonPress>", "<Destroy>"}
    assert all(add == "+" for _, add in widget.bindings.values())


def test_tooltip_schedules_and_cancels_delayed_display():
    widget = FakeWidget()
    tooltip = Tooltip(widget, "Helpful text", delay_ms=250)

    tooltip._schedule()
    tooltip._hide()

    assert widget.scheduled[0][0] == 250
    assert widget.cancelled == ["after-1"]


@pytest.mark.parametrize("text", ["", "   "])
def test_tooltip_rejects_empty_text(text):
    with pytest.raises(ValueError, match="must not be empty"):
        Tooltip(FakeWidget(), text)


def test_tooltip_rejects_negative_delay():
    with pytest.raises(ValueError, match="must not be negative"):
        Tooltip(FakeWidget(), "Helpful text", delay_ms=-1)
