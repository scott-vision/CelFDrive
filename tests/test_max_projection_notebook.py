"""Regression checks for the tracked max-projection tutorial notebook."""

import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "max_projection_sahi"
    / "CelFDrive_max_projection_SAHI_example.ipynb"
)


def _notebook_source():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_configures_the_context_local_prediction_runtime():
    source = _notebook_source()

    compile(source, str(NOTEBOOK_PATH), "exec")
    assert "predict.configure_prediction_runtime(config)" in source
    assert "experiment_path = predict.get_runtime().experiment_path" in source
    assert "predict.config =" not in source
    assert "predict.model =" not in source
    assert "predict.experiment_path =" not in source
