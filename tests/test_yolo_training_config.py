from pathlib import Path
import importlib

import pytest
import yaml

from CellClicker import yolo_training
from CellClicker.yolo_training_ui import YOLOTrainingUI


def training_config(tmp_path):
    model_path = tmp_path / "models" / "base.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"weights")
    splits = {}
    for split_name in ("train", "val", "test"):
        project_path = tmp_path / "projects" / split_name
        project_path.mkdir(parents=True)
        splits[split_name] = [str(project_path)]
    return {
        "schema_version": 1,
        "splits": splits,
        "model": {"path": str(model_path)},
        "run": {"output_root": str(tmp_path / "runs"), "name": "experiment"},
        "training": {
            "epochs": 20,
            "imgsz": 640,
            "batch": 4,
            "patience": 5,
            "device": "cpu",
        },
    }


def test_training_config_round_trip_uses_paths_relative_to_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "configs" / "training.yaml"
    config_path.parent.mkdir()
    original = yolo_training.validate_training_config(training_config(tmp_path))

    yolo_training.write_training_config(config_path, original)
    serialized = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert not Path(serialized["model"]["path"]).is_absolute()

    unrelated_directory = tmp_path / "elsewhere"
    unrelated_directory.mkdir()
    monkeypatch.chdir(unrelated_directory)
    loaded = yolo_training.load_training_config(config_path)
    assert loaded == original


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda config: config.update({"schema_version": 2}), "schema_version"),
        (lambda config: config["splits"].update({"test": []}), "splits.test"),
        (lambda config: config["splits"].update({"val": config["splits"]["train"]}), "both training and validation"),
        (lambda config: config["training"].update({"epochs": 0}), "training.epochs"),
    ],
)
def test_training_config_rejects_invalid_schema_and_values(tmp_path, update, message):
    config = training_config(tmp_path)
    update(config)
    with pytest.raises(ValueError, match=message):
        yolo_training.validate_training_config(config)


def test_training_config_reports_missing_model_and_project_paths(tmp_path):
    config = training_config(tmp_path)
    Path(config["model"]["path"]).unlink()
    with pytest.raises(FileNotFoundError, match="Training model does not exist"):
        yolo_training.validate_training_config(config)

    config = training_config(tmp_path / "second")
    missing_project = Path(config["splits"]["test"][0])
    missing_project.rmdir()
    with pytest.raises(FileNotFoundError, match="split `test` project"):
        yolo_training.validate_training_config(config)


def test_gui_fields_produce_the_same_validated_configuration(tmp_path):
    expected = yolo_training.validate_training_config(training_config(tmp_path))

    class Field:
        def __init__(self, value):
            self.value = str(value)

        def get(self):
            return self.value

    ui = object.__new__(YOLOTrainingUI)
    ui.train_projects = expected["splits"]["train"]
    ui.val_projects = expected["splits"]["val"]
    ui.test_projects = expected["splits"]["test"]
    ui.model_path_var = Field(expected["model"]["path"])
    ui.output_root_var = Field(expected["run"]["output_root"])
    ui.run_name_var = Field(expected["run"]["name"])
    ui.epochs_var = Field(expected["training"]["epochs"])
    ui.imgsz_var = Field(expected["training"]["imgsz"])
    ui.batch_var = Field(expected["training"]["batch"])
    ui.patience_var = Field(expected["training"]["patience"])
    ui.device_var = Field(expected["training"]["device"])

    assert ui._validate_inputs() == expected

def test_run_training_config_maps_all_fields_and_writes_resolved_snapshot(tmp_path, monkeypatch):
    config = training_config(tmp_path)
    captured = {}

    def fake_train_yolo_model(**kwargs):
        captured.update(kwargs)
        run_dir = tmp_path / "runs" / "experiment"
        run_dir.mkdir(parents=True)
        return {"run_dir": str(run_dir)}

    monkeypatch.setattr(yolo_training, "train_yolo_model", fake_train_yolo_model)
    result = yolo_training.run_training_config(config)

    assert captured["train_dirs"] == config["splits"]["train"]
    assert captured["model_path"] == config["model"]["path"]
    assert captured["epochs"] == 20 and captured["device"] == "cpu"
    snapshot = yaml.safe_load(Path(result["training_config"]).read_text(encoding="utf-8"))
    assert snapshot["model"]["path"] == config["model"]["path"].replace("\\", "/")


def test_train_cli_is_import_safe_and_applies_name_override(monkeypatch, tmp_path):
    train_cli = importlib.import_module("train")
    config = training_config(tmp_path)
    calls = []
    monkeypatch.setattr(train_cli, "load_training_config", lambda path: config)
    monkeypatch.setattr(
        train_cli,
        "run_training_config",
        lambda loaded: calls.append(loaded) or {"run_dir": "completed-run"},
    )

    result = train_cli.main(["--config", "training.yaml", "--name", "cli-run"])

    assert result["run_dir"] == "completed-run"
    assert calls[0]["run"]["name"] == "cli-run"
    assert config["run"]["name"] == "experiment"
