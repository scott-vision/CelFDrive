from pathlib import Path

from run_config_gui import ConfigEditor


def test_initial_config_prefers_default_yaml(monkeypatch, tmp_path):
    backup = tmp_path / "backup.yaml"
    default = tmp_path / "default.yaml"
    backup.touch()
    default.touch()
    editor = ConfigEditor.__new__(ConfigEditor)
    monkeypatch.setattr("run_config_gui.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(editor, "get_config_files", lambda: [backup, default])

    assert editor.get_initial_config_path() == default
