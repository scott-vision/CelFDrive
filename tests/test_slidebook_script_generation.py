from pathlib import Path

import pytest

from run_config_gui import deploy_slidebook_script, render_slidebook_script
from test_config_gui import valid_config


def test_render_slidebook_script_uses_the_configured_environment_root_and_objective(monkeypatch, tmp_path):
    config = valid_config()
    config["project"]["repo_path"] = "microscope/CelFDrive"
    config["slidebook"]["python_environment"] = "SlideBookPython"
    config["slidebook"]["objective_before_target_search"] = "20x Air"
    config["slidebook"]["highres_objective"] = "63x Water"
    monkeypatch.setattr("run_config_gui.REPO_ROOT", tmp_path)

    script = render_slidebook_script(config)

    assert 'Environment = "SlideBookPython"' in script
    assert "sys.path.insert(0, r'" + (tmp_path / "microscope" / "CelFDrive").as_posix() + "')" in script
    assert "sys.path.insert(0, r'" + (tmp_path / "microscope" / "CelFDrive" / "SlideBook").as_posix() + "')" in script
    assert 'ChangeObjective(Objective = "63x Water")' in script
    assert "Python_RunHierarchicalCaptureFunction" in script
    assert script.index('ChangeObjective(Objective = "20x Air")') < script.index("Python_RunHierarchicalCaptureFunction")


def test_deploy_slidebook_script_copies_the_generated_macro(tmp_path):
    source = tmp_path / "CelFDrive.sbs"
    source.write_text("macro", encoding="utf-8")
    destination = tmp_path / "Scripts"
    destination.mkdir()

    deploy_slidebook_script(source, destination)

    assert (destination / "CelFDrive.sbs").read_text(encoding="utf-8") == "macro"


def test_deploy_slidebook_script_reports_a_missing_scripts_folder(tmp_path):
    source = tmp_path / "CelFDrive.sbs"
    source.write_text("macro", encoding="utf-8")
    destination = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="SlideBook scripts folder does not exist"):
        deploy_slidebook_script(source, destination)
