import importlib.util
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "examples" / "run_sample_workflow.py"


def load_sample_workflow_module():
    specification = importlib.util.spec_from_file_location("sample_workflow", SCRIPT_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_sample_fixture_matches_expected_empty_output():
    sample_workflow = load_sample_workflow_module()
    expected_path = REPOSITORY_ROOT / "sample_data" / "expected_detections.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    assert sample_workflow.matches_expected([], expected)
    assert not sample_workflow.matches_expected([[0, 1, 2, 3, 4, 0.5]], expected)
