import importlib
import runpy
import sys

import pytest


def test_benchmark_commands_resolve_package_relative_imports():
    comparison = importlib.import_module("benchmarks.compare_single_field_sahi")

    assert callable(comparison.run)


def test_preparation_tool_uses_shared_benchmark_preprocessing():
    from benchmarks.core import preprocess_image
    from tools.prepare_cellclicker_timelapse import preprocess_image as tool_preprocess_image

    assert tool_preprocess_image is preprocess_image


def test_benchmark_runner_is_invocable_as_a_module(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmarks.run_benchmark", "--help"])

    with pytest.raises(SystemExit) as exit_status:
        runpy.run_module("benchmarks.run_benchmark", run_name="__main__")

    assert exit_status.value.code == 0
