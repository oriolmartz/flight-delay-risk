"""Lightweight CLI smoke tests for script entry points."""
from __future__ import annotations

import subprocess
import sys


def test_train_model_help_exposes_real_data_flags():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_model", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--include-gradient-boosting" in result.stdout
    assert "--max-rows" in result.stdout


def test_run_real_data_demo_help_exposes_validation_selection_flags():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_real_data_demo", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--selection-metric" in result.stdout
    assert "--validation-size" in result.stdout
    assert "--max-rows-per-month" in result.stdout
