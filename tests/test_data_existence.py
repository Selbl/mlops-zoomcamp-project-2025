"""
Smoke-test to ensure critical CSV inputs are present.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────
# Resolve <project_root>/data  no matter where pytest is run
# __file__ → tests/test_data_files.py
# parent (tests/)  → parent (project root)  → /data
# ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_students_performance_exists():
    csv_path = DATA_DIR / "students_performance.csv"
    assert csv_path.is_file(), f"Missing {csv_path}"


def test_sample_input_exists():
    csv_path = DATA_DIR / "sample_test_input.csv"
    assert csv_path.is_file(), f"Missing {csv_path}"
