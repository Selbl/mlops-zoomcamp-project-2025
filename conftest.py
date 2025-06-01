"""
Make sure our project package `code/` shadows the std-lib `code` module
during pytest collection.
"""

import sys
from pathlib import Path

# 1) Drop std-lib `code` if some plugin imported it already
sys.modules.pop("code", None)

# 2) Prepend ./code to sys.path so it wins the next import
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "code"))
