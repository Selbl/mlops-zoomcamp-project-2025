# conftest.py  (project root)
import os
import sys
from pathlib import Path

# ──────── Prefect: start a per-process ephemeral server ──────────────
os.environ["PREFECT_API_MODE"] = "ephemeral"  # spin up local server
os.environ.pop("PREFECT_API_URL", None)  # ensure no URL is preset
# ---------------------------------------------------------------------

# make ./code shadow the std-lib “code” module
sys.modules.pop("code", None)
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
