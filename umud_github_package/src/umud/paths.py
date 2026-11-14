"""Portable paths for the preserved experiment inputs (never a temporary directory)."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = Path(os.environ.get("UMUD_RESEARCH_ROOT", PROJECT_ROOT / "artifacts/legacy_research"))
