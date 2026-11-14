#!/usr/bin/env python
"""Entry point: rebuild the final submissions end to end (see src/umud/run.py and README.md "Reproduce").

    python scripts/umud_run.py --config configs/final_A.yaml --stage build
    python scripts/umud_run.py --config configs/final_B.yaml --stage build

Same as the `umud-run` console script installed by `uv sync` ([project.scripts] in pyproject.toml).
GPL-3.0-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from umud.run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
