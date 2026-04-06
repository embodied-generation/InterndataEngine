"""Pytest configuration for local unit tests."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"

if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))
