"""Optional pinned upstream checkout for research-only parity checks."""

import os
from pathlib import Path

UPSTREAM_NAVIGATION = Path(os.environ.get("UPSTREAM_NAVIGATION_SOURCE", str(Path(__file__).resolve().parents[1] / ".upstream-reference-absent")))
