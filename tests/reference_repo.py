"""External NavRL source used only by parity checks."""
import os
from pathlib import Path

NAVRL = Path(os.environ.get("NAVRL_REFERENCE_ROOT", Path(__file__).resolve().parents[2] / "NavRL"))
