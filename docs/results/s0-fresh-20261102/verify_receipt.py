"""Verify published fresh-S0 training receipt with Python standard library."""

import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
receipt = json.loads((root / "receipt.json").read_text())
with (root / "training-windows.csv").open(newline="") as f:
    windows = list(csv.DictReader(f))
with (root / "update-timing.csv").open(newline="") as f:
    updates = list(csv.DictReader(f))
assert receipt["schema"] == "aerialnav_s0_training_evidence_v1"
assert len(updates) == 201
assert len(windows) == receipt["training_result"]["training_window_count"] == 55
assert [int(r["update"]) for r in updates] == list(range(1, 202))
assert int(windows[-1]["update"]) == 201
assert all(float(r["success_rate"]) < 0.8 for r in windows[:-1])
assert abs(float(windows[-1]["success_rate"]) - 978 / 1203) < 1e-10
assert sum(int(windows[-1][k]) for k in ("success", "collision", "bounds", "timeout", "invalid")) == 1203
assert [int(windows[-1][k]) for k in ("success", "collision", "bounds", "timeout", "invalid")] == [978, 125, 19, 81, 0]
assert int(updates[-1]["environment_steps"]) == 201 * 1024 * 96
assert abs(float(updates[-1]["trainer_elapsed_s"]) - receipt["training_result"]["trainer_elapsed_s"]) < 1e-5
assert abs(float(updates[-1]["process_elapsed_s"]) - receipt["training_result"]["process_elapsed_s"]) < 1e-5
assert receipt["training_result"]["stage_passed"]
assert not receipt["training_result"]["five_stage_acceptance_passed"]
print("Fresh S0 evidence verified: U201, 978/1203, 19,759,104 transitions, 28m26s process elapsed")
