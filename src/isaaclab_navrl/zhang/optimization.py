"""Small optimizer measurements and report-driven training controls.

Recovery only requests actions.  The runner performs and acknowledges them;
this module cannot load a model, reset the simulator, rewind global updates,
or change a curriculum/acceptance outcome.  All patience values count fresh
reports of naturally completed training episodes, not optimizer updates.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import torch


class OptimizerStepMonitor:
    """Measure the actual optimizer displacement with one host transfer.

    ``param_norm`` is the parameter norm *before* the step, which is also the
    denominator of ``update_ratio``.  Layers combine their weights and biases.
    The six FC rows follow [ax, vx, ay, vy, az, vz].  A monitor holds one detached
    parameter snapshot, never gradients or a rollout graph.
    """

    head_names = ("ax", "vx", "ay", "vy", "az", "vz")

    def __init__(self, model: torch.nn.Module, epsilon: float = 1e-12):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = epsilon
        self.parameters = tuple((name, value) for name, value in model.named_parameters() if value.requires_grad)
        if not self.parameters:
            raise ValueError("Cannot monitor a model with no trainable parameters")
        if len({value.device for _, value in self.parameters}) != 1:
            raise ValueError("OptimizerStepMonitor expects one model device")
        self.groups: dict[str, list[tuple[int, int]]] = {"Optim": []}
        offset = 0
        for name, parameter in self.parameters:
            end = offset + parameter.numel()
            self.groups["Optim"].append((offset, end))
            layer = name.rsplit(".", 1)[0] if "." in name else name
            self.groups.setdefault(f"Optim/layers/{layer}", []).append((offset, end))
            if name in ("fc.weight", "fc.bias"):
                if parameter.shape[0] != len(self.head_names):
                    raise ValueError("Expected the six interleaved Zhang FC outputs")
                row_width = parameter.numel() // len(self.head_names)
                for row, label in enumerate(self.head_names):
                    self.groups.setdefault(f"Optim/head/{label}", []).append(
                        (offset + row * row_width, offset + (row + 1) * row_width)
                    )
            offset = end
        self._before: torch.Tensor | None = None

    def _flat_parameters(self) -> torch.Tensor:
        return torch.cat([value.detach().reshape(-1) for _, value in self.parameters])

    @torch.no_grad()
    def before_step(self) -> None:
        if self._before is not None:
            raise RuntimeError("The preceding monitored step has not been finished")
        self._before = self._flat_parameters()

    @torch.no_grad()
    def after_step(self) -> dict[str, float]:
        if self._before is None:
            raise RuntimeError("Call before_step() before the optimizer step")
        before, after = self._before, self._flat_parameters()
        delta = after - before
        rows = []
        for ranges in self.groups.values():
            # Reduction stays on the model device.  Float64 accumulation is
            # over scalar sums, without a second full parameter/gradient copy.
            parameter_squared = sum(before[start:end].square().sum(dtype=torch.float64) for start, end in ranges)
            update_squared = sum(delta[start:end].square().sum(dtype=torch.float64) for start, end in ranges)
            rows.append(torch.stack((parameter_squared.sqrt(), update_squared.sqrt())))
        values = torch.stack(rows).cpu().tolist()  # One synchronization per update.
        self._before = None
        result = {}
        for prefix, (parameter_norm, update_norm) in zip(self.groups, values, strict=True):
            result[f"{prefix}/param_norm"] = parameter_norm
            result[f"{prefix}/update_norm"] = update_norm
            result[f"{prefix}/update_ratio"] = update_norm / (parameter_norm + self.epsilon)
        return result


def make_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str = "cosine",
    *,
    factor: float = 0.5,
    patience: int = 5,
    threshold: float = 0.005,
    cooldown: int = 2,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Return the official scheduler, or None for another exclusive schedule.

    Only call ``scheduler.step(report['success_rate'])`` for a *new* natural
    episode report.  Save/load its standard state_dict with the optimizer.
    The runner must not also assign a cosine learning rate in plateau mode.
    """
    if schedule not in ("cosine", "plateau", "constant"):
        raise ValueError("schedule must be cosine, plateau, or constant")
    if schedule != "plateau":
        return None
    if not math.isfinite(factor) or not 0 < factor < 1:
        raise ValueError("plateau factor must be finite and in (0, 1)")
    if any(type(value) is not int or value < 0 for value in (patience, cooldown)):
        raise ValueError("plateau patience and cooldown must be nonnegative integers")
    if any(not math.isfinite(value) or value < 0 for value in (threshold, min_lr)):
        raise ValueError("plateau threshold and min_lr must be finite and nonnegative")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=factor, patience=patience, threshold=threshold,
        threshold_mode="abs", cooldown=cooldown, min_lr=min_lr,
    )


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    stable_reports: int = 3
    stable_min_success: float = 0.20
    stable_max_spread: float = 0.05
    stable_improvement: float = 0.005
    degrade_reports: int = 3
    degrade_abs_drop: float = 0.05
    degrade_relative_drop: float = 0.2
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    observe_reports: int = 3
    cooldown_reports: int = 3

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("Recovery enabled must be bool")
        if any(type(value) is not int or value < 1 for value in (
            self.stable_reports, self.degrade_reports, self.observe_reports
        )):
            raise ValueError("Recovery report counts must be positive integers")
        if type(self.cooldown_reports) is not int or self.cooldown_reports < 0:
            raise ValueError("Recovery cooldown must be a nonnegative integer")
        for name in ("stable_min_success", "stable_max_spread", "stable_improvement",
                     "degrade_abs_drop", "degrade_relative_drop"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.stable_improvement <= 0 or max(self.degrade_abs_drop, self.degrade_relative_drop) <= 0:
            raise ValueError("Stable improvement and at least one degradation margin must be positive")
        if not math.isfinite(self.lr_factor) or not 0 < self.lr_factor < 1:
            raise ValueError("Recovery lr_factor must be finite and in (0, 1)")
        if not math.isfinite(self.min_lr) or self.min_lr < 0:
            raise ValueError("Recovery min_lr must be finite and nonnegative")


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    stage: str
    update: int
    reason: str
    reference_success: float | None = None
    checkpoint: str | None = None
    lr_factor: float | None = None
    min_lr: float | None = None


class RecoveryController:
    """Request stable save -> LR reduction -> observation -> optional restore.

    A saved reference is the minimum success of a stable report window, not
    one lucky report.  Degradation means a drop larger than both the configured
    absolute and relative margins.  Action acknowledgements happen only after
    the runner succeeds.  During restore, keep this *current* controller and
    all global/report/acceptance counters; do not load their older values from
    the selected model checkpoint.  The runner also owns persistent LR scaling
    so a later cosine assignment cannot undo a recovery reduction.
    """

    def __init__(self, stage: str, config: RecoveryConfig | None = None):
        self.config = config or RecoveryConfig()
        self.config.validate()
        self.last_report_update = 0
        self.restore_count = 0
        self.reset_stage(stage)

    def reset_stage(self, stage: str) -> None:
        if not isinstance(stage, str) or not stage:
            raise ValueError("A recovery controller requires an explicit training stage")
        self.stage = stage
        self.state = "warming" if self.config.enabled else "disabled"
        self.window: deque[dict] = deque(maxlen=self.config.stable_reports)
        self.stable_checkpoint: str | None = None
        self.reference_success: float | None = None
        self.bad_reports = self.observed_reports = self.observed_bad_reports = 0
        self.cooldown_remaining = 0
        self.pending: RecoveryDecision | None = None

    def _decision(self, action: str, reason: str, **kwargs) -> RecoveryDecision:
        return RecoveryDecision(action, self.stage, self.last_report_update, reason,
                                reference_success=self.reference_success, **kwargs)

    def observe(self, report: dict) -> RecoveryDecision:
        if self.pending is not None:
            raise RuntimeError("Acknowledge the pending recovery action before another report")
        if report.get("stage") != self.stage:
            raise ValueError("Recovery reports and stable checkpoints must belong to the same stage")
        update, episodes, success = report.get("update"), report.get("episodes"), report.get("success_rate")
        if type(update) is not int or update <= self.last_report_update:
            raise ValueError("Recovery requires a new report with a strictly increasing global update")
        if type(episodes) is not int or episodes < 1:
            raise ValueError("Recovery requires naturally completed episodes")
        if not isinstance(success, (int, float)) or not math.isfinite(success) or not 0 <= success <= 1:
            raise ValueError("Report success_rate must be finite and in [0, 1]")
        self.last_report_update = update
        if not self.config.enabled:
            return self._decision("none", "disabled")
        self.window.append({"success_rate": float(success), "episodes": episodes})
        if self.state == "cooldown":
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.state = "armed"
            return self._decision("none", "post_restore_cooldown")
        degraded = self.reference_success is not None and success < self.reference_success - max(
            self.config.degrade_abs_drop, self.reference_success * self.config.degrade_relative_drop
        )
        if self.state == "observing":
            self.observed_reports += 1
            self.observed_bad_reports += int(degraded)
            if self.observed_reports < self.config.observe_reports:
                return self._decision("none", "observing_after_lr_reduction")
            if self.observed_bad_reports == self.config.observe_reports:
                self.pending = self._decision("restore", "persistent_degradation", checkpoint=self.stable_checkpoint)
                self.state = "restore_requested"
                return self.pending
            self.state = "armed"
            self.bad_reports = 0
            return self._decision("none", "degradation_did_not_persist")
        self.bad_reports = self.bad_reports + 1 if degraded else 0
        if self.bad_reports >= self.config.degrade_reports:
            self.pending = self._decision("reduce_lr", "consecutive_degraded_reports",
                                          lr_factor=self.config.lr_factor, min_lr=self.config.min_lr)
            self.state = "reduce_requested"
            return self.pending
        if len(self.window) == self.config.stable_reports and not degraded:
            rates = [entry["success_rate"] for entry in self.window]
            reference = min(rates)
            stable = min(rates) >= self.config.stable_min_success and max(rates) - min(rates) <= self.config.stable_max_spread
            improved = self.reference_success is None or reference >= self.reference_success + self.config.stable_improvement
            if stable and improved:
                self.pending = RecoveryDecision("save_stable", self.stage, update, "stable_success_window", reference)
                self.state = "save_requested"
                return self.pending
        return self._decision("none", "collecting_reports")

    def acknowledge(self, action: str, checkpoint: str | None = None) -> None:
        if self.pending is None or self.pending.action != action:
            raise ValueError("Acknowledgement does not match the pending recovery action")
        if action == "save_stable":
            if not isinstance(checkpoint, str) or not checkpoint:
                raise ValueError("A saved stable checkpoint path is required")
            self.stable_checkpoint = checkpoint
            self.reference_success = self.pending.reference_success
            self.state = "armed"
            self.bad_reports = 0
        elif action == "reduce_lr":
            self.state = "observing"
            self.observed_reports = self.observed_bad_reports = self.bad_reports = 0
        elif action == "restore":
            self.restore_count += 1
            self.window.clear()
            self.bad_reports = self.observed_reports = self.observed_bad_reports = 0
            self.cooldown_remaining = self.config.cooldown_reports
            self.state = "cooldown" if self.cooldown_remaining else "armed"
        else:
            raise ValueError("Unsupported recovery action")
        self.pending = None

    def state_dict(self) -> dict:
        return {
            "version": 1, "config": asdict(self.config), "stage": self.stage, "state": self.state,
            "last_report_update": self.last_report_update, "restore_count": self.restore_count,
            "window": [dict(entry) for entry in self.window], "stable_checkpoint": self.stable_checkpoint,
            "reference_success": self.reference_success, "bad_reports": self.bad_reports,
            "observed_reports": self.observed_reports, "observed_bad_reports": self.observed_bad_reports,
            "cooldown_remaining": self.cooldown_remaining,
            "pending": asdict(self.pending) if self.pending is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or state.get("stage") != self.stage or state.get("config") != asdict(self.config):
            raise ValueError("Recovery state must have the same version, stage, and configuration")
        for name in ("state", "last_report_update", "restore_count", "stable_checkpoint", "reference_success",
                     "bad_reports", "observed_reports", "observed_bad_reports", "cooldown_remaining"):
            setattr(self, name, state[name])
        self.window = deque((dict(entry) for entry in state["window"]), maxlen=self.config.stable_reports)
        self.pending = RecoveryDecision(**state["pending"]) if state["pending"] is not None else None
