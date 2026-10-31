"""Height-priority raw gradients and fixed-input Adam(W) counterfactual probes.

Projection constrains a raw gradient, not Adam's preconditioned/momentum update.
The probe measures that distinction using two copies of the same old optimizer
state. It never steps the real optimizer or advances a simulator/trajectory.
"""

import copy
import math
import time

import torch

from .model import decode_action


@torch.no_grad()
def height_priority_gradient(total_flat, height_flat, height_weight):
    """Remove only the other objectives' component opposing the height gradient.

    Inputs use the same trainable-parameter order and are never modified. The
    result is detached and keeps the total gradient's dtype/device. Normalizing
    the height direction in float64 before projection avoids dividing a tiny
    exponential gradient's dot product by its possibly underflowing square.
    """
    if (
        total_flat.ndim != 1
        or total_flat.numel() == 0
        or height_flat.shape != total_flat.shape
        or total_flat.device != height_flat.device
        or total_flat.dtype != height_flat.dtype
        or not total_flat.is_floating_point()
    ):
        raise ValueError("total_flat and height_flat must be nonempty matching floating vectors")
    if not math.isfinite(height_weight) or height_weight < 0:
        raise ValueError("height_weight must be finite and nonnegative")
    if not bool(torch.isfinite(total_flat).all() & torch.isfinite(height_flat).all()):
        raise ValueError("Gradient projection requires finite input gradients")
    total, height = total_flat.detach().double(), height_flat.detach().double()
    other = total - float(height_weight) * height
    scale = height.abs().max()
    zero_height = bool(scale == 0)
    applied = False
    removed_norm = 0.0
    if zero_height:
        height_norm = 0.0
        projected = total_flat.detach().clone()
    else:
        unit = height / scale
        unit_norm = unit.norm()
        unit = unit / unit_norm
        height_norm = float(scale * unit_norm)
        other_along_height = torch.dot(other, unit)
        applied = bool(other_along_height < 0)
        if applied:
            # Keep multiplication in float64: the equivalent dot/||h||^2
            # coefficient can overflow float32 even when its vector is finite.
            projected = (other - other_along_height * unit + float(height_weight) * height).to(total_flat.dtype)
            removed_norm = -float(other_along_height)
        else:
            projected = total_flat.detach().clone()
    after = projected.double()
    before_norm, after_norm = float(total.norm()), float(after.norm())
    cosine = float(torch.dot(total / before_norm, after / after_norm)) if before_norm and after_norm else None
    metrics = {
        "applied": applied,
        "height_gradient_zero": zero_height,
        "height_weight": float(height_weight),
        "height_gradient_norm": height_norm,
        "other_dot_height_before": float(torch.dot(other, height)),
        "other_dot_height_after": float(torch.dot(after - float(height_weight) * height, height)),
        "height_dot_total_before": float(torch.dot(total, height)),
        "height_dot_total_after": float(torch.dot(after, height)),
        "total_gradient_norm_before": before_norm,
        "total_gradient_norm_after": after_norm,
        "removed_opposing_component_norm": removed_norm,
        "gradient_change_norm": float((after - total).norm()),
        "gradient_cosine": None if cosine is None else max(-1.0, min(1.0, cosine)),
        "scope": "raw_gradient_only_not_an_adam_update_guarantee",
    }
    return projected.detach(), metrics


def _named_parameters(model):
    values = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not values or len({(parameter.device, parameter.dtype) for _, parameter in values}) != 1:
        raise ValueError("ActionUpdateProbe expects trainable parameters of one device and dtype")
    return values


def _flat_parameters(named):
    return torch.cat([parameter.detach().reshape(-1) for _, parameter in named])


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _command_delta(before, after):
    delta = after.double() - before.double()
    return {
        "xyz_mean": delta.mean(dim=0).cpu().tolist(),
        "xy_rms": float(delta[:, :2].square().sum(dim=-1).mean().sqrt()),
        "z_rms": float(delta[:, 2].square().mean().sqrt()),
        "xyz_rms": float(delta.square().sum(dim=-1).mean().sqrt()),
        "max_abs": float(delta.abs().max()),
    }


class ActionUpdateProbe:
    """Compare complete optimizer steps on at most 16 immutable policy inputs.

    Construct after the real raw ``backward`` and before modifying/clipping its
    gradients. ``total_flat`` and ``height_flat`` use trainable named-parameter
    order. ``probe`` contains depth/features/hidden/heading/thrust_error from
    before the selected policy call, including already-applied normalization.
    These inputs are held fixed; action delay and the simulator are not run.
    """

    @torch.no_grad()
    def __init__(self, model, optimizer, *, total_flat, height_flat, height_weight, max_grad_norm, probe):
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            raise ValueError("ActionUpdateProbe supports the existing Adam/AdamW optimizers")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        named = _named_parameters(model)
        self._layout = [(name, tuple(parameter.shape)) for name, parameter in named]
        self._device = named[0][1].device
        self._dtype = named[0][1].dtype
        self._cuda_devices = (
            [self._device.index if self._device.index is not None else torch.cuda.current_device()]
            if self._device.type == "cuda"
            else []
        )
        _synchronize(self._device)
        started = time.perf_counter()
        self._before = _flat_parameters(named)
        if (
            total_flat.shape != self._before.shape
            or total_flat.device != self._device
            or total_flat.dtype != self._dtype
        ):
            raise ValueError("Probe gradients must match the model's flat trainable-parameter layout")
        projected, projection_metrics = height_priority_gradient(total_flat, height_flat, height_weight)
        self._height = height_flat.detach().clone()
        self._total = total_flat.detach().clone()
        self._probe = self._snapshot_probe(probe, model.recurrent_state_size)
        active_gradients = [parameter.grad is not None for _, parameter in named]
        optimized = [
            parameter for group in optimizer.param_groups for parameter in group["params"] if parameter.requires_grad
        ]
        if len(optimized) != len(named) or {id(parameter) for parameter in optimized} != {
            id(parameter) for _, parameter in named
        }:
            raise ValueError("Probe optimizer must own exactly the model's trainable parameters")
        self._after_parameters = {}
        self._after_commands = {}
        self._branches = {}
        self._before_command = None
        # No manual seeding: fork_rng only restores the formal stream after the
        # isolated copies. Eval-mode DiffNavPolicy has no stochastic operations.
        with torch.random.fork_rng(devices=self._cuda_devices):
            for label, flat in (("original", self._total), ("height_priority", projected)):
                copied_model, copied_optimizer = copy.deepcopy((model, optimizer))
                copied_model.eval()
                copied_named = _named_parameters(copied_model)
                if self._before_command is None:
                    self._before_command = self._command(copied_model)
                copied_optimizer.zero_grad(set_to_none=True)
                offset = 0
                for (_, parameter), active in zip(copied_named, active_gradients, strict=True):
                    end = offset + parameter.numel()
                    value = flat[offset:end].view_as(parameter)
                    if active:
                        parameter.grad = value.clone()
                    elif bool(value.ne(0).any()):
                        raise ValueError("A nonzero probe gradient cannot belong to a parameter with grad=None")
                    offset = end
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in copied_named],
                    max_grad_norm,
                    error_if_nonfinite=True,
                )
                copied_optimizer.step()
                after = _flat_parameters(copied_named)
                command = self._command(copied_model)
                delta = after.double() - self._before.double()
                norm = float(grad_norm)
                parameter_norm = float(self._before.double().norm())
                self._after_parameters[label] = after
                self._after_commands[label] = command
                self._branches[label] = {
                    "pre_clip_grad_norm": norm,
                    "clip_scale": min(1.0, float(max_grad_norm) / (norm + 1e-6)),
                    "parameter_update_norm": float(delta.norm()),
                    "parameter_update_ratio": float(delta.norm()) / parameter_norm if parameter_norm else None,
                    "height_gradient_dot_parameter_delta": float(torch.dot(self._height.double(), delta)),
                    "original_gradient_dot_parameter_delta": float(torch.dot(self._total.double(), delta)),
                    "command_after_xyz_mean": command.double().mean(dim=0).cpu().tolist(),
                    "command_delta": _command_delta(self._before_command, command),
                }
                del copied_model, copied_optimizer
        self._record = {
            "schema_version": 1,
            "scope": "one_complete_optimizer_step_on_fixed_policy_inputs_not_a_simulated_trajectory",
            "command_semantics": "decoded_world_acceleration_before_action_delay",
            "xy_rms_definition": "sqrt(mean(delta_x_squared_plus_delta_y_squared))",
            "num_probes": self._probe["depth"].shape[0],
            "max_grad_norm": float(max_grad_norm),
            "optimizer": {
                "type": type(optimizer).__name__,
                "groups": [
                    {
                        "lr": float(group["lr"]),
                        "betas": list(group["betas"]),
                        "eps": float(group["eps"]),
                        "weight_decay": float(group["weight_decay"]),
                        "amsgrad": bool(group.get("amsgrad", False)),
                        "maximize": bool(group.get("maximize", False)),
                        "foreach": group.get("foreach"),
                        "fused": group.get("fused"),
                    }
                    for group in optimizer.param_groups
                ],
            },
            "projection": projection_metrics,
            "command_before_xyz_mean": self._before_command.double().mean(dim=0).cpu().tolist(),
            "branches": self._branches,
            "height_priority_minus_original": {
                "parameter_difference_norm": float(
                    (
                        self._after_parameters["height_priority"].double() - self._after_parameters["original"].double()
                    ).norm()
                ),
                "command_delta": _command_delta(
                    self._after_commands["original"], self._after_commands["height_priority"]
                ),
            },
        }
        _synchronize(self._device)
        self.diagnostic_s = time.perf_counter() - started

    def _snapshot_probe(self, probe, recurrent_state_size):
        required = ("depth", "features", "hidden", "heading", "thrust_error")
        if any(name not in probe for name in required):
            raise ValueError(f"Action probes require {required}")
        if any(probe[name] is None for name in required if name != "hidden"):
            raise ValueError("Only the probe's initial hidden state may be None")
        result = {name: None if probe[name] is None else probe[name].detach().clone() for name in required}
        if result["depth"].ndim != 4 or result["depth"].shape[1:] != (1, 12, 16):
            raise ValueError("Action probe depth must contain pooled [N, 1, 12, 16] features")
        count = result["depth"].shape[0]
        if not 1 <= count <= 16:
            raise ValueError("ActionUpdateProbe requires 1 to 16 fixed observations")
        if (
            result["features"].shape != (count, 10)
            or result["heading"].shape != (count, 3, 3)
            or result["thrust_error"].shape != (count, 1)
            or (result["hidden"] is not None and result["hidden"].shape != (count, recurrent_state_size))
        ):
            raise ValueError("Action probe state/heading/thrust/hidden shapes must match DiffNavPolicy")
        if any(
            value is not None and (value.device != self._device or value.dtype != self._dtype)
            for value in result.values()
        ):
            raise ValueError("Action probe tensors must match the policy device and dtype")
        return result

    def _command(self, model):
        output, _ = model(self._probe["depth"], self._probe["features"], self._probe["hidden"])
        command, _ = decode_action(output, self._probe["heading"], self._probe["thrust_error"])
        return command.detach().clone()

    @torch.no_grad()
    def finish(self, actual_model, *, applied_projection="none"):
        """Compare the real completed update with its matching counterfactual."""
        if applied_projection not in ("none", "height_priority"):
            raise ValueError("applied_projection must be none or height_priority")
        named = _named_parameters(actual_model)
        if [(name, tuple(parameter.shape)) for name, parameter in named] != self._layout:
            raise ValueError("Actual policy layout differs from the probe snapshot")
        _synchronize(self._device)
        started = time.perf_counter()
        actual = _flat_parameters(named)
        label = "original" if applied_projection == "none" else "height_priority"
        expected = self._after_parameters[label]
        difference = actual.double() - expected.double()
        delta = actual.double() - self._before.double()
        with torch.random.fork_rng(devices=self._cuda_devices):
            # Even diagnostic forward evaluation runs on a copy: no module
            # buffers, training flags or registered state on the real actor change.
            copied_model = copy.deepcopy(actual_model)
            copied_model.eval()
            actual_command = self._command(copied_model)
        self._record["actual_step"] = {
            "applied_projection": applied_projection,
            "parameter_l2_error": float(difference.norm()),
            "parameter_max_abs_error": float(difference.abs().max()),
            "parameter_relative_l2_error": float(difference.norm()) / max(float(expected.double().norm()), 1e-12),
            "command_match_error": _command_delta(self._after_commands[label], actual_command),
            "command_after_xyz_mean": actual_command.double().mean(dim=0).cpu().tolist(),
            "command_delta": _command_delta(self._before_command, actual_command),
            "height_gradient_dot_parameter_delta": float(torch.dot(self._height.double(), delta)),
            "original_gradient_dot_parameter_delta": float(torch.dot(self._total.double(), delta)),
        }
        _synchronize(self._device)
        self.diagnostic_s += time.perf_counter() - started
        self._record["diagnostic_s"] = self.diagnostic_s
        return copy.deepcopy(self._record)
