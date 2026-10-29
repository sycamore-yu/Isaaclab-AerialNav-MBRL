"""Optional loss-gradient measurements on an existing Zhang rollout graph.

This module never writes parameter gradients, changes loss weights, or steps an
optimizer. Only detached scalars and matrices leave ``finish``. A selected update
pays for seven extra backward traversals; the default training path does not.
"""

import math
import time

import torch

LOSS_COEFFICIENTS = (
    ("velocity", "coef_v"),
    ("velocity_prediction", "coef_v_pred"),
    ("obstacle_avoidance", "coef_obj_avoidance"),
    ("collision", "coef_collide"),
    ("acceleration", "coef_d_acc"),
    ("jerk", "coef_d_jerk"),
    ("height_boundary", "coef_height_boundary"),
)


def _cosine(dot, left_norm, right_norm):
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _select_columns(values, spans):
    if len(spans) == 1:
        start, end = spans[0]
        return values[..., start:end]
    return torch.cat([values[..., start:end] for start, end in spans], dim=-1)


class GradientDiagnostic:
    """Keep detached component gradients until the original optimizer step ends."""

    def __init__(self, model, losses, cfg):
        named_parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        self._parameters = tuple(value for _, value in named_parameters)
        self._device = self._parameters[0].device
        self.diagnostic_s = 0.0
        started = self._start_timing()
        self.term_names = [name for name, _ in LOSS_COEFFICIENTS]
        self.coefficients = [float(getattr(cfg, attribute)) for _, attribute in LOSS_COEFFICIENTS]
        self.loss_values = [float(losses[name].detach()) for name in self.term_names]
        self.parameter_layout = []
        trunk, z_head = [], []
        offset = 0
        for name, parameter in named_parameters:
            end = offset + parameter.numel()
            self.parameter_layout.append({
                "name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
                "dtype": str(parameter.dtype), "flat_offset": offset,
            })
            if name.startswith(("stem.", "v_proj.", "gru.")):
                trunk.append((offset, end))
            if name == "fc.weight":
                if parameter.ndim != 2 or parameter.shape[0] != 6:
                    raise ValueError("Zhang gradient diagnostics require the interleaved six-row fc head")
                z_head.append((offset + 4 * parameter.shape[1], end))
            offset = end
        if not trunk or not z_head:
            raise ValueError("Zhang gradient diagnostics require the policy trunk and fc z rows")
        self._groups = {"all": [(0, offset)], "trunk": trunk, "z_head": z_head}
        vectors = []
        for name in self.term_names:
            gradients = torch.autograd.grad(
                losses[name], self._parameters, retain_graph=True, create_graph=False, allow_unused=True
            )
            vectors.append(self._flatten(gradients))
        self._gradients = torch.stack(vectors).detach()
        self._before = self._flatten(self._parameters)
        self._sum_check = None
        self._finish_timing(started)

    def component_gradient(self, name):
        """Return an independent detached flat component in parameter-layout order."""
        if name not in self.term_names:
            raise ValueError(f"Unknown gradient component: {name}")
        return self._gradients[self.term_names.index(name)].detach().clone()

    def _flatten(self, values):
        return torch.cat([
            (torch.zeros_like(parameter) if value is None else value.detach()).reshape(-1)
            for parameter, value in zip(self._parameters, values, strict=True)
        ])

    def _start_timing(self):
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        return time.perf_counter()

    def _finish_timing(self, started):
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self.diagnostic_s += time.perf_counter() - started

    @torch.no_grad()
    def check_total_gradient(self):
        """Compare with the original ``total.backward`` result before clipping."""
        started = self._start_timing()
        actual = self._flatten([parameter.grad for parameter in self._parameters]).double()
        coefficients = actual.new_tensor(self.coefficients)
        expected = coefficients @ self._gradients.double()
        difference = actual - expected
        actual_norm, expected_norm = float(actual.norm()), float(expected.norm())
        error = float(difference.norm())
        self._sum_check = {
            "actual_total_norm": actual_norm, "component_sum_norm": expected_norm,
            "l2_error": error, "max_abs_error": float(difference.abs().max()),
            "relative_l2_error": error / max(actual_norm, expected_norm, 1e-12),
        }
        self._finish_timing(started)

    @torch.no_grad()
    def finish(self, optimizer, grad_norm, max_grad_norm):
        """Measure the actual AdamW displacement and return JSON-safe data."""
        if self._sum_check is None:
            raise RuntimeError("Check the original total gradient before clipping and finishing the diagnostic")
        started = self._start_timing()
        delta = self._flatten(self._parameters) - self._before
        groups = {}
        for name, spans in self._groups.items():
            gradients = _select_columns(self._gradients, spans).double()
            before = _select_columns(self._before, spans).double()
            displacement = _select_columns(delta, spans).double()
            coefficients = gradients.new_tensor(self.coefficients)
            gram = gradients @ gradients.T
            total = coefficients @ gradients
            raw_dots = gradients @ total
            total_norm = float(total.norm())
            raw_gram = gram.cpu().tolist()
            norms = [math.sqrt(max(row[index], 0.0)) for index, row in enumerate(raw_gram)]
            weighted_norms = [abs(weight) * norm for weight, norm in zip(self.coefficients, norms, strict=True)]
            update_norm, parameter_norm = float(displacement.norm()), float(before.norm())
            linearized_delta = (gradients @ displacement).cpu().tolist()
            denominator = sum(weighted_norms)
            groups[name] = {
                "parameter_count": before.numel(), "raw_gram": raw_gram,
                "raw_grad_norm": norms, "weighted_grad_norm": weighted_norms,
                "cosine": [[_cosine(value, norms[i], norms[j]) for j, value in enumerate(row)]
                           for i, row in enumerate(raw_gram)],
                "component_sum_norm": total_norm,
                "cancellation_ratio": total_norm / denominator if denominator else None,
                "raw_loss_dot_total": raw_dots.cpu().tolist(),
                "signed_total_share": ((coefficients * raw_dots) / total_norm**2).cpu().tolist()
                                      if total_norm else [None] * len(norms),
                "parameter_norm": parameter_norm, "update_norm": update_norm,
                "relative_update_norm": update_norm / parameter_norm if parameter_norm else None,
                "linearized_delta_loss": linearized_delta,
                "cosine_with_actual_descent": [
                    _cosine(-value, norm, update_norm)
                    for value, norm in zip(linearized_delta, norms, strict=True)
                ],
            }
            height = self.term_names.index("height_boundary")
            height_square = raw_gram[height][height]
            others_dot_height = float(raw_dots[height]) - self.coefficients[height] * height_square
            groups[name]["height_weight_threshold_raw_gradient"] = (
                max(0.0, -others_dot_height / height_square) if height_square else None
            )
            counterfactuals = {}
            for label, term, weight in (
                ("height_coefficient_5", "height_boundary", 5.0),
                ("avoidance_coefficient_half", "obstacle_avoidance",
                 self.coefficients[self.term_names.index("obstacle_avoidance")] * 0.5),
            ):
                changed = coefficients.clone()
                changed[self.term_names.index(term)] = weight
                changed_total = changed @ gradients
                counterfactuals[label] = {
                    "term": term, "coefficient": weight,
                    "component_sum_norm": float(changed_total.norm()),
                    "raw_loss_dot_total": (gradients @ changed_total).cpu().tolist(),
                }
            groups[name]["raw_gradient_counterfactuals"] = counterfactuals
        norm = float(grad_norm)
        record = {
            "schema_version": 1, "term_order": self.term_names,
            "coefficients": self.coefficients, "raw_loss": self.loss_values,
            "weighted_loss": [weight * value for weight, value in zip(
                self.coefficients, self.loss_values, strict=True
            )],
            "parameter_layout": self.parameter_layout, "groups": groups,
            "total_gradient_check": self._sum_check,
            "pre_clip_grad_norm": norm, "max_grad_norm": float(max_grad_norm),
            "clip_scale": min(1.0, float(max_grad_norm) / (norm + 1e-6)),
            "optimizer": [{
                "lr": float(group["lr"]), "betas": list(group["betas"]),
                "eps": float(group["eps"]), "weight_decay": float(group["weight_decay"]),
            } for group in optimizer.param_groups],
            "linearized_delta_loss_sign": "negative_predicts_improvement_on_this_rollout",
            "counterfactual_scope": "raw_gradient_direction_only_not_an_adamw_step_or_new_rollout",
        }
        self._finish_timing(started)
        record["diagnostic_s"] = self.diagnostic_s
        return record
