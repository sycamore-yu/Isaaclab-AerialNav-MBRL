"""Source point-mass transition with PhysX state alignment.

The source CUDA adjoint omits the derivative of its airmode correction. Its
value is retained and detached here. Position/velocity identity edges decay by
grad_decay ** dt; the integration and command edges do not receive that factor.
"""

from dataclasses import dataclass

import torch

from .model import quaternion_matrix


def gradient_decay(value, factor):
    return value * factor + value.detach() * (1.0 - factor)


def align_state(prediction, measurement):
    return prediction + (measurement - prediction).detach()


@dataclass
class PointMassState:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    actuator: torch.Tensor

    def detach(self):
        return PointMassState(*(value.detach() for value in (
            self.position, self.velocity, self.acceleration, self.actuator
        )))


def _disturbance(velocity, rotation, parameters):
    relative_velocity = velocity - parameters["wind"]
    local_velocity = torch.einsum("bi,bij->bj", relative_velocity, rotation)
    axes_scale = torch.cat((torch.ones_like(parameters["z_drag_coef"]).expand(-1, 2),
                            parameters["z_drag_coef"]), dim=-1)
    quadratic = torch.einsum("bij,bj->bi", rotation, local_velocity * local_velocity.abs() * axes_scale)
    linear = torch.einsum("bij,bj->bi", rotation, local_velocity * axes_scale)
    return (parameters["dg"] - quadratic * parameters["drag_2"][:, :1]
            - linear * parameters["drag_2"][:, 1:])


def disturbance_acceleration(velocity, quaternion_xyzw, parameters):
    """World acceleration from drag, wind and force bias for both simulators."""
    return _disturbance(velocity, quaternion_matrix(quaternion_xyzw), parameters)


def point_mass_step(state, command, rotation, parameters, dt, grad_decay):
    alpha = torch.exp(-parameters["pitch_ctl_delay"] * dt)
    actuator = command * (1 - alpha) + state.actuator * alpha
    with torch.no_grad():
        gravity = command.new_tensor((0.0, 0.0, -9.80665))
        previous_thrust = state.actuator - gravity
        next_thrust = actuator - gravity
        previous_norm = previous_thrust.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cosine = (previous_thrust * next_thrust).sum(dim=-1, keepdim=True)
        cosine = cosine / (previous_norm * next_thrust.norm(dim=-1, keepdim=True)).clamp_min(1e-8)
        angular_speed = torch.acos(cosine.clamp(-1, 1)) / dt
        airmode = previous_thrust / previous_norm * angular_speed * 0.5
    acceleration = actuator + _disturbance(state.velocity, rotation, parameters) + airmode
    position = (gradient_decay(state.position, grad_decay ** dt) + state.velocity * dt
                + 0.5 * state.acceleration * dt * dt)
    velocity = gradient_decay(state.velocity, grad_decay ** dt) + 0.5 * (state.acceleration + acceleration) * dt
    return PointMassState(position, velocity, acceleration, actuator)


def policy_transition(state, command, rotation, parameters, cfg):
    for _ in range(round(cfg.policy_dt / cfg.physics_dt)):
        state = point_mass_step(state, command, rotation, parameters, cfg.physics_dt, cfg.grad_decay)
    return state


def align_transition(prediction, measured_position, measured_velocity, prior_velocity, done,
                     reset_position, reset_velocity, policy_dt):
    """Align the terminal transition first; reset graphs only for the next state."""
    aligned = PointMassState(
        align_state(prediction.position, measured_position),
        align_state(prediction.velocity, measured_velocity),
        align_state(prediction.acceleration, (measured_velocity - prior_velocity.detach()) / policy_dt),
        prediction.actuator,
    )
    mask = done[:, None]
    next_state = PointMassState(
        torch.where(mask, reset_position.detach(), aligned.position),
        torch.where(mask, reset_velocity.detach(), aligned.velocity),
        torch.where(mask, torch.zeros_like(aligned.acceleration), aligned.acceleration),
        torch.where(mask, torch.zeros_like(aligned.actuator), aligned.actuator),
    )
    return aligned, next_state

