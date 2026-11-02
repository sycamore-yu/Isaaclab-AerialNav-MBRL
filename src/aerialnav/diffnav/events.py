"""Isaac Lab Event Manager callbacks for DiffNav parameter randomization."""

import math

import torch


@torch.no_grad()
def randomize_diffnav_parameters(env, env_ids, body_radius=0.3, target_speed_range=None):
    """Resample reset drones without changing the Navigation world distribution.

    body_radius is measured from the collision geometry by the task. Training
    adds 0--0.2 m safety margin. Evaluation uses the physical radius and nominal
    dynamics; desired speed keeps the approved distribution.
    """
    if not hasattr(env, "diffnav_parameters"):
        env._diffnav_rng = torch.Generator(device=env.device).manual_seed(env.cfg.seed + 1729)
        shapes = {
            "max_speed": 1,
            "margin": None,
            "thr_est_error": 1,
            "pitch_ctl_delay": 1,
            "yaw_ctl_delay": 1,
            "drag_2": 2,
            "z_drag_coef": 1,
            "wind": 3,
            "dg": 3,
            "camera_pitch": None,
            "camera_fov_scale": None,
            "mass_scale": 1,
            "inertia_scale": 1,
            "thrust_scale": 1,
            "control_gain_scale": 1,
        }
        env.diffnav_parameters = {
            name: torch.zeros((env.num_envs,) if size is None else (env.num_envs, size), device=env.device)
            for name, size in shapes.items()
        }
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    count = len(env_ids)
    generator = env._diffnav_rng

    def uniform(*shape):
        return torch.rand(count, *shape, device=env.device, generator=generator)

    def normal(*shape):
        return torch.randn(count, *shape, device=env.device, generator=generator)

    parameters = env.diffnav_parameters
    low, high = (1.0, 6.0) if target_speed_range is None else target_speed_range
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
        raise ValueError("target speed range must be finite with 0 < min < max")
    parameters["max_speed"][env_ids] = low + (high - low) * uniform(1)
    parameters["z_drag_coef"][env_ids] = 1
    if env.cfg.evaluation or env.cfg.training_stage is not None:
        parameters["margin"][env_ids] = body_radius
        parameters["thr_est_error"][env_ids] = 1
        parameters["pitch_ctl_delay"][env_ids] = 12
        parameters["yaw_ctl_delay"][env_ids] = 6
        parameters["drag_2"][env_ids] = 0
        parameters["wind"][env_ids] = 0
        parameters["dg"][env_ids] = 0
        parameters["camera_pitch"][env_ids] = 0
        parameters["camera_fov_scale"][env_ids] = 1
    else:
        parameters["margin"][env_ids] = body_radius + 0.2 * uniform()
        parameters["thr_est_error"][env_ids] = 1 + 0.01 * normal(1)
        parameters["pitch_ctl_delay"][env_ids] = (12 + 1.2 * normal(1)).clamp_min(0.1)
        parameters["yaw_ctl_delay"][env_ids] = (6 + 0.6 * normal(1)).clamp_min(0.1)
        parameters["drag_2"][env_ids, 0] = 0
        parameters["drag_2"][env_ids, 1] = 0.3 + 0.15 * uniform()
        parameters["wind"][env_ids] = normal(3) * parameters["wind"].new_tensor((1, 1, 0.2))
        parameters["dg"][env_ids] = 0.2 * normal(3)
        parameters["camera_pitch"][env_ids] = normal() * (math.pi / 180)
        parameters["camera_fov_scale"][env_ids] = 0.95 + 0.1 * uniform()
    parameters["mass_scale"][env_ids] = env.mass_scale[env_ids]
    parameters["inertia_scale"][env_ids] = env.inertia_scale[env_ids]
    parameters["thrust_scale"][env_ids] = env.perturbation.thrust_scale[env_ids]
    parameters["control_gain_scale"][env_ids] = env.perturbation.control_gain_scale[env_ids]
    env.apply_diffnav_camera_randomization(env_ids)


@torch.no_grad()
def update_diffnav_disturbance(env, env_ids, interval_s=0.1):
    """Source OU force-bias update, scheduled by the Event Manager."""
    if env.cfg.evaluation or env.cfg.training_stage is not None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    noise = torch.randn(len(env_ids), 3, device=env.device, generator=env._diffnav_rng)
    env.diffnav_parameters["dg"][env_ids] = env.diffnav_parameters["dg"][env_ids] * math.sqrt(
        1 - interval_s / 4
    ) + noise * 0.2 * math.sqrt(interval_s / 4)
