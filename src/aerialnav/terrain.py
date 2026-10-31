# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Height-field obstacle sampler adapted from upstream code for Isaac Lab 3.0."""

from __future__ import annotations

import numpy as np
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils import configclass


def obstacle_height_field(difficulty, cfg, rng=None):
    """Preserve the original sampling order, grid, overlap rule and final rounding."""
    rng = np.random if rng is None else rng
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    widths = np.arange(width_min, width_max, 4)
    lengths = np.arange(width_min, width_max, 4)
    x_positions = np.arange(0, width_pixels, 4)
    y_positions = np.arange(0, length_pixels, 4)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    heights = np.zeros((width_pixels, length_pixels))
    history = []
    probability_length = len(cfg.obstacle_height_probability)

    def good_distance(x, y, width, length):
        for xp, yp, _wp, _lp in history:
            dx = abs(xp - x) - width
            dy = abs(yp - y) - length
            if dx < 0 and dy < 0:
                continue
            distance = np.sqrt(dx**2 + dy**2)
            if distance >= 2 and distance <= 10:
                return False
        return True

    for _ in range(cfg.num_obstacles):
        random_roll = rng.choice(probability_length, 1, p=cfg.obstacle_height_probability)
        index = int(random_roll[0])
        height = rng.uniform(
            cfg.obstacle_height_range[index] / cfg.vertical_scale,
            cfg.obstacle_height_range[index + 1] / cfg.vertical_scale,
        )
        attempts = 0
        while attempts < 100000:
            width = int(rng.choice(widths))
            length = int(rng.choice(lengths))
            x = int(rng.choice(x_positions))
            y = int(rng.choice(y_positions))
            if x + width > width_pixels:
                x = width_pixels - width
            if y + length > length_pixels:
                y = length_pixels - length
            if good_distance(x, y, width, length) or not history:
                break
            attempts += 1
        history.append((x, y, width, length))
        heights[x : x + width, y : y + length] = height
    x1, x2 = (width_pixels - platform_width) // 2, (width_pixels + platform_width) // 2
    y1, y2 = (length_pixels - platform_width) // 2, (length_pixels + platform_width) // 2
    heights[x1:x2, y1:y2] = 0
    return np.rint(heights).astype(np.int16)


def discrete_obstacles_height_field(difficulty, cfg):
    # Orbit seeded MT19937, then sampled the sole sub-terrain and its difficulty.
    # Lab 3.0 instead uses default_rng outside this function; replay the two old
    # draws locally so a terrain seed still denotes exactly the same static map.
    rng = np.random.RandomState(cfg.seed)
    rng.choice(1, p=[1.0])
    rng.uniform(0.0, 1.0)
    heights = obstacle_height_field(difficulty, cfg, rng)
    # Isaac Lab 3.0 asks the sampler for its unpadded origin as well as heights.
    x1, x2 = int((cfg.size[0] / 2 - 1) / cfg.horizontal_scale), int((cfg.size[0] / 2 + 1) / cfg.horizontal_scale)
    y1, y2 = int((cfg.size[1] / 2 - 1) / cfg.horizontal_scale), int((cfg.size[1] / 2 + 1) / cfg.horizontal_scale)
    origin = np.array(
        [
            (cfg.size[0] - cfg.horizontal_scale) / 2,
            (cfg.size[1] - cfg.horizontal_scale) / 2,
            np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale,
        ]
    )
    return heights, origin


def discrete_obstacles_terrain(difficulty, cfg):
    """Load the official mesh converter only when the scene generates terrain."""
    from isaaclab.terrains.height_field.utils import height_field_to_mesh

    return height_field_to_mesh(discrete_obstacles_height_field)(difficulty, cfg)


@configclass
class NavigationObstaclesTerrainCfg(HfTerrainBaseCfg):
    function = "aerialnav.terrain:discrete_obstacles_terrain"
    convert_to_heightfield = False
    seed = 0
    obstacle_width_range = (0.4, 1.1)
    obstacle_height_range = [1.0, 1.5, 2.0, 4.0, 6.0]
    obstacle_height_probability = [0.1, 0.15, 0.20, 0.55]
    num_obstacles = 350
    platform_width = 0.0


def build_terrain_cfg(num_envs, num_obstacles, seed):
    return TerrainImporterCfg(
        num_envs=num_envs,
        env_spacing=0.0,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=seed,
            size=(40.0, 40.0),
            border_width=5.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.1,
            slope_threshold=0.75,
            use_cache=False,
            color_scheme="height",
            sub_terrains={
                "obstacles": NavigationObstaclesTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=num_obstacles,
                )
            },
        ),
        visual_material=None,
        max_init_terrain_level=None,
        collision_group=-1,
        debug_vis=False,
    )
