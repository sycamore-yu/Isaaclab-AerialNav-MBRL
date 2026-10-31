"""CUDA Graph execution of the unchanged Lee/RotorGroup controller for DiffNav."""

from __future__ import annotations

import time

import torch

from aerialnav.control import LeeRotorController


class GraphedLeeRotorController:
    """Retain original motor state and kernels while removing repeated dispatch.

    DiffNav supplies explicit target acceleration and yaw at every physics step.
    Optional arguments retain the Lee defaults before input copying. The graph
    captures the fixed tensor argument structure. Input and output storage
    stays alive for the adapter's lifetime; resets only update values in place.
    """

    @torch.no_grad()
    def __init__(self, controller: LeeRotorController, sample_inputs):
        started = time.perf_counter()
        self.controller = controller
        self.device = sample_inputs[0].device
        if self.device.type != "cuda":
            raise ValueError("The DiffNav CUDA Graph controller requires a CUDA device.")
        self.num_envs = controller.num_envs
        self.inputs = tuple(value.clone() for value in sample_inputs)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        # Preserve the application's RNG during preparation. RotorGroup's
        # original zero-weight noise still consumes the same RNG on each replay.
        with torch.random.fork_rng(devices=[self.device.index]):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    controller.compute(*self.inputs)
            current.wait_stream(stream)
            torch.cuda.synchronize(self.device)
            with torch.cuda.graph(self.graph, stream=stream):
                controller.compute(*self.inputs)
            current.wait_stream(stream)
        self.reset(torch.arange(self.num_envs, device=self.device))
        torch.cuda.synchronize(self.device)
        self.capture_setup_wall_s = time.perf_counter() - started

    @property
    def commands(self):
        return self.controller.commands

    @property
    def velocity_gain(self):
        return self.controller.velocity_gain

    @property
    def nominal_velocity_gain(self):
        return self.controller.nominal_velocity_gain

    @property
    def rotor_parameters(self):
        return self.controller.rotor_parameters

    @property
    def rotor_group(self):
        return self.controller.rotor_group

    @property
    def thrusts(self):
        return self.controller.thrusts

    @property
    def moments(self):
        return self.controller.moments

    @property
    def throttle(self):
        return self.controller.throttle

    def compute(self, root_state_xyzw, velocity_world, acceleration_world=None, yaw_world=None):
        if acceleration_world is None:
            # The original optional acceleration is the same zero feed-forward.
            self.inputs[2].zero_()
            acceleration_world = self.inputs[2]
        if yaw_world is None:
            # Preserve the original Lee current-yaw default (input is XYZW).
            x, y, z, w = root_state_xyzw[:, 3:7].unbind(-1)
            yaw_world = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        for destination, source in zip(
            self.inputs, (root_state_xyzw, velocity_world, acceleration_world, yaw_world.reshape(-1, 1)), strict=True
        ):
            destination.copy_(source)
        self.graph.replay()
        return self.commands

    def apply(self, drone, disturbance_acceleration_w=None):
        return self.controller.apply(drone, disturbance_acceleration_w)

    def reset(self, env_ids):
        # The original reset writes all four motor buffers in place. Never
        # recreate the controller or rebind any captured tensor on a reset.
        return self.controller.reset(env_ids)
