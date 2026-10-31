"""Numerical parity checks for the Navigation Warp kernels, without simulation.

Run in the project's environment, for example:
    .venv/bin/python tests/warp_task_parity.py --device cuda:0
Or pass --project-root when this script is outside the repository.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(obstacle_count, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    count = 32

    def normal(*shape):
        return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

    # Exercise the strided position/velocity views used by real root-state arrays.
    root = normal(count, 13)
    position, velocity = root[:, :3], root[:, 7:10]
    position[:, :2] *= 3
    position[:, 2] = torch.linspace(0.1, 4.1, count, device=device)
    position[0] = torch.tensor((0.0, 0.0, 2.0), device=device)
    position[6, 2], position[7, 2] = 0.2, 4.0
    target = position + normal(count, 3)
    target[0] = position[0]
    target[3] = position[3] + torch.tensor((0.49, 0.0, 0.0), device=device)
    target[4] = position[4] + torch.tensor((0.5, 0.0, 0.0), device=device)
    direction = normal(count, 3)
    direction[0] = 0
    direction[1] = torch.tensor((1.0e-12, -1.0e-12, 1.0), device=device)
    direction[2] = torch.tensor((1.0, 0.0, 0.0), device=device)
    lidar = torch.rand((count, 144), generator=generator, device=device) * 5.0 - 0.2
    lidar[0, :5] = torch.tensor((0.0, 1.0e-8, 4.0, 4.0001, 1.0), device=device)
    obstacle_position = normal(obstacle_count, 3) * 6.0
    obstacle_velocity = normal(obstacle_count, 3)
    obstacle_size = torch.ones((obstacle_count, 3), device=device)
    for index in range(obstacle_count):
        width = (index % 4 + 1) * 0.25
        obstacle_size[index] = torch.tensor((width, width, 5.0 if index % 2 else 1.0), device=device)
    positions = ((4.0, 0.0, 2.0), (4.0001, 0.0, 2.0), (0.0, 0.0, 2.0), (-3.0, 0.0, 9.0))
    for index, value in enumerate(positions[:obstacle_count]):
        obstacle_position[index] = torch.tensor(value, device=device)
    selected_distance, selected_indices = (
        (obstacle_position[None, :, :2] - position[:, None, :2])
        .norm(dim=-1)
        .topk(min(5, obstacle_count), dim=-1, largest=False)
    )
    return dict(
        position=position,
        velocity=velocity,
        target=target,
        direction=direction,
        lidar_distance=lidar,
        obstacle_position=obstacle_position,
        obstacle_velocity=obstacle_velocity,
        obstacle_size=obstacle_size,
        selected_indices=selected_indices,
        selected_distance_xy=selected_distance,
        previous_velocity=normal(count, 3),
        height_range=torch.tensor((1.0, 3.0), device=device).expand(count, -1).clone(),
    )


def reference_terms(task_logic, inputs):
    motion = SimpleNamespace(
        position=inputs["obstacle_position"],
        velocity=inputs["obstacle_velocity"],
        size=inputs["obstacle_size"],
        cylinders=inputs["obstacle_size"][:, 2] > 1.0,
    )
    return task_logic.navigation_terms(
        inputs["position"],
        inputs["velocity"],
        inputs["target"],
        inputs["direction"],
        inputs["lidar_distance"].view(-1, 1, 36, 4),
        motion,
        inputs["previous_velocity"],
        inputs["height_range"],
    )


def check_terms(runner, reference):
    assert runner.terms.keys() == reference.keys()
    largest = 0.0
    for key, expected in reference.items():
        actual = runner.terms[key]
        if expected.dtype == torch.bool:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=key)
        else:
            torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-5, equal_nan=True, msg=key)
            finite = torch.isfinite(actual) & torch.isfinite(expected)
            if finite.any():
                largest = max(largest, (actual[finite] - expected[finite]).abs().max().item())
    expected_policy = torch.cat(
        tuple(reference[key].flatten(1) for key in ("state", "lidar", "dynamic_obstacle")), dim=-1
    )
    torch.testing.assert_close(runner.policy_tensor, expected_policy, rtol=3.0e-5, atol=3.0e-5, equal_nan=True)
    return largest


def check_navigation(task_logic, kernels, device):
    results = []
    for obstacle_count in (0, 1, 5, 8, 120):
        inputs = make_inputs(obstacle_count, device, 270 + obstacle_count)
        runner = kernels.WarpNavigationTerms(**inputs)
        pointers = {key: value.data_ptr() for key, value in runner.terms.items()}
        runner.launch()
        wp.synchronize_device(str(device))
        largest = check_terms(runner, reference_terms(task_logic, inputs))
        graph = None
        if device.type == "cuda":
            with wp.ScopedCapture(device=str(device)) as capture:
                runner.launch()
            graph = capture.graph
        # Fresh values and allocations are copied into the original bindings.
        runner.update(**make_inputs(obstacle_count, device, 920 + obstacle_count))
        if graph is None:
            runner.launch()
        else:
            wp.capture_launch(graph)
        wp.synchronize_device(str(device))
        largest = max(largest, check_terms(runner, reference_terms(task_logic, inputs)))
        assert pointers == {key: value.data_ptr() for key, value in runner.terms.items()}
        results.append(
            {"obstacles": obstacle_count, "max_absolute_error": largest, "graph_replayed": graph is not None}
        )

    # Non-finite masks and clamp behavior must also agree with the Torch code.
    inputs = make_inputs(8, device, 551)
    inputs["position"][0] = torch.tensor((100.0, 100.0, 2.0), device=device)
    inputs["direction"][0, 0] = torch.nan
    inputs["lidar_distance"][1, :3] = torch.tensor((torch.nan, torch.inf, -torch.inf), device=device)
    distances, indices = (
        (inputs["obstacle_position"][None, :, :2] - inputs["position"][:, None, :2])
        .norm(dim=-1)
        .topk(5, dim=-1, largest=False)
    )
    inputs["selected_distance_xy"].copy_(distances)
    inputs["selected_indices"].copy_(indices)
    runner = kernels.WarpNavigationTerms(**inputs)
    runner.launch()
    wp.synchronize_device(str(device))
    check_terms(runner, reference_terms(task_logic, inputs))
    return results


def check_goal_frames(task_logic, kernels, device):
    inputs = make_inputs(0, device, 440)
    out = torch.empty_like(inputs["position"])
    views = [wp.from_torch(inputs[key]) for key in ("position", "direction")]
    for inverse in (False, True):
        wp.launch(
            kernels.goal_frame_kernel,
            dim=len(out),
            inputs=views + [inverse],
            outputs=[wp.from_torch(out)],
            device=str(device),
        )
        wp.synchronize_device(str(device))
        expected = task_logic.goal_frame(inputs["position"], inputs["direction"], inverse=inverse)
        torch.testing.assert_close(out, expected, rtol=3.0e-5, atol=3.0e-5)


def check_events(termination, kernels, device):
    count = 16
    root = torch.zeros((count, 13), device=device)
    root[:, 2], root[:, 6] = 2.0, 1.0
    target = root[:, :3].clone()
    target[:, 0] += 3.0
    gap = torch.ones(count, device=device)
    root[1:4, 2] = 0.1
    target[:4] = root[:4, :3]
    gap[2:5] = torch.tensor((0.0, -1.0, -0.1), device=device)
    gap[5] = torch.nan
    root[3, 6] = torch.nan
    root[6, 2], root[7, 2], root[8, 2] = 4.1, 0.2, 4.0
    root[9, 7], root[10, 12] = torch.inf, -torch.inf
    target[7] = root[7, :3] + torch.tensor((0.5, 0.0, 0.0), device=device)
    codes = torch.zeros(count, dtype=torch.int8, device=device)
    times = torch.zeros(count, device=device)
    codes[11], times[11] = 5, 0.125
    elapsed = torch.linspace(0.1, 1.6, count, device=device)
    expected_codes, expected_times = codes.clone(), times.clone()
    reference = termination.first_events(root, target, gap)

    for collision_terminates, bounds_terminates in ((True, True), (False, False), (True, False), (False, True)):
        runner = kernels.WarpFirstEvents(
            root_state=root,
            target=target,
            collision_gap=gap,
            collision_terminates=collision_terminates,
            bounds_terminates=bounds_terminates,
            event_code=codes,
            event_time=times,
            elapsed_s=elapsed,
        )
        runner.launch()
        wp.synchronize_device(str(device))
        for key, expected in reference.items():
            torch.testing.assert_close(runner.terms[key], expected, rtol=0, atol=0, msg=key)
        torch.testing.assert_close(
            runner.terminated,
            termination.termination_mask(
                reference,
                collision_terminates=collision_terminates,
                bounds_terminates=bounds_terminates,
            ),
            rtol=0,
            atol=0,
        )
    # The final runner is reused to prove first-event priority and persistence.
    termination.latch_first_event(expected_codes, expected_times, reference, elapsed)
    runner.launch_latch()
    wp.synchronize_device(str(device))
    torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(times, expected_times, rtol=0, atol=0)
    assert codes[:4].tolist() == [1, 3, 2, 4]
    graph = None
    if device.type == "cuda":
        with wp.ScopedCapture(device=str(device)) as capture:
            runner.launch()
            runner.launch_latch()
        graph = capture.graph
    runner.update(collision_gap=torch.full_like(gap, -1.0), elapsed_s=elapsed + 2.0)
    reference = termination.first_events(root, target, gap)
    termination.latch_first_event(expected_codes, expected_times, reference, elapsed)
    if graph is None:
        runner.launch()
        runner.launch_latch()
    else:
        wp.capture_launch(graph)
    wp.synchronize_device(str(device))
    for key, expected in reference.items():
        torch.testing.assert_close(runner.terms[key], expected, rtol=0, atol=0, msg=key)
    torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(times, expected_times, rtol=0, atol=0)
    reset_mask = torch.arange(count, device=device) % 3 == 0
    wp.launch(
        kernels.reset_first_event_kernel,
        dim=count,
        inputs=[wp.from_torch(reset_mask)],
        outputs=[wp.from_torch(codes), wp.from_torch(times)],
        device=str(device),
    )
    termination.reset_first_event(expected_codes, expected_times, reset_mask.nonzero().flatten())
    wp.synchronize_device(str(device))
    torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(times, expected_times, rtol=0, atol=0)
    return {"first_event_priority": [1, 3, 2, 4], "graph_replayed": graph is not None, "partial_reset": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    navigation = args.project_root / "src/aerialnav/tasks/navigation"
    task_logic = load_module("parity_original_task_logic", navigation / "task_logic.py")
    kernels = load_module("parity_warp_task_kernels", navigation / "warp_task_kernels.py")
    termination = load_module(
        "parity_original_termination", args.project_root / "src/aerialnav/diffnav/termination.py"
    )
    device = torch.device(args.device)
    wp.init()
    caller_stream = None
    torch_context = contextlib.nullcontext()
    warp_context = contextlib.nullcontext()
    if device.type == "cuda":
        # Warp capture needs a registered non-default stream. Both libraries
        # execute on it, with explicit ordering against the caller at boundaries.
        caller_stream = torch.cuda.current_stream(device)
        warp_stream = wp.Stream(str(device))
        torch_stream = wp.stream_to_torch(warp_stream)
        torch_stream.wait_stream(caller_stream)
        torch_context = torch.cuda.stream(torch_stream)
        warp_context = wp.ScopedStream(warp_stream)
    try:
        with wp.ScopedDevice(str(device)), torch_context, warp_context:
            terms = check_navigation(task_logic, kernels, device)
            check_goal_frames(task_logic, kernels, device)
            events = check_events(termination, kernels, device)
    finally:
        if caller_stream is not None:
            caller_stream.wait_stream(torch_stream)
    print(json.dumps({"status": "passed", "device": str(device), "navigation": terms, "events": events}, indent=2))


if __name__ == "__main__":
    main()
