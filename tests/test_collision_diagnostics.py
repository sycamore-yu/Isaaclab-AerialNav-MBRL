"""First collision diagnostics preserve subtype and pre-reset relative velocity."""

from types import SimpleNamespace

import pytest
import torch

from aerialnav.diffnav.diagnostics import collision_records


def test_collision_categories_do_not_count_collision_after_success():
    env = SimpleNamespace(
        terrain_distance=SimpleNamespace(signed_distance=lambda p: p[:, 0] + 10),
        motion=SimpleNamespace(size=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0]])),
    )
    transition = {
        "collision_position": torch.tensor(
            [[-9.8, 0, 2.0], [5.0, 0, 0.2], [0.7, 0, 2.0], [10.7, 0, 2.0], [0.7, 0, 2.0]]
        ),
        "collision_velocity": torch.tensor([[-2.0, 0, 0], [0, 0, -3.0], [-2.0, 0, 0], [-2.0, 0, 0], [-2.0, 0, 0]]),
        "collision_substep": torch.ones(5, dtype=torch.long),
        "first_event_code": torch.tensor([2, 2, 2, 2, 1]),
        "dynamic_positions_by_substep": torch.tensor([[[0.0, 0, 2.0], [10.0, 0, 2.0]]]),
        "dynamic_velocities_by_substep": torch.tensor([[[1.0, 0, 0], [1.0, 0, 0]]]),
    }
    result = collision_records(env, transition)
    assert [event["type"] for event in result] == ["static", "ground", "dynamic_box", "dynamic_cylinder"]
    assert [event["closing_speed_mps"] for event in result] == pytest.approx([2, 3, 3, 3])
    assert all(event["clearance_m"] < 0 for event in result)
    assert result[-1]["height_m"] == 2


def test_dynamic_metrics_match_smooth_face_derivatives_without_equal_scaling():
    from aerialnav.diffnav.diagnostics import compare_dynamic_metrics

    report = compare_dynamic_metrics(
        torch.tensor([[2.0, 0.1, 2.0]], dtype=torch.double),
        torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.double),
        torch.ones(1, 3, dtype=torch.double),
    )
    assert report["metrics"]["dcol"]["gradient_norm_mean"] == pytest.approx(0.375)
    assert report["metrics"]["sdf"]["gradient_norm_mean"] == pytest.approx(1.0)
    assert report["gradient_cosine_mean"] == pytest.approx(1.0)
    assert all(item["fd_error_max"] < 1e-10 for item in report["metrics"].values())


def test_cached_static_query_retains_its_live_escape_gradient():
    from aerialnav.diffnav.runner import DiffNavTrainer

    def forbidden_query(_):
        raise AssertionError("cached path must not query terrain again")

    env = SimpleNamespace(
        terrain_distance=SimpleNamespace(signed_distance=forbidden_query),
        motion=SimpleNamespace(position=torch.empty(0, 3), size=torch.empty(0, 3)),
    )
    trainer = SimpleNamespace(env=env, cfg=SimpleNamespace(collision_metric="dcol"))
    points = torch.tensor([[[0.0, 0.0, 2.0]]], requires_grad=True)
    cached = 1 + 0.4 * points[..., 0]
    gap = DiffNavTrainer._dcol_gap(trainer, points, torch.tensor([0.3]), static_distance=cached)
    assert float(gap.detach()) == pytest.approx(0.7)
    torch.testing.assert_close(torch.autograd.grad(gap.sum(), points)[0], torch.tensor([[[0.4, 0.0, 0.0]]]))
