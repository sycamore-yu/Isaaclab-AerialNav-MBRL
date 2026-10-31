"""The five named stages gate on completed training episodes, including exactly 80%."""

import pytest
import torch

from aerialnav.tasks.navigation.curriculum import STAGE_CONFIG, TrainingCurriculum


def test_formal_curriculum_starts_without_dynamic_obstacles():
    assert list(STAGE_CONFIG.items()) == [
        ("pretrain_s0", {"static_obstacles": 300, "dynamic_obstacles": 0, "randomization": False}),
        ("pretrain_s1", {"static_obstacles": 300, "dynamic_obstacles": 40, "randomization": False}),
        ("pretrain_s2", {"static_obstacles": 400, "dynamic_obstacles": 120, "randomization": False}),
        ("finetune_s1", {"static_obstacles": 300, "dynamic_obstacles": 40, "randomization": True}),
        ("finetune_s2", {"static_obstacles": 400, "dynamic_obstacles": 100, "randomization": True}),
    ]
    curriculum = TrainingCurriculum(10, "cpu", stage="pretrain_s0")
    transition = {
        "first_event_code": torch.tensor([1] * 8 + [3] * 2),
        "first_event_time_s": torch.full((10,), 4.5),
        "terminated": torch.zeros(10, dtype=torch.bool),
        "timeout": torch.zeros(10, dtype=torch.bool),
        "episode_return": torch.full((10,), 20.0),
        "step": torch.full((10,), 50),
    }
    curriculum.observe(transition)
    assert curriculum.pop_report(update=1) is None and not curriculum.stage_passed
    transition.update(timeout=torch.ones(10, dtype=torch.bool), step=torch.full((10,), 100))
    curriculum.observe(transition)
    report = curriculum.pop_report(update=2)
    assert report["episodes"] == 10 and report["success_rate"] == 0.8
    assert report["static_obstacles"] == 300 and report["dynamic_obstacles"] == 0
    assert report["randomization"] is False and report["bounds_rate"] == 0.2
    assert curriculum.stage_passed and not curriculum.acceptance_passed
    assert not curriculum.ready_to_advance


@pytest.mark.parametrize("stage", list(STAGE_CONFIG))
def test_named_stage_requires_eighty_percent_and_has_no_internal_map_switch(stage):
    curriculum = TrainingCurriculum(10, "cpu", stage=stage)
    for successes in (7, 8):
        code = torch.tensor([1] * successes + [2] * (10 - successes))
        curriculum.observe(
            {
                "first_event_code": code,
                "first_event_time_s": torch.full((10,), 4.5),
                "terminated": torch.ones(10, dtype=torch.bool),
                "timeout": torch.zeros(10, dtype=torch.bool),
                "episode_return": torch.full((10,), 20.0),
                "step": torch.full((10,), 50),
            }
        )
        report = curriculum.pop_report(update=successes)
        assert report["success_rate"] == successes / 10
        assert report["static_obstacles"] == STAGE_CONFIG[stage]["static_obstacles"]
        assert report["dynamic_obstacles"] == STAGE_CONFIG[stage]["dynamic_obstacles"]
        assert report["randomization"] == STAGE_CONFIG[stage]["randomization"]
        assert curriculum.stage_passed is (successes == 8)
        assert not curriculum.ready_to_advance
        assert curriculum.acceptance_passed is (stage == "finetune_s2" and successes == 8)
    assert curriculum.state_dict()["stages"] == list(STAGE_CONFIG)
