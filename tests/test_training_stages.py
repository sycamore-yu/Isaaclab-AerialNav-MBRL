"""The four named stages gate on completed training episodes, including exactly 80%."""

import pytest
import torch

from isaaclab_navrl.tasks.navigation.curriculum import STAGE_CONFIGS, TrainingCurriculum


@pytest.mark.parametrize("stage", list(STAGE_CONFIGS))
def test_named_stage_requires_eighty_percent_and_has_no_internal_map_switch(stage):
    curriculum = TrainingCurriculum(10, "cpu", stage=stage)
    for successes in (7, 8):
        code = torch.tensor([1] * successes + [2] * (10 - successes))
        curriculum.observe({
            "first_event_code": code, "first_event_time_s": torch.full((10,), 4.5),
            "terminated": torch.ones(10, dtype=torch.bool), "timeout": torch.zeros(10, dtype=torch.bool),
            "episode_return": torch.full((10,), 20.0), "step": torch.full((10,), 50),
        })
        report = curriculum.pop_report(update=successes)
        assert report["success_rate"] == successes / 10
        assert report["static_obstacles"] == STAGE_CONFIGS[stage]["static_obstacles"]
        assert report["dynamic_obstacles"] == STAGE_CONFIGS[stage]["dynamic_obstacles"]
        assert report["randomization"] == STAGE_CONFIGS[stage]["randomization"]
        assert curriculum.stage_passed is (successes == 8)
        assert not curriculum.ready_to_advance
        assert curriculum.acceptance_passed is (stage == "finetune_s2" and successes == 8)
    assert curriculum.state_dict()["stages"] == list(STAGE_CONFIGS)
