"""The optional backend is isolated and checkpoint factor changes are explicit."""

import pytest
import torch

from isaaclab_navrl.tasks.navigation.geometry import nearest_obstacle_signed_distance
from isaaclab_navrl.zhang.cli import parse_train_args
from isaaclab_navrl.zhang.config import ZhangTrainCfg


def test_optional_backend_cli_and_validation():
    assert ZhangTrainCfg().geometry_backend == "curobo_v2"
    values = parse_train_args(["--geometry_backend", "curobo_v2"])
    assert values["geometry_backend"] == "curobo_v2"
    ZhangTrainCfg(geometry_backend="curobo_v2").validate()
    with pytest.raises(ValueError, match="finite_difference"):
        ZhangTrainCfg(geometry_backend="curobo_v2", approach_mode="relative").validate()


def test_all_geometry_consumers_can_select_the_backend_with_query_gradients():
    p = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], requires_grad=True)
    positions = torch.zeros(8, 3)
    sizes = torch.ones(8, 3)
    seen = []

    def backend(query, poses, extents, *, return_index):
        assert return_index and poses is positions and extents is sizes
        seen.append(query)
        return query[:, 0] * 0.5, torch.full((2,), 7, dtype=torch.long)

    distance = nearest_obstacle_signed_distance(p, p[:, 0] * 0 + 100.0, positions, sizes, dynamic_backend=backend)
    torch.testing.assert_close(
        torch.autograd.grad(distance.sum(), p)[0], torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    )
    assert len(seen) == 1
