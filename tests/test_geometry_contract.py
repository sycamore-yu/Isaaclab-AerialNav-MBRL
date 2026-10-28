import torch
import trimesh

from isaaclab_navrl.tasks.navigation.geometry import TerrainDistance, nearest_obstacle_vector


def test_nearest_box_has_the_source_position_gradient():
    query = torch.tensor([[2.0, 0.0, 2.0]], requires_grad=True)
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 2.0))
    mesh.apply_translation((0.0, 0.0, 2.0))
    distance = TerrainDistance(mesh.vertices, mesh.faces, "cpu")
    empty = torch.empty(0, 3)
    vector = nearest_obstacle_vector(query, distance.closest_points(query), empty, empty)
    torch.testing.assert_close(vector, torch.tensor([[-1.5, 0.0, 0.0]]))
    vector.norm(dim=-1).sum().backward()
    torch.testing.assert_close(query.grad, torch.tensor([[1.0, 0.0, 0.0]]))


def test_dynamic_cylinder_keeps_its_round_cross_section():
    query = torch.tensor([[1.0, 1.0, 2.0]])
    distant = torch.full_like(query, 100.0)
    vector = nearest_obstacle_vector(query, distant, torch.tensor([[0.0, 0.0, 2.5]]), torch.tensor([[1.0, 1.0, 5.0]]))
    torch.testing.assert_close(vector.norm(dim=-1), torch.tensor([2**0.5 - 0.5]))
