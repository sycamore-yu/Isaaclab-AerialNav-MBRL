"""Passive configuration; the PhysX sensor implementation loads after launch."""

from isaaclab.sensors import MultiMeshRayCasterCameraCfg
from isaaclab.utils import configclass


@configclass
class BatchedDepthCameraCfg(MultiMeshRayCasterCameraCfg):
    class_type = "aerialnav.tasks.navigation.depth_camera:BatchedDepthCamera"
    batch_size: int = 16
    pixel_noise_std: float = 0.02
    seed: int = 0
    update_mesh_ids: bool = False
