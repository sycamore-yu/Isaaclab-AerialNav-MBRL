"""Newton depth sensor configuration."""

from isaaclab.sensors import MultiMeshRayCasterCameraCfg
from isaaclab.utils import configclass


@configclass
class BatchedDepthCameraCfg(MultiMeshRayCasterCameraCfg):
    class_type = "isaaclab_navrl.tasks.navigation.newton_sensors:NewtonBatchedDepthCamera"
    batch_size: int = 16
    pixel_noise_std: float = 0.02
    seed: int = 0
    update_mesh_ids: bool = False
