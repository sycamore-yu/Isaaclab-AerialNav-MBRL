"""Record the actual queried mesh, vehicle pose and valid LiDAR hits in Rerun."""

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import warp as wp
from pxr import Usd, UsdGeom


def recording_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world/vehicle", contents=["/world/**"],
                name="Flight · body-fixed chase · actual hits",
                eye_controls=rrb.EyeControls3D(
                    position=(3, -7, 4), look_target=(0, 0, 0), eye_up=(0, 0, 1),
                ),
            ),
            rrb.Vertical(
                rrb.Spatial3DView(
                    origin="body_frame", name="Body XYZ · valid hits only",
                    eye_controls=rrb.EyeControls3D(
                        position=(-4, -6, 5), look_target=(4, 0, 0), eye_up=(0, 0, 1),
                    ),
                ),
                rrb.TimeSeriesView(origin="metrics", name="Speed / clearance / hit fraction"),
                rrb.TextDocumentView(origin="status", name="Policy / outcome"),
                row_shares=[3, 2, 1],
            ), column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="sim_time", expanded=True), collapse_panels=True,
    )


class PointCloudRecording:
    def __init__(self, folder, env):
        self.folder, self.env = Path(folder), env
        if env.num_envs != 1:
            raise ValueError("Record one navigation episode at a time")
        self.stream = rr.RecordingStream("PointNet · static obstacle avoidance · measured LiDAR hits")
        self.stream.save(self.folder / "navigation.rrd", default_blueprint=recording_blueprint())
        self.stream.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self.stream.log("body_frame", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        sensor = env.scene["point_lidar"]
        mesh = sensor.meshes[(sensor.cfg.mesh_prim_paths[0], env.device)]
        self.stream.log(
            "world/obstacles",
            rr.Mesh3D(
                vertex_positions=wp.to_torch(mesh.points).cpu().numpy(),
                triangle_indices=wp.to_torch(mesh.indices).cpu().numpy().reshape(-1, 3),
                albedo_factor=[105, 119, 133, 255],
            ),
            static=True,
        )
        self.stream.log(
            "world/goal",
            rr.Points3D(env.target_pos.cpu().numpy(), radii=0.25, colors=[255, 209, 76], labels=["Goal"]),
            static=True,
        )
        self.stream.log("body_frame/origin", rr.Points3D([[0, 0, 0]], radii=0.12, colors=[90, 175, 255]), static=True)
        source = Path(__file__).resolve().parents[1] / "assets/hummingbird.usd"
        stage = Usd.Stage.Open(str(source))
        cache = UsdGeom.XformCache()
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or prim.GetName() != "visuals":
                continue
            visual = UsdGeom.Mesh(prim)
            faces = np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            points = np.asarray(visual.GetPointsAttr().Get())
            vertices = np.column_stack((points, np.ones(len(points)))) @ np.asarray(
                cache.GetLocalToWorldTransform(prim)
            )
            color = [65, 165, 235] if "/base_link/" in str(prim.GetPath()) else [245, 130, 85]
            self.stream.log(
                "world/vehicle/" + str(prim.GetPath()).strip("/"),
                rr.Mesh3D(vertex_positions=vertices[:, :3], triangle_indices=faces, albedo_factor=color),
                static=True,
            )
        self.positions, self.quaternions, self.velocities, self.times = [], [], [], []
        self.hits, self.body_hits, self.offsets, self.clearances = [], [], [0], []

    def frame(self, time_s):
        state = self.env.root_state()[0].detach().cpu().numpy()
        observation = self.env.get_diffnav_observation()
        packed = observation["policy_points"][0].cpu().numpy()
        valid = packed[3] > 0
        hits = self.env.scene["point_lidar"].data.ray_hits_w.torch[0].cpu().numpy()[valid]
        assert np.isfinite(hits).all()
        position, quaternion, velocity = state[:3], state[3:7], state[7:10]
        clearance = float(self.env.nearest_obstacle_signed_distance(self.env.root_state()[:, :3])[0]) - float(
            self.env.diffnav_parameters["margin"][0]
        )
        self.positions.append(position.copy())
        self.quaternions.append(quaternion.copy())
        self.velocities.append(velocity.copy())
        self.times.append(time_s)
        self.clearances.append(clearance)
        self.hits.append(hits.copy())
        self.body_hits.append(packed[:3, valid].T.copy())
        self.offsets.append(self.offsets[-1] + len(hits))
        self.stream.set_time("sim_time", duration=time_s)
        self.stream.log("world/vehicle", rr.Transform3D(translation=position, rotation=rr.Quaternion(xyzw=quaternion)))
        self.stream.log(
            "world/trajectory", rr.LineStrips3D([np.asarray(self.positions)], colors=[255, 184, 90], radii=0.035)
        )
        self.stream.log("world/lidar/hits", rr.Points3D(hits, colors=[70, 225, 210], radii=0.035))
        self.stream.log("body_frame/hits", rr.Points3D(packed[:3, valid].T, colors=[70, 225, 210], radii=0.04))
        for name, value in [
            ("speed_mps", np.linalg.norm(velocity)),
            ("clearance_m", clearance),
            ("hit_fraction", len(hits) / packed.shape[-1]),
        ]:
            self.stream.log("metrics/" + name, rr.Scalars(float(value)))
        self.stream.log(
            "status",
            rr.TextDocument(
                "PointNet–GRU192 · static S0\n"
                f"t={time_s:.2f} s · {len(hits)}/5400 valid hits\n"
                "Cyan: measured hits; misses omitted. Orange: executed trajectory."
            ),
        )

    def close(self, result):
        self.stream.log(
            "status", rr.TextDocument(
                f"Episode: {result['outcome']}\n"
                f"Static S0 · {len(self.times)} frames\n"
                f"Minimum recorded clearance: {min(self.clearances):.3f} m\n"
                "Cyan: actual hits only · Orange: flown trajectory"
            )
        )
        self.stream.flush()
        np.savez_compressed(
            self.folder / "trace.npz",
            time_s=self.times,
            position=self.positions,
            quaternion_xyzw=self.quaternions,
            velocity=self.velocities,
            clearance_m=self.clearances,
            hit_offsets=self.offsets,
            hits_world=np.concatenate(self.hits),
            hits_body=np.concatenate(self.body_hits),
        )
        self.stream.disconnect()
