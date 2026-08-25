#!/usr/bin/env python3
"""Isaac Sim soccer setup — Booster K1 + ball + goals + field, ROS-free.

Foundation for the kick task: one Booster K1 facing a size-5 ball with a goal
behind it, on a marked pitch matching the MuJoCo robocup demo (8 x 5 m, goals
2.0 x 0.9 m). Renders for visual verification:

  docs/images/isaac_soccer_field.png      overhead view of the full pitch
  docs/images/isaac_soccer_robot.png      front view from robot-head height
  docs/images/isaac_soccer_seg.png        instance segmentation (front cam)

Usage (venv-isaac python):
  python src/k1_sim_isaac/scripts/soccer_sim.py --capture
  python src/k1_sim_isaac/scripts/soccer_sim.py --steps 100
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=60,
                    help="settle steps before capture")
parser.add_argument("--outdir", default="docs/images")
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

# field dims — parity with mujoco_robocup_demo.py
FIELD_L, FIELD_W = 8.0, 5.0
GOAL_W, GOAL_H = 2.0, 0.9
BALL_R = 0.11

ROBOT_X, BALL_X = -2.5, 0.0  # robot 2.5 m behind the ball, facing +x at the goal


def _visual_box(size, pos, color, name):
    """Visual-only box (no physics) for pitch markings."""
    return AssetBaseCfg(
        prim_path=f"/World/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def look_at_quat_xyzw(pos, target, up=(0.0, 0.0, 1.0)):
    """World-convention camera quaternion (x, y, z, w) for OffsetCfg with
    convention="world": rotates world +X onto the (target - pos) forward
    direction with world +Z as up."""
    pos, target, up = map(lambda v: np.asarray(v, float), (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    up2 = up - fwd * np.dot(up, fwd)      # up orthogonal to fwd
    up2 /= np.linalg.norm(up2)
    right = np.cross(up2, fwd)            # y = z x x keeps right-handed
    R = np.column_stack([fwd, right, up2])  # images of +X, +Y, +Z
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
        x, y, z, w = q
    q = np.array([x, y, z, w])
    return tuple(float(v) for v in q / np.linalg.norm(q))


@configclass
class SoccerSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = None  # set in __post_init__ per --robot

    def __post_init__(self):
        from k1_velocity.tasks.velocity.velocity_env_cfg import K1_ARTICULATION_CFG

        self.robot = K1_ARTICULATION_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_0",
            init_state=K1_ARTICULATION_CFG.init_state.replace(
                pos=(ROBOT_X, 0.0, K1_ARTICULATION_CFG.init_state.pos[2]),
            ),
        )

        # pitch
        self.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        self.turf = _visual_box((FIELD_L + 2.0, FIELD_W + 2.0, 0.01),
                                (0.0, 0.0, 0.005), (0.12, 0.45, 0.16),
                                "turf")
        # markings (white, visual-only)
        line_specs = [
            ("line_mid", (0.06, FIELD_W, 0.01), (0.0, 0.0, 0.011)),
            ("line_n", (FIELD_L, 0.06, 0.01), (0.0, FIELD_W / 2, 0.011)),
            ("line_s", (FIELD_L, 0.06, 0.01), (0.0, -FIELD_W / 2, 0.011)),
            ("line_w", (0.06, FIELD_W, 0.01), (-FIELD_L / 2, 0.0, 0.011)),
            ("line_e", (0.06, FIELD_W, 0.01), (FIELD_L / 2, 0.0, 0.011)),
        ]
        for name, size, pos in line_specs:
            setattr(self, name, _visual_box(size, pos, (0.95, 0.95, 0.95),
                                            name))

        # goals: posts + crossbar (kinematic so the ball bounces off)
        # NOTE: USD prim names allow [A-Za-z0-9_] only — never f"{sign}" with
        # +/- (Sdf.Path silently parses invalid names to empty).
        def goal(sign):
            side = "pos" if sign > 0 else "neg"
            x = sign * FIELD_L / 2
            parts = []
            for i, y in enumerate((GOAL_W / 2, -GOAL_W / 2)):
                parts.append((f"goal_post_{side}_{i}",
                              AssetBaseCfg(
                                  prim_path=f"/World/goal_post_{side}_{i}",
                                  spawn=sim_utils.CylinderCfg(
                                      radius=0.04, height=GOAL_H,
                                      rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                          kinematic_enabled=True),
                                      visual_material=sim_utils.PreviewSurfaceCfg(
                                          diffuse_color=(0.9, 0.9, 0.9)),
                                  ),
                                  init_state=AssetBaseCfg.InitialStateCfg(
                                      pos=(x, y, GOAL_H / 2)),
                              )))
            parts.append((f"goal_bar_{side}",
                          AssetBaseCfg(
                              prim_path=f"/World/goal_bar_{side}",
                              spawn=sim_utils.CuboidCfg(
                                  size=(0.04, GOAL_W + 0.08, 0.04),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                      kinematic_enabled=True),
                                  visual_material=sim_utils.PreviewSurfaceCfg(
                                      diffuse_color=(0.9, 0.9, 0.9)),
                              ),
                              init_state=AssetBaseCfg.InitialStateCfg(
                                  pos=(x, 0.0, GOAL_H)),
                          )))
            return parts

        for name, cfg in goal(+1) + goal(-1):
            setattr(self, name, cfg)

        # ball (size-5, RoboCup mass)
        self.ball = AssetBaseCfg(
            prim_path="/World/ball",
            spawn=sim_utils.SphereCfg(
                radius=BALL_R,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.1)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(BALL_X, 0.0, BALL_R + 0.01)),
        )

        self.sky = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=400.0),
        )

        # front camera at robot-head height, looking at ball/goal (+x)
        f_px = 527.6  # ZED 2i 2.1 mm @ HD720 (matches stereo rig)
        self.cam_front = CameraCfg(
            prim_path="/World/cam_front",
            offset=CameraCfg.OffsetCfg(
                pos=(ROBOT_X + 0.45, 0.0, 0.95),
                rot=(0.5, -0.5, 0.5, -0.5)),  # ROS optical, +x forward
            height=720, width=1280, update_period=0.0,
            data_types=["rgb", "instance_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=f_px * 0.1,
                horizontal_aperture=1280 * 0.1,
                clipping_range=(0.1, 50.0),
            ),
        )
        self.cam_overhead = CameraCfg(
            prim_path="/World/cam_overhead",
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -7.0, 5.0),
                rot=look_at_quat_xyzw((0.0, -7.0, 5.0), (0.0, 0.0, 0.3)),
                convention="world"),
            height=720, width=1280, update_period=0.0,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=35.0, horizontal_aperture=50.0,
                clipping_range=(0.1, 100.0),
            ),
        )


def main():
    import cv2  # post-app ok
    import imageio.v2 as imageio

    os.makedirs(args.outdir, exist_ok=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, render_interval=4))
    scene = InteractiveScene(SoccerSceneCfg(num_envs=1, env_spacing=10.0))
    sim.reset()
    scene.update(0.005)

    art = scene["robot"]
    cam_front: Camera = scene["cam_front"]
    cam_overhead: Camera = scene["cam_overhead"]

    # hold default pose (actuator PD keeps stance)
    for _ in range(args.steps):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(0.005)
        cam_front.update(0.005)
        cam_overhead.update(0.005)

    def rgb(cam):
        return cam.data.output["rgb"][0, :, :, :3].cpu().numpy()[..., ::-1].copy()

    front = rgb(cam_front)
    over = rgb(cam_overhead)
    imageio.imwrite(f"{args.outdir}/isaac_soccer_field.png", over)
    imageio.imwrite(f"{args.outdir}/isaac_soccer_robot.png", front)

    # instance segmentation from the front camera (best-effort).
    # instance_segmentation_fast returns an (H, W, 4) ID encoding — decode the
    # integer instance ID from the first three channels, then colorize.
    try:
        seg = cam_front.data.output["instance_segmentation_fast"][0].cpu()
        seg = seg.numpy()
        if seg.ndim == 3 and seg.shape[-1] >= 3:
            ids = (seg[..., 0].astype(np.int32)
                   + (seg[..., 1].astype(np.int32) << 8)
                   + (seg[..., 2].astype(np.int32) << 16))
        else:
            ids = seg.astype(np.int32)
        rng = np.random.default_rng(0)
        palette = rng.integers(60, 255, size=(int(ids.max()) + 2, 3),
                               dtype=np.uint8)
        palette[0] = 0  # background black
        vis = palette[np.clip(ids, 0, len(palette) - 1)].astype(np.uint8)
        imageio.imwrite(f"{args.outdir}/isaac_soccer_seg.png", vis[..., ::-1])
        n_inst = int((np.bincount(ids.flatten(), minlength=2)[1:] > 0).sum())
        print(f"[soccer] segmentation saved ({n_inst} non-bg instances)")
    except Exception as e:  # noqa: BLE001 — annotator availability varies
        print(f"[soccer] segmentation unavailable: {e}")

    # ground-truth task vectors the kick policy will consume (nominal spawns;
    # per-step ball state comes from a RigidObject asset in the task env)
    robot_pos = art.data.root_pos_w[0]
    print("[soccer] task vectors (nominal):")
    print(f"  ball_pos_w  = [{BALL_X:.3f}, 0.0, {BALL_R + 0.01:.3f}]")
    print(f"  robot_pos_w = {robot_pos.cpu().numpy().round(3)}")
    print(f"  goal_pos_w  = [{FIELD_L / 2:.3f}, 0.0, {GOAL_H / 2:.3f}]  "
          f"(goal {GOAL_W} x {GOAL_H} m)")
    print("[soccer] saved:")
    for f in ("isaac_soccer_field", "isaac_soccer_robot",
              "isaac_soccer_seg"):
        print("  ", os.path.join(args.outdir, f + ".png"))

    simulation_app.close()


if __name__ == "__main__":
    main()
