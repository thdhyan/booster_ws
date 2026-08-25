#!/usr/bin/env python3
"""K1 stereo head capture — ZED 2i replica in Isaac Sim.

Two pinhole cameras mounted on the Head_2 link (±60 mm baseline, matching the
ZED 2i's 120 mm), HD720 rectified FOV per Stereolabs table (H101°/V68° for the
2.1 mm lens). Captures left/right RGB, computes SGBM disparity, colorizes left
depth, and saves everything under docs/images/.

Pure-Isaac script (no ROS needed for capture).

Usage (venv-isaac python):
  python src/k1_sim_isaac/scripts/stereo_cam_test.py --steps 40
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--outdir", default="docs/images")
parser.add_argument("--baseline", type=float, default=0.120, help="ZED 2i baseline m")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from k1_velocity.tasks.velocity.velocity_env_cfg import K1_ARTICULATION_CFG  # noqa: E402

RES_W, RES_H = 1280, 720          # HD720 mode
HFOV_DEG = 101.0                  # ZED 2i 2.1mm @ HD720 (Stereolabs table)
OPTICAL_Q = (0.5, -0.5, 0.5, -0.5)  # wxyz, ROS-optical in head link frame (fwd-facing)


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = K1_ARTICULATION_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot_0",
        init_state=K1_ARTICULATION_CFG.init_state,
    )
    ground = AssetBaseCfg(prim_path="/World/defaultGround",
                          spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/domeLight",
                         spawn=sim_utils.DomeLightCfg(intensity=600.0))

    # ZED 2i replica: ±60 mm on Head_2, HD720 rectified FOV (2.1 mm lens)
    def __post_init__(self):
        f_px = RES_W / (2 * np.tan(np.deg2rad(HFOV_DEG) / 2))
        head = "{ENV_REGEX_NS}/Robot_0/Geometry/Trunk/Head_1/Head_2"
        common = dict(
            update_period=0.0,
            height=RES_H,
            width=RES_W,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=f_px * 0.1,          # mm (aperture in cm*10)
                horizontal_aperture=RES_W * 0.1,  # -> f_px preserved exactly
                clipping_range=(0.1, 20.0),
            ),
        )
        self.zed_left = CameraCfg(
            prim_path=f"{head}/zed_left",
            offset=CameraCfg.OffsetCfg(pos=(0.10, +args.baseline / 2, 0.0),
                                       rot=OPTICAL_Q),
            **common,
        )
        self.zed_right = CameraCfg(
            prim_path=f"{head}/zed_right",
            offset=CameraCfg.OffsetCfg(pos=(0.10, -args.baseline / 2, 0.0),
                                       rot=OPTICAL_Q),
            **common,
        )
        # obstacles so disparity/depth have structure
        for i, (x, z, c) in enumerate([(1.5, 0.25, (0.8, 0.2, 0.2, 1)),
                                       (2.8, 0.40, (0.2, 0.4, 0.85, 1)),
                                       (4.2, 0.18, (0.9, 0.7, 0.1, 1))]):
            setattr(self, f"box_{i}", AssetBaseCfg(
                prim_path=f"/World/box_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 2 * z),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c[:3]),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, 0.3 * (i - 1), z)),
            ))


def find_head_prim(stage, contains="Robot_0", name="Head_2"):
    """Return the rigid-body prim path for the head link (nested under Geometry/)."""
    from pxr import UsdPhysics  # noqa: E402

    candidates = []
    for p in stage.Traverse():
        s = str(p.GetPath())
        if p.GetName() == name and contains in s:
            candidates.append((p.HasAPI(UsdPhysics.RigidBodyAPI), s))
    if not candidates:
        return None
    candidates.sort(reverse=True)  # prefer RigidBodyAPI-bearing prim
    return candidates[0][1]


def main():
    import cv2  # post-app ok

    os.makedirs(args.outdir, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    scene.update(0.005)

    import omni.usd  # noqa: E402

    stage = omni.usd.get_context().get_stage()
    head_path = find_head_prim(stage)
    assert head_path, "Head_2 prim not found"
    print(f"[stereo] mounting on {head_path}")

    cam_left = scene["zed_left"]
    cam_right = scene["zed_right"]

    art = scene["robot"]

    # hold default pose while stepping (actuator PD keeps stance/head level)
    for i in range(args.steps):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(0.005)
        cam_left.update(0.005)
        cam_right.update(0.005)

    def rgb_np(cam):
        img = cam.data.output["rgb"][0].cpu().numpy()  # (H,W,4)
        return img[..., :3]

    left = rgb_np(cam_left)[..., ::-1].copy()   # to BGR
    right = rgb_np(cam_right)[..., ::-1].copy()

    import imageio.v2 as imageio

    imageio.imwrite(f"{args.outdir}/k1_stereo_left.png", left)
    imageio.imwrite(f"{args.outdir}/k1_stereo_right.png", right)

    # SGBM disparity (ZED-like block matching, CPU)
    N = 96  # odd disparity count
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=N, blockSize=7,
        P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2,
        uniquenessRatio=8, speckleWindowSize=80, speckleRange=2,
    )
    g_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    g_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    disp = sgbm.compute(g_l, g_r).astype(np.float32) / 16.0
    disp_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(disp, alpha=255.0 / max(N, 1)), cv2.COLORMAP_TURBO)
    valid = disp > 0
    disp_vis[~valid] = 0
    imageio.imwrite(f"{args.outdir}/k1_stereo_disparity.png", disp_vis)

    # sensor depth from left camera
    depth = cam_left.data.output["depth"][0, :, :, 0].cpu().numpy().astype(np.float32)
    d8 = np.clip(depth / 10.0, 0, 1)  # 10 m range visualized
    dcol = cv2.applyColorMap((d8 * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    imageio.imwrite(f"{args.outdir}/k1_stereo_depth.png", dcol)

    # side-by-side strip for README
    strip = np.hstack([left, right])
    imageio.imwrite(f"{args.outdir}/k1_stereo_pair.png", strip)

    f_px = RES_W / (2 * np.tan(np.deg2rad(HFOV_DEG) / 2))
    d_med = float(np.median(depth[valid.astype(bool) & (depth > 0)])) if valid.any() else float("nan")
    print("[stereo] saved:")
    for f in ("k1_stereo_left", "k1_stereo_right", "k1_stereo_pair",
              "k1_stereo_disparity", "k1_stereo_depth"):
        print("  ", os.path.join(args.outdir, f + ".png"))
    print(f"[stereo] f_px={f_px:.1f}  baseline={args.baseline:.3f} m  "
          f"median depth={d_med:.2f} m")

    simulation_app.close()


if __name__ == "__main__":
    main()
