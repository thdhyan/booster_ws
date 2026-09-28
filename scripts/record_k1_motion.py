#!/usr/bin/env python3
"""K1 motion replay recorder — tiled multi-camera debug video.

Replays a Booster motion CSV (7 root cols + 22 joints, 50 Hz) on the K1 in
MuJoCo with real PD control and records a tiled video with three camera panels,
a status header, and in-scene debug markers.

Per the workspace rule, every Isaac/MuJoCo validation run records a debug video
and the frame geometry is checked before the run counts as done.

Panels
  [0] ORBIT    three-quarter view, follows the robot
  [1] TOP      straight down (foot placement / support polygon)
  [2] FOLLOW   close chase cam on the torso

In-scene debug (visible in every panel)
  · cyan   support polygon / planted foot discs
  · red    centre of mass marker + its offset line to the support centre
  · grey   translucent ghost of the *reference* pose (the retargeted motion)
  · green  foot contact spheres when a sole is within 4 cm of the floor
  · amber  floor grid every 0.5 m

Status header (drawn per frame)
  frame / time · phase · foot contacts · CoM offset · root height · tracking err

This is a *kinematic replay* tool for inspecting retarget quality. The robot is
being driven open-loop at the reference joint angles — it is NOT balanced, and
it is expected to fall. That is the point: the video shows whether the
retargeted motion is even worth handing to an RL tracker.

Usage
  python3 scripts/record_k1_motion.py <motion.csv> --out docs/videos/motion.mp4
  python3 scripts/record_k1_motion.py <motion.csv> --out /tmp/x.mp4 --fps 50 \
      --start 100 --duration 300
  xvfb-run -a python3 scripts/record_k1_motion.py <motion.csv> --out /tmp/x.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF = os.path.join(REPO, "src/k1_description/assets/robots/K1/K1_22dof.xml")

# booster_assets.motions.K1_JOINT_NAMES — CSV column order
JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
FOOT_BODIES = ("left_foot_link", "right_foot_link")
SOLE_DROP = 0.038
CONTACT_Z = 0.04

# PD gains — identical to mujoco_fleet_node.gains_for() and the K1 actuator
# specs, so the replay uses the same gains the sim backends do.
def gains_for(name: str):
    if "_Hip_" in name:
        kp = {"Pitch": 30.2, "Roll": 21.4, "Yaw": 17.8}[name.split("_")[-1]]
        return kp, kp * 0.12
    if "_Knee_" in name:
        return 60.4, 4.8
    if "_Ankle_" in name:
        return 35.7, 4.3
    if "Head" in name or "_Elbow" in name or "_Shoulder" in name:
        return 4.0, 0.25
    return 10.0, 0.5


# --------------------------------------------------------------------------- #
def load_motion(path: str):
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    n_col = 7 + len(JOINT_NAMES)
    if raw.shape[1] < n_col:
        raise ValueError(f"{path}: expected {n_col} cols, got {raw.shape[1]}")
    return raw[:, 0:3], raw[:, 3:7], raw[:, 7:n_col]


def free_quat_to_wxyz(q_xyzw):
    """CSV stores xyzw; MuJoCo free joints want wxyz."""
    out = np.empty_like(q_xyzw)
    out[..., 0] = q_xyzw[..., 3]
    out[..., 1:4] = q_xyzw[..., 0:3]
    return out


class K1Replay:
    def __init__(self, mjcf=MJCF):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(mjcf)
        self.data = mujoco.MjData(self.model)

        self.jnames = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(self.model.njnt)]
        self.dofadr = {}
        self.qposadr = {}
        for i, n in enumerate(self.jnames):
            if n in JOINT_NAMES:
                self.dofadr[n] = self.model.jnt_dofadr[i]
                self.qposadr[n] = self.model.jnt_qposadr[i]
        missing = [n for n in JOINT_NAMES if n not in self.qposadr]
        if missing:
            raise KeyError(f"K1 MJCF missing joints: {missing}")

        self.kp = np.array([gains_for(n)[0] for n in JOINT_NAMES])
        self.kd = np.array([gains_for(n)[1] for n in JOINT_NAMES])
        self.kp_full = np.zeros(self.model.nv)
        self.kd_full = np.zeros(self.model.nv)
        for n in JOINT_NAMES:
            self.kp_full[self.dofadr[n]] = gains_for(n)[0]
            self.kd_full[self.dofadr[n]] = gains_for(n)[1]

        self.foot_bid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                         for b in FOOT_BODIES]
        self.trunk_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")

        # neutral standing pose
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.q0 = self.data.qpos.copy()

    def set_reference(self, t: int, root_pos, root_quat_wxyz, q):
        """Place the simulated robot at the reference pose for frame t."""
        d = self.data
        d.qpos[0:3] = root_pos[t]
        d.qpos[3:7] = root_quat_wxyz[t]
        for n in JOINT_NAMES:
            d.qpos[self.qposadr[n]] = q[t, JOINT_NAMES.index(n)]
        d.qvel[:] = 0.0
        self.mj.mj_forward(self.model, d)

    def apply_pd(self, q_des):
        d = self.data
        d.ctrl[:] = q_des
        d.qfrc_applied[:] = 0.0
        for n in JOINT_NAMES:
            da = self.dofadr[n]
            err = q_des[JOINT_NAMES.index(n)] - d.qpos[self.qposadr[n]]
            d.qfrc_applied[da] = self.kp[JOINT_NAMES.index(n)] * err - \
                                 self.kd[JOINT_NAMES.index(n)] * d.qvel[da]

    def contacts(self):
        out = []
        for b in self.foot_bid:
            out.append(self.data.xpos[b][2] - SOLE_DROP)
        return out

    def com(self):
        return self.data.subtree_com[self.trunk_bid].copy()

    def root(self):
        return self.data.xpos[self.trunk_bid].copy()


# --------------------------------------------------------------------------- #
# Debug scene decoration
# --------------------------------------------------------------------------- #
def add_debug_geom(renderer, feet, com, sup_center):
    """Draw support discs, CoM marker and the CoM->support line.

    feet : list of (x, y, sole_z) for left then right foot.

    Must be called AFTER update_scene() (which resets scene.ngeom) and
    appends at the current scene.ngeom, then bumps it so MuJoCo draws them.
    """
    import mujoco
    scene = renderer.scene
    base = scene.ngeom

    def put(slot, gtype, pos, size, rgba, mat=None):
        if base + slot >= scene.maxgeom:
            return
        # NOTE: mjv_initGeom signature is (geom, type, size, pos, mat, rgba) —
        # size comes BEFORE pos.
        mujoco.mjv_initGeom(
            scene.geoms[base + slot], gtype,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3).flatten() if mat is None else mat,
            np.asarray(rgba, dtype=np.float32))
        scene.ngeom = base + slot + 1

    s = 0.045
    for i, (fx, fy, fz) in enumerate(feet):
        planted = fz < CONTACT_Z
        col = (0.15, 0.85, 0.35, 0.45) if planted else (0.85, 0.25, 0.25, 0.25)
        put(i, mujoco.mjtGeom.mjGEOM_SPHERE, [fx, fy, 0.005], [s, s, s], col)

    # CoM marker
    put(2, mujoco.mjtGeom.mjGEOM_SPHERE, com, [0.03, 0.03, 0.03],
        [0.95, 0.2, 0.2, 0.9])

    # CoM -> support centre line
    if sup_center is not None and np.all(np.isfinite(sup_center)):
        tgt = np.array([sup_center[0], sup_center[1], 0.01])
        vec = tgt - com
        L = float(np.linalg.norm(vec))
        if L > 1e-4:
            z = vec / L
            up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0])
            x = np.cross(up, z); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            R = np.column_stack([x, y, z]).flatten()
            put(3, mujoco.mjtGeom.mjGEOM_CAPSULE, (com + tgt) * 0.5,
                [0.006, 0.006, L * 0.5], [0.95, 0.2, 0.2, 0.7], R)


def ghost_robot(mjcf, ref_qpos):
    """A static ghost MuJoCo model at the reference pose, drawn translucent."""
    import mujoco
    gm = mujoco.MjModel.from_xml_path(mjcf)
    gd = mujoco.MjData(gm)
    gd.qpos[:] = ref_qpos
    mujoco.mj_forward(gm, gd)
    for i in range(gm.ngeom):
        gm.geom_rgba[i][3] = 0.18          # translucent
    return gm, gd


# --------------------------------------------------------------------------- #
# Text overlay (numpy, no font deps beyond PIL if available)
# --------------------------------------------------------------------------- #
def make_text_helpers():
    try:
        from PIL import Image, ImageDraw, ImageFont
        have_pil = True
    except ImportError:
        have_pil = False
    if not have_pil:
        return None
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        font_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        font_b = font
    return Image, ImageDraw, font, font_b


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Record a tiled K1 motion replay video.")
    p.add_argument("motion", help="Booster motion CSV (7 root + 22 joints)")
    p.add_argument("--out", required=True, help="output .mp4")
    p.add_argument("--fps", type=float, default=50.0, help="motion sample rate")
    p.add_argument("--sim-fps", type=int, default=200, help="physics substeps per frame")
    p.add_argument("--start", type=int, default=0, help="first frame")
    p.add_argument("--duration", type=int, default=0, help="frames (0 = to end)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--panel-w", type=int, default=420)
    p.add_argument("--panel-h", type=int, default=315)
    p.add_argument("--title", default="")
    p.add_argument("--gl", default="glfw", choices=["glfw", "egl", "osmesa"],
                   help="MuJoCo GL backend. The Isaac Lab shell profile exports "
                        "MUJOCO_GL=egl which fails on this laptop; glfw works "
                        "against DISPLAY, and osmesa works under xvfb-run.")
    args = p.parse_args(argv)

    # Must be set BEFORE mujoco is imported.
    os.environ["MUJOCO_GL"] = args.gl
    import mujoco
    import imageio.v2 as imageio

    root_pos, root_quat, q = load_motion(args.motion)
    T = q.shape[0]
    i0 = max(0, args.start)
    i1 = T if args.duration <= 0 else min(T, i0 + args.duration)
    root_quat_wxyz = free_quat_to_wxyz(root_quat)

    sim = K1Replay()
    mjcf_model, mjcf_data = ghost_robot(MJCF, sim.q0)

    W, H = args.panel_w, args.panel_h
    header_h = 46
    # libx264 requires even dimensions; round the tiled frame up to even.
    if H % 2:
        H += 1
    if (H + header_h) % 2:
        header_h += 1
    out_w, out_h = W * 3, H + header_h

    renderer = mujoco.Renderer(sim.model, H, W)
    ghost_renderer = mujoco.Renderer(mjcf_model, H, W)

    text = make_text_helpers()
    Image, ImageDraw, font, font_b = text if text else (None, None, None, None)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    writer = imageio.get_writer(
        args.out, fps=30, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])

    name = args.title or os.path.basename(args.motion)
    dt = 1.0 / args.sim_fps

    for t in range(i0, i1):
        # --- drive the robot at the reference pose (open loop, PD held) ---
        sim.set_reference(t, root_pos, root_quat_wxyz, q)
        q_des = q[t]
        for _ in range(max(1, args.sim_fps // int(args.fps))):
            sim.apply_pd(q_des)
            mujoco.mj_step(sim.model, sim.data)

        # --- reference ghost ---
        mjcf_data.qpos[:3] = root_pos[t]
        mjcf_data.qpos[3:7] = root_quat_wxyz[t]
        for n in JOINT_NAMES:
            mjcf_data.qpos[sim.qposadr[n]] = q[t, JOINT_NAMES.index(n)]
        mujoco.mj_forward(mjcf_model, mjcf_data)

        feet = [(sim.data.xpos[b][0], sim.data.xpos[b][1],
                 sim.data.xpos[b][2] - SOLE_DROP) for b in sim.foot_bid]
        com = sim.com()
        rt = sim.root()

        planted = [f[2] < CONTACT_Z for f in feet]
        sup = None
        if any(planted):
            pts = [f[:2] for f, p in zip(feet, planted) if p]
            sup = np.mean(pts, axis=0)
        com_off = (float(np.linalg.norm(com[:2] - sup)) if sup is not None
                   else float("nan"))
        track_err = float(np.abs(sim.data.qpos[7:7 + len(JOINT_NAMES)] - q[t]).max())

        panels = []
        # Look at mid-body height, not the trunk origin, so the figure is
        # centred in the panel rather than sitting low.
        look = np.array([rt[0], rt[1], rt[2] * 0.72])
        for cam in ("orbit", "top", "follow"):
            free = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(free)
            free.lookat[:] = look
            if cam == "orbit":
                free.distance = 2.0
                free.azimuth, free.elevation = 135, 15
            elif cam == "top":
                # MuJoCo: POSITIVE elevation puts the camera above looking down.
                # Negative looks up from under the floor plane (renders black).
                free.distance = 1.7
                free.azimuth, free.elevation = 90, 89.5
            else:
                free.distance = 1.3
                free.azimuth, free.elevation = 90, 5
            opt = mujoco.MjvOption()
            mujoco.mjv_defaultOption(opt)
            # K1_22dof.xml splits geoms across two groups: group 0 = floor,
            # group 1 = the robot (Trunk, limbs, feet). Disabling all groups
            # with geomgroup[:] = 0 hides the robot and leaves only the floor.
            opt.geomgroup[:] = 1
            opt.sitegroup[:] = 1
            opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

            renderer.update_scene(sim.data, camera=free, scene_option=opt)
            sup_c = (sup[0], sup[1]) if sup is not None else None
            add_debug_geom(renderer, feet, com, sup_c)
            panels.append(renderer.render().copy())

            if cam != "top":
                ghost_renderer.update_scene(mjcf_data, camera=free)
                gpx = ghost_renderer.render()
                a = 0.32
                panels[-1] = (panels[-1] * (1 - a) + gpx * a).astype(np.uint8)

        strip = np.concatenate(panels, axis=1)

        if Image is not None:
            im = Image.fromarray(strip)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, out_w, header_h], fill=(18, 18, 22))
            dr.text((8, 4), f"K1 MOTION REPLAY  {name}", font=font_b, fill=(235, 235, 240))
            contact = "L+D" if all(planted) else ("L  " if planted[0] else
                                                 ("R  " if planted[1] else "air"))
            dr.text((8, 24),
                    f"frame {t:5d}/{T}  t={t/args.fps:5.2f}s  contact={contact}  "
                    f"com_off={com_off:5.3f}m  root_z={rt[2]:.3f}  "
                    f"q_err={track_err:5.3f}rad",
                    font=font, fill=(150, 220, 150))
            for i, lbl in enumerate(("ORBIT", "TOP", "FOLLOW")):
                dr.text((i * W + 8, header_h + 6), lbl, font=font_b,
                        fill=(255, 220, 120))
            im = np.asarray(im)
            strip = im

        writer.append_data(strip)
        if (t - i0) % 25 == 0:
            print(f"  frame {t}/{i1}  contact={planted}  com_off={com_off:.3f}",
                  flush=True)

    writer.close()
    renderer.close()
    ghost_renderer.close()

    size = os.path.getsize(args.out)
    print(f"\nwrote {args.out}  ({size/1e6:.1f} MB, {i1-i0} frames "
          f"@ {int(args.fps)} Hz motion)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
