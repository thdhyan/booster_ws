"""K1 carry (P7) MDP: a box welded between both forearm stubs, walked to a goal.

Geometry (all in the Trunk frame, from K1_22dof.urdf):

- The ``*_hand_link`` body is the FOREARM: its origin sits at the elbow-yaw
  joint and the stub runs 0.228 m along its local +-y. ``STUB_TIP`` is the
  contact point on that axis.
- ``HOLD_Q_LEFT`` is a width -> arm-pose table solved offline (least squares on
  the stub-tip position, forearm pointing forward, joints inside the 0.85 soft
  range). The right arm mirrors it through the sagittal plane: rotations about
  y keep their sign, about x/z flip (``MIRROR``).
- The weld frames are computed from FK of the INTERPOLATED pose, not from the
  table's targets, so the two FixedJoints are exactly satisfied at reset
  whatever residual the offline IK left.

The box has no collider (collision_enabled=False): it is welded, never touches
the ground in a live episode, and would otherwise self-collide with the arms it
sits between. Its mass/inertia come from an explicit MassAPI.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

LEFT_ARM_JOINTS = ["ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw"]
RIGHT_ARM_JOINTS = ["ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"]
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
LEFT_EE, RIGHT_EE = "left_hand_link", "right_hand_link"

# URDF joint origins (Trunk -> Arm_1 -> Arm_2 -> Arm_3 -> hand), all rpy = 0.
_LEFT_ORIGINS = ((0.0, 0.077, 0.1845), (0.0025, 0.068, -0.0135), (0.0, 0.044428, 0.0), (0.0, 0.1215, 0.0))
_AXES = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
MIRROR = (1.0, -1.0, 1.0, -1.0)
STUB_TIP = 0.20            # m along the forearm's +-y (mesh is 0.228 long)

# Offline IK: stub tip at (0.29, +w/2, 0.04), forearm along +x.
WIDTH_TABLE = (0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40)
HOLD_Q_LEFT = (
    (-0.542, -1.492, 0.194, -1.084),
    (-0.536, -1.492, 0.136, -1.084),
    (-0.532, -1.492, 0.079, -1.082),
    (-0.532, -1.458, 0.066, -1.081),
    (-0.533, -1.377, 0.113, -1.085),
    (-0.536, -1.296, 0.160, -1.090),
    (-0.539, -1.215, 0.205, -1.097),
)
BOX_DEPTH, BOX_HEIGHT = 0.18, 0.18
BOX_PRIM = "{ENV_REGEX_NS}/CarryBox"


# ---------------------------------------------------------------------------
# kinematics (numpy; USD-time + tests)
# ---------------------------------------------------------------------------
def _rotvec(axis, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def arm_fk(q, side: str) -> np.ndarray:
    """4x4 pose of ``{side}_hand_link`` in the Trunk frame for arm joints q (4,)."""
    sy = 1.0 if side == "left" else -1.0
    T = np.eye(4)
    for (ox, oy, oz), axis, qi in zip(_LEFT_ORIGINS, _AXES, q):
        A = np.eye(4)
        A[:3, 3] = (ox, sy * oy, oz)
        A[:3, :3] = _rotvec(axis, float(qi))
        T = T @ A
    return T


def hold_pose(width: float) -> tuple[np.ndarray, np.ndarray]:
    """(q_left, q_right) arm joints that put both stub tips on the box faces."""
    table = np.asarray(HOLD_Q_LEFT)
    w = float(np.clip(width, WIDTH_TABLE[0], WIDTH_TABLE[-1]))
    q_left = np.array([np.interp(w, WIDTH_TABLE, table[:, j]) for j in range(4)])
    return q_left, q_left * np.asarray(MIRROR)


def carry_geometry(width: float) -> dict:
    """Hold pose, box centre and the two weld frames for one box width (Trunk frame)."""
    q_left, q_right = hold_pose(width)
    T_l, T_r = arm_fk(q_left, "left"), arm_fk(q_right, "right")
    tip_l = (T_l @ np.array([0.0, STUB_TIP, 0.0, 1.0]))[:3]
    tip_r = (T_r @ np.array([0.0, -STUB_TIP, 0.0, 1.0]))[:3]
    # box centred between the tips laterally, its front face 2 cm past them
    center = np.array([0.5 * (tip_l[0] + tip_r[0]) + 0.02 - 0.5 * BOX_DEPTH, 0.0, 0.5 * (tip_l[2] + tip_r[2])])
    return {
        "q_left": q_left, "q_right": q_right, "center": center,
        "anchor_l": tip_l - center, "anchor_r": tip_r - center,   # weld point in the box frame
        "rot_l": T_l[:3, :3], "rot_r": T_r[:3, :3],              # hand orientation in the box frame
    }


def _mat_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    return w, x, y, z


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def _state(env) -> SimpleNamespace:
    st = getattr(env, "carry_state", None)
    if st is None:
        n, dev = env.num_envs, env.device
        geo = carry_geometry(0.325)
        st = SimpleNamespace(
            width=torch.full((n,), 0.325, device=dev),
            mass=torch.full((n,), 4.0, device=dev),
            hold_q=torch.tensor(np.concatenate([geo["q_left"], geo["q_right"]]), device=dev,
                                dtype=torch.float32).repeat(n, 1),
            box_offset=torch.tensor(geo["center"], device=dev, dtype=torch.float32).repeat(n, 1),
            anchors=torch.tensor(np.stack([geo["anchor_l"], geo["anchor_r"]]), device=dev,
                                 dtype=torch.float32).repeat(n, 1, 1),
            defaults_written=False,
        )
        env.carry_state = st
    return st


def _find_body_prim(stage, root_path: str, name: str) -> str:
    from pxr import Usd

    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.GetName() == name:
            return str(prim.GetPath())
    raise RuntimeError(f"[carry] no prim named {name!r} under {root_path}")


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def spawn_carry_box(env, env_ids, width_range: tuple[float, float], mass_range: tuple[float, float]) -> None:
    """prestartup: per-env box width + mass and the two wrist FixedJoints (USD).

    Width scales the box's ``geometry`` child, NOT the rigid-body root, so the weld
    frames authored on the root stay in metres. PhysX parses the joints once at
    play, which is why this runs before it, like push's randomize_box_geometry.
    """
    import isaaclab.sim as sim_utils
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    st = _state(env)
    stage = env.sim.stage
    paths = sim_utils.find_matching_prim_paths(env.scene["carry_box"].cfg.prim_path)
    rng = np.random.default_rng()
    widths, masses, holds, offsets, anchors = [], [], [], [], []
    for path in paths:
        w = float(rng.uniform(*width_range))
        m = float(rng.uniform(*mass_range))
        geo = carry_geometry(w)

        geom = UsdGeom.Xformable(stage.GetPrimAtPath(path + "/geometry"))
        scale = stage.GetPrimAtPath(path + "/geometry").GetAttribute("xformOp:scale")
        if scale and scale.IsValid():
            scale.Set(Gf.Vec3f(BOX_DEPTH, w, BOX_HEIGHT))
        else:
            geom.AddScaleOp().Set(Gf.Vec3f(BOX_DEPTH, w, BOX_HEIGHT))

        mass_api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path))
        mass_api.CreateMassAttr().Set(m)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(
            m / 12.0 * (w * w + BOX_HEIGHT ** 2),
            m / 12.0 * (BOX_DEPTH ** 2 + BOX_HEIGHT ** 2),
            m / 12.0 * (BOX_DEPTH ** 2 + w * w),
        ))

        robot_root = path.rsplit("/", 1)[0] + "/Robot"
        for side, ee, sign in (("l", LEFT_EE, 1.0), ("r", RIGHT_EE, -1.0)):
            joint = UsdPhysics.FixedJoint.Define(stage, f"{path}/weld_{side}")
            joint.CreateBody0Rel().SetTargets([Sdf.Path(path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(_find_body_prim(stage, robot_root, ee))])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in geo[f"anchor_{side}"]]))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(*_mat_to_quat_wxyz(geo[f"rot_{side}"])))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, sign * STUB_TIP, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateExcludeFromArticulationAttr().Set(True)

        widths.append(w)
        masses.append(m)
        holds.append(np.concatenate([geo["q_left"], geo["q_right"]]))
        offsets.append(geo["center"])
        anchors.append(np.stack([geo["anchor_l"], geo["anchor_r"]]))

    dev = env.device
    st.width = torch.tensor(widths, device=dev, dtype=torch.float32)
    st.mass = torch.tensor(masses, device=dev, dtype=torch.float32)
    st.hold_q = torch.tensor(np.stack(holds), device=dev, dtype=torch.float32)
    st.box_offset = torch.tensor(np.stack(offsets), device=dev, dtype=torch.float32)
    st.anchors = torch.tensor(np.stack(anchors), device=dev, dtype=torch.float32)
    print(f"[carry] box DR: width {st.width.min():.2f}-{st.width.max():.2f} m, "
          f"mass {st.mass.min():.1f}-{st.mass.max():.1f} kg, {len(paths)} boxes welded to both forearms")


def reset_carry(env, env_ids: torch.Tensor) -> None:
    """reset: arms to the per-env hold pose, box placed on the welds.

    Runs AFTER reset_scene_to_default / reset_robot_joints (field order = event
    order), so the robot root is the default root + env origin -- recomputed here
    rather than read back, because asset data only refreshes on the next sim step.
    """
    from isaaclab.utils.math import quat_apply

    st = _state(env)
    robot, box = env.scene["robot"], env.scene["carry_box"]
    arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
    hold = st.hold_q[env_ids]

    if not st.defaults_written:
        # per-env defaults: joint_pos_rel / joint_deviation / the arm action offset
        # are then all relative to the hold pose, not the arms-down URDF default
        robot.data.default_joint_pos.torch[:, arm_ids] = st.hold_q
        term = env.action_manager.get_term("arm_pos")
        term._offset[:] = st.hold_q
        st.defaults_written = True

    robot.write_joint_state_to_sim_index(
        position=hold, velocity=torch.zeros_like(hold), joint_ids=arm_ids, env_ids=env_ids
    )
    robot.set_joint_position_target_index(target=hold, joint_ids=arm_ids, env_ids=env_ids)

    root = robot.data.default_root_pose.torch[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    pose = root.clone()
    pose[:, :3] = root[:, :3] + quat_apply(root[:, 3:7], st.box_offset[env_ids])
    box.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    box.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(env_ids), 6), device=env.device),
                                         env_ids=env_ids)


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
def goal_command_obs(env, command_name: str = "base_velocity") -> torch.Tensor:
    """(n,3) goal in the base frame (x, y) and its distance, clipped to 5 m."""
    term = env.command_manager.get_term(command_name)
    return torch.cat([term.goal_pos_b, term.goal_dist[:, None]], dim=-1).clamp(-5.0, 5.0)


def box_params_obs(env) -> torch.Tensor:
    """(n,2) privileged box width (m) and mass (/10 kg)."""
    st = _state(env)
    return torch.stack([st.width, 0.1 * st.mass], dim=-1)


# ---------------------------------------------------------------------------
# rewards / terminations
# ---------------------------------------------------------------------------
def goal_distance_tanh(env, std: float, command_name: str = "base_velocity") -> torch.Tensor:
    return 1.0 - torch.tanh(env.command_manager.get_term(command_name).goal_dist / std)


def goal_reached(env, command_name: str = "base_velocity") -> torch.Tensor:
    """1 on the step the dwell completes (the command then resamples the goal)."""
    return env.command_manager.get_term(command_name).just_reached.float()


def box_tilt_l2(env) -> torch.Tensor:
    """Squared xy of gravity in the box frame (0 when the box is level)."""
    from isaaclab.utils.math import quat_apply_inverse

    q = env.scene["carry_box"].data.root_quat_w.torch
    g = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(q.shape[0], 3)
    return quat_apply_inverse(q, g)[:, :2].square().sum(-1)


def weld_error(env) -> torch.Tensor:
    """(n,) worst distance between a stub tip and its weld point on the box (m).

    The FixedJoints should hold this near 0; a growing value means the solver is
    losing the loop (box too heavy for the arm gains) or the weld frames are wrong.
    """
    from isaaclab.utils.math import quat_apply

    st = _state(env)
    robot, box = env.scene["robot"], env.scene["carry_box"]
    names = [n.split("/")[-1] for n in robot.body_names]
    errs = []
    for k, (ee, sign) in enumerate(((LEFT_EE, 1.0), (RIGHT_EE, -1.0))):
        i = names.index(ee)
        tip_local = torch.tensor([0.0, sign * STUB_TIP, 0.0], device=env.device).expand(env.num_envs, 3)
        tip = robot.data.body_pos_w.torch[:, i] + quat_apply(robot.data.body_quat_w.torch[:, i], tip_local)
        anchor = box.data.root_pos_w.torch + quat_apply(box.data.root_quat_w.torch, st.anchors[:, k])
        errs.append((tip - anchor).norm(dim=-1))
    return torch.maximum(errs[0], errs[1])


def weld_error_l1(env) -> torch.Tensor:
    return weld_error(env)


def weld_broken(env, threshold: float) -> torch.Tensor:
    return weld_error(env) > threshold
