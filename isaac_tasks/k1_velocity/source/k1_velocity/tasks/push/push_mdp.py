"""MDP terms for the K1 box-push task (hierarchical: frozen locomotion base).

Architecture (user decision 2026-09-24):
  action 9 = [velocity override (3) | left wrist EE delta (3) | right wrist EE delta (3)]
  - the velocity slice drives the FROZEN Run-11 partial-control policy
    (TorchScript) whose 68-dim obs is assembled here in the exact partial
    term order; its 12 leg + 2 head joint targets are written to the
    articulation (the new model never touches the legs),
  - each wrist slice is a position delta consumed by a
    DifferentialInverseKinematicsAction term with EE body = the hand link
    (left_hand_link / right_hand_link - palm colliders, contact pushing).
  - the env's `wrist_target` command (2x3 contact points on the box's near
    face) is the task reference: obs + tracking/proximity rewards; the policy
    emits wrist deltas relative to its current EE pose (stock relative-mode IK).

Box state: 1 m prototype cube; per-env scale 0.7-1.5x, mass 3-25 kg,
friction 0.3-1.2 are USD-cooked so they are sampled at STARTUP (event mode
"usd", replicate_physics=False) and read back per prim into push_state.

Goals: rigid pose that walks away from the box - goal_offset is a cumulative
integrator advanced along the sampled push heading (capped by the curriculum
distance), and goal corners = corners(goal pose, CURRENT half-extents), so
corner rewards + the corner centroid ("cumulative sum of corners") are the
primary signals.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

try:  # Isaac Lab 3.0-EA layout (dl); isaac-lab image renamed this package
    import isaaclab_tasks.core.velocity.mdp as vmdp
except (ImportError, ModuleNotFoundError):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vmdp

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
K1_LEG_JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
K1_HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]
K1_LEFT_ARM_JOINTS = ["ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw"]
K1_RIGHT_ARM_JOINTS = ["ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"]
LEFT_EE = "left_hand_link"
RIGHT_EE = "right_hand_link"

PROTO_HALF = 0.5          # 1 m prototype cube
CORNER_SIGNS = torch.tensor(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=torch.float32,
)                          # (8, 3)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def _state(env: ManagerBasedRLEnv) -> SimpleNamespace:
    st = getattr(env, "push_state", None)
    if st is None:
        n, dev = env.num_envs, env.device
        st = SimpleNamespace(
            half_extents=torch.full((n, 3), PROTO_HALF, device=dev),
            mass=torch.full((n,), 8.0, device=dev),
            goal_offset=torch.zeros((n, 3), device=dev),
            push_dir=torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(n, 1),
            goal_rate=0.06,          # m/s of goal advance (capped)
            goal_dist_max=0.3,      # curriculum-owned (m)
            last_vel_cmd=torch.zeros((n, 3), device=dev),
            prev_goal_dist=torch.zeros(n, device=dev),
        )
        env.push_state = st
    return st


# ---------------------------------------------------------------------------
# USD-time DR: per-env size + mass (read back after the built-ins write them)
# ---------------------------------------------------------------------------
def randomize_box_geometry(env: ManagerBasedRLEnv, env_ids, scale_range, mass_range) -> None:
    """Event mode='usd': scale + mass per env, fixed for the run (PhysX parses
    USD once at startup). Values are read back per prim into push_state."""
    from isaaclab.envs.mdp import events as il_events
    from pxr import Usd, UsdGeom, UsdPhysics

    st = _state(env)
    box = env.scene["box"]
    asset_cfg = SceneEntityCfg("box")
    asset_cfg.resolve(env.scene)
    env_ids = torch.arange(env.num_envs, device=env.device)

    il_events.randomize_rigid_body_scale(env, env_ids, scale_range, asset_cfg)
    il_events.randomize_rigid_body_mass(
        env, env_ids, asset_cfg,
        mass_distribution_params=mass_range,
        distribution="uniform",
        operation="replace",
    )

    scales, masses = [], []
    for path in box.prim_paths:
        prim = Usd.PrimAtPath(path)
        s = UsdGeom.Xformable(prim).GetAttribute("xformOp:scale").Get()
        scales.append(torch.tensor([float(s[0])] * 3, device=env.device))
        masses.append(torch.tensor(float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()), device=env.device))
    st.half_extents = PROTO_HALF * torch.stack(scales)
    st.mass = torch.stack(masses)
    print(f"[push] box DR: edge {2 * st.half_extents[:, 0].min():.2f}-{2 * st.half_extents[:, 0].max():.2f} m, "
          f"mass {st.mass.min():.1f}-{st.mass.max():.1f} kg")


def apply_box_green_alpha(env: ManagerBasedRLEnv) -> None:
    """Startup: semi-transparent green PBR material on the box (pxr binding)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    box = env.scene["box"]
    for path in box.prim_paths:
        prim = Usd.PrimAtPath(path)
        mat = UsdShade.Material.Define(Usd.PrimAtPath(path + "/green_alpha"))
        shader = UsdShade.Shader.Define(Usd.PrimAtPath(path + "/green_alpha/surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.8, 0.15))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _corners_from_pose(pos: torch.Tensor, quat: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """(n,8,3) world corners of an oriented box."""
    from isaaclab.utils.math import quat_apply

    signs = CORNER_SIGNS.to(pos.device)
    pts = signs[None] * half[:, None, :]                 # (n,8,3)
    return quat_apply(quat, pts) + pos[:, None, :]


def box_corners_world(env: ManagerBasedRLEnv) -> torch.Tensor:
    st = _state(env)
    box = env.scene["box"]
    return _corners_from_pose(box.data.root_pos_w.torch, box.data.root_quat_w.torch, st.half_extents)


def _to_base(env: ManagerBasedRLEnv, pts_w: torch.Tensor) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse

    robot = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_quat_w.torch, pts_w - robot.data.root_pos_w.torch[:, None, :])


def box_corners_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n,8,3) current box corners in the robot base frame."""
    return _to_base(env, box_corners_world(env))


def goal_corners_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n,8,3) goal corners = box pose + cumulative goal offset, same yaw/size."""
    st = _state(env)
    box = env.scene["box"]
    pos = box.data.root_pos_w.torch + st.goal_offset
    return _to_base(env, _corners_from_pose(pos, box.data.root_quat_w.torch, st.half_extents))


def wrist_positions_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n,2,3) left/right wrist (hand-link) positions in the base frame."""
    from isaaclab.utils.math import quat_apply_inverse

    robot = env.scene["robot"]
    names = [n.split("/")[-1] for n in robot.body_names]
    idx = [names.index(LEFT_EE), names.index(RIGHT_EE)]
    pos_w = robot.data.body_pos_w[:, idx, :]
    return quat_apply_inverse(
        robot.data.root_quat_w.torch[:, None], pos_w - robot.data.root_pos_w.torch[:, None, :]
    )[:, 0]


# ---------------------------------------------------------------------------
# reset + goal integrator
# ---------------------------------------------------------------------------
def _write_box_pose(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pos: torch.Tensor, yaw: torch.Tensor) -> None:
    box = env.scene["box"]
    root = box.data.default_root_state.torch[env_ids].clone()
    root[:, :3] = pos + env.scene.env_origins[env_ids]
    cy, sy = torch.cos(yaw / 2.0), torch.sin(yaw / 2.0)
    z0 = torch.zeros_like(cy)
    root[:, 3:7] = torch.stack([z0, z0, sy, cy], dim=-1)  # xyzw
    root[:, 7:] = 0.0
    box.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    box.write_root_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)


def reset_box(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Box 0.9-1.4 m ahead, yaw +-30 deg; goal integrator re-zeroed."""
    st = _state(env)
    n, dev = len(env_ids), env.device
    dist = 0.9 + 0.5 * torch.rand(n, device=dev)
    lat = 0.3 * (2.0 * torch.rand(n, device=dev) - 1.0)
    yaw = (torch.rand(n, device=dev) - 0.5) * math.radians(60.0)
    pos = torch.zeros((n, 3), device=dev)
    pos[:, 0], pos[:, 1] = dist, lat
    _write_box_pose(env, env_ids, pos, yaw)
    st.push_dir[env_ids] = torch.tensor([1.0, 0.0, 0.0], device=dev)
    st.goal_offset[env_ids] = 0.0
    st.prev_goal_dist[env_ids] = dist


def advance_goal(env: ManagerBasedRLEnv, env_ids: torch.Tensor, dt: float = 0.25) -> None:
    """Interval event: the cumulative goal walks away along the push heading."""
    st = _state(env)
    st.goal_offset[env_ids] += st.push_dir[env_ids] * st.goal_rate * dt
    n = st.goal_offset[env_ids].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    over = n > st.goal_dist_max
    st.goal_offset[env_ids] = torch.where(over, st.goal_offset[env_ids] / n * st.goal_dist_max, st.goal_offset[env_ids])


def goal_dist_curriculum(env: ManagerBasedRLEnv, env_ids, start_iter: int = 200,
                         end_iter: int = 1800, d_max: float = 1.5,
                         steps_per_iter: int = 24) -> float:
    """Cumulative goal walks 0.3 -> d_max metres (iterations via physics steps)."""
    st = _state(env)
    iteration = (env.sim.get_physics_step_count() // env.cfg.decimation) / steps_per_iter
    span = max(end_iter - start_iter, 1)
    frac = min(max((iteration - start_iter) / span, 0.0), 1.0)
    st.goal_dist_max = 0.3 + (d_max - 0.3) * frac
    return st.goal_dist_max


# ---------------------------------------------------------------------------
# wrist target command: 2 contact points on the box's near face (base frame)
# ---------------------------------------------------------------------------
class WristTargetCommand(CommandTerm):
    """Holds the 2x3 wrist target command; sampled by the reset event (needs
    the freshly placed box pose)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._pos = torch.zeros((env.num_envs, 6), device=env.device)
        self.metrics = {}

    @property
    def command(self) -> torch.Tensor:
        return self._pos

    def set_values(self, env_ids: torch.Tensor, pos6: torch.Tensor) -> None:
        self._pos[env_ids] = pos6

    def _resample_command(self, env_ids) -> None:
        pass  # the reset event fills the buffer (box-dependent); no auto-resample

    def _update_command(self, env_ids) -> None:
        pass

    def _update_metrics(self) -> None:
        pass

    def _set_debug_vis_impl(self, env_ids) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass


@configclass
class WristTargetCommandCfg(CommandTermCfg):
    """2x3 wrist target command buffer (filled by the reset event; never
    auto-resamples - the reset event owns sampling)."""

    class_type = WristTargetCommand
    resampling_time_range = (1.0e9, 1.0e9)


def reset_wrist_targets(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Writes 2x3 contact targets (box near face, mid height) into the command."""
    cmd_term = env.command_manager.get_term("wrist_target")
    if cmd_term is None:
        return
    st = _state(env)
    box = env.scene["box"]
    pos = box.data.root_pos_w.torch[env_ids]
    half = st.half_extents[env_ids]
    n = len(env_ids)
    near_x = pos[:, 0:1] - half[:, 0:1]                     # face toward the robot
    y = pos[:, 1:2] + torch.stack([-0.35 * half[:, 1], 0.35 * half[:, 1]], dim=-1)
    z = pos[:, 2:3] + 0.55 * half[:, 2:3]
    tgt_w = torch.cat([near_x.expand(n, 2), y, z], dim=-1)  # (n,2,3) world
    tgt = _to_base(env, tgt_w)
    tgt[..., 0] = tgt[..., 0].clamp(0.30, 0.75)
    tgt[..., 1] = tgt[..., 1].clamp(-0.45, 0.45)
    tgt[..., 2] = tgt[..., 2].clamp(0.35, 1.15)
    cmd_term.set_values(env_ids, tgt.reshape(n, 6))


# ---------------------------------------------------------------------------
# observations (teacher group = fully observable by design)
# ---------------------------------------------------------------------------
def push_box_teacher(env: ManagerBasedRLEnv) -> torch.Tensor:
    """mass 1 | half-extents 3 | corners 24 | vel 6 | goal pose 7 | goal offset 3 | goal corners 24 = 68."""
    from isaaclab.utils.math import quat_apply_inverse

    st = _state(env)
    robot = env.scene["robot"]
    box = env.scene["box"]
    corners_bf = box_corners_base(env).flatten(1)                            # (n,24)
    goal_corners_bf = goal_corners_base(env).flatten(1)                     # (n,24)
    vel_bf = quat_apply_inverse(robot.data.root_quat_w.torch, box.data.root_lin_vel_w.torch - robot.data.root_lin_vel_w.torch)
    ang_bf = quat_apply_inverse(robot.data.root_quat_w.torch, box.data.root_ang_vel_w.torch)
    goal_pos_bf = quat_apply_inverse(
        robot.data.root_quat_w.torch, (box.data.root_pos_w.torch + st.goal_offset) - robot.data.root_pos_w.torch
    )
    goal_quat_bf = quat_apply_inverse(robot.data.root_quat_w.torch[:, None], box.data.root_quat_w.torch[:, None])[:, 0]
    goal_offset_bf = quat_apply_inverse(robot.data.root_quat_w.torch, st.goal_offset)
    return torch.cat(
        [st.mass[:, None] / 25.0, st.half_extents, corners_bf, vel_bf, ang_bf,
         goal_pos_bf, goal_quat_bf, goal_offset_bf, goal_corners_bf], dim=-1
    )


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------
def corner_goal_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """-mean_i ||corner_i - goal_corner_i||, normalized by box size."""
    st = _state(env)
    d = (box_corners_base(env) - goal_corners_base(env)).norm(dim=-1).mean(-1, keepdim=True)
    return (-d / st.half_extents.mean(-1, keepdim=True).clamp(min=0.2)).squeeze(-1)


def centroid_goal_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """-||mean(current corners) - mean(goal corners)|| (cumulative corner sum)."""
    st = _state(env)
    d = (box_corners_base(env).mean(1) - goal_corners_base(env).mean(1)).norm(dim=-1, keepdim=True)
    return (-d / st.half_extents.mean(-1, keepdim=True).clamp(min=0.2)).squeeze(-1)


def box_goal_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    st = _state(env)
    cur = (box_corners_base(env).mean(1) - goal_corners_base(env).mean(1)).norm(dim=-1)
    prog = (st.prev_goal_dist - cur) / st.half_extents.mean(-1).clamp(min=0.2)
    st.prev_goal_dist = cur
    return prog


def box_vel_toward_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse

    st = _state(env)
    robot = env.scene["robot"]
    box = env.scene["box"]
    v = quat_apply_inverse(robot.data.root_quat_w.torch, box.data.root_lin_vel_w.torch - robot.data.root_lin_vel_w.torch)
    return (v * st.push_dir).sum(-1)


def box_spin_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    w = env.scene["box"].data.root_ang_vel_w.torch
    return -w[:, :2].norm(dim=-1)


def wrist_target_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """-mean ||wrist_pos_base - commanded target|| (both wrists)."""
    pos_bf = wrist_positions_base(env)
    tgt = env.command_manager.get_term("wrist_target").command.reshape(-1, 2, 3)
    return -(pos_bf - tgt).norm(dim=-1).mean(-1)


def wrist_box_proximity(env: ManagerBasedRLEnv, scale: float = 0.08) -> torch.Tensor:
    """exp(-mean wrist-to-box-surface gap / scale): contact-ready shaping."""
    st = _state(env)
    wrist_bf = wrist_positions_base(env)                     # (n,2,3)
    corners_bf = box_corners_base(env)                        # (n,8,3)
    center = corners_bf.mean(1, keepdim=True)
    half = st.half_extents[:, None, :]
    wrist_local = wrist_bf[:, None, :] - center               # (n,1,3) vs (n,8,3) AABB
    clamped = torch.maximum(torch.minimum(wrist_local.expand_as(corners_bf - center), half), -half)
    gap = (wrist_local.expand_as(corners_bf - center) - clamped).norm(dim=-1).min(-1).values
    return torch.exp(-gap.mean(-1) / scale)


def track_cmd_lin_vel_exp(env: ManagerBasedRLEnv, std: float = 0.5) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse

    st = _state(env)
    robot = env.scene["robot"]
    v = quat_apply_inverse(robot.data.root_quat_w.torch, robot.data.root_lin_vel_w.torch)
    return torch.exp(-((v[:, :2] - st.last_vel_cmd[:, :2]).norm(dim=-1) ** 2) / (std * std))


def track_cmd_ang_vel_exp(env: ManagerBasedRLEnv, std: float = 0.5) -> torch.Tensor:
    st = _state(env)
    w = env.scene["robot"].data.root_ang_vel_w.torch
    return torch.exp(-((w[:, 2] - st.last_vel_cmd[:, 2]) ** 2) / (std * std))


# ---------------------------------------------------------------------------
# frozen-base velocity action: velocity override -> frozen partial policy -> legs
# ---------------------------------------------------------------------------
def _action_term_base():
    for mod, name in (
        ("isaaclab.envs.mdp.actions.action_term", "ActionTerm"),
        ("isaaclab.managers.action_manager", "ActionTerm"),
        ("isaaclab.envs.mdp.actions", "ActionTerm"),
    ):
        try:
            return getattr(__import__(mod, fromlist=[name]), name)
        except (ImportError, AttributeError):
            continue
    raise ImportError("ActionTerm base class not found")


ActionTerm = _action_term_base()


class FrozenBaseVelocityAction(ActionTerm):
    """3-dim action = velocity command override for the FROZEN base policy.

    The frozen policy is the Run-11 partial-control actor exported to
    TorchScript (flat 68-dim obs -> 14 joint-target offsets). The obs is
    assembled here in the exact partial PolicyCfg term order (no noise):
      lin_vel 3 | ang_vel 3 | gravity 3 | cmd 3 | leg_pos 12 | leg_vel 12 |
      head_pos 2 | arm_pos 8 | arm_vel 8 | last_base 14
    Outputs: leg targets (scale 0.25) + head targets (scale 0.5), written to
    the articulation. The arms are untouched here (the IK terms own them).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._asset = env.scene[cfg.asset_name]
        self._policy = torch.jit.load(cfg.base_policy_path, map_location=self._asset.device)
        self._policy.eval()
        dev = self._asset.device
        # find_joints returns python lists in this Isaac Lab build -> tensors here
        leg_ids, _ = self._asset.find_joints(K1_LEG_JOINTS, preserve_order=True)
        head_ids, _ = self._asset.find_joints(K1_HEAD_JOINTS, preserve_order=True)
        arm_l_ids, _ = self._asset.find_joints(K1_LEFT_ARM_JOINTS, preserve_order=True)
        arm_r_ids, _ = self._asset.find_joints(K1_RIGHT_ARM_JOINTS, preserve_order=True)
        self._leg_ids = torch.as_tensor(leg_ids, dtype=torch.long, device=dev)
        self._head_ids = torch.as_tensor(head_ids, dtype=torch.long, device=dev)
        self._arm_l_ids = torch.as_tensor(arm_l_ids, dtype=torch.long, device=dev)
        self._arm_r_ids = torch.as_tensor(arm_r_ids, dtype=torch.long, device=dev)
        self._ids14 = torch.cat([self._leg_ids, self._head_ids], dim=0)
        self._arm_ids = torch.cat([self._arm_l_ids, self._arm_r_ids], dim=0)
        self._default14 = self._asset.data.default_joint_pos[:, self._ids14].clone()
        self._last = torch.zeros((env.num_envs, 14), device=self._asset.device)
        self._vel = torch.zeros((env.num_envs, 3), device=self._asset.device)
        self._raw_actions = torch.zeros((env.num_envs, 3), device=self._asset.device)
        self._processed_actions = torch.zeros((env.num_envs, 3), device=self._asset.device)
        print(f"[push] frozen base loaded: {cfg.base_policy_path}")

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        st = _state(self._env)
        self._raw_actions[:] = actions
        vel = actions[:, :3].clamp(-0.8, 0.8)
        self._processed_actions[:] = vel
        self._vel = vel
        st.last_vel_cmd = vel
        robot = self._asset
        obs = torch.cat(
            [
                vmdp.base_lin_vel(self._env),
                vmdp.base_ang_vel(self._env),
                vmdp.projected_gravity(self._env),
                vel,
                robot.data.joint_pos[:, self._leg_ids] - robot.data.default_joint_pos[:, self._leg_ids],
                robot.data.joint_vel[:, self._leg_ids],
                robot.data.joint_pos[:, self._head_ids] - robot.data.default_joint_pos[:, self._head_ids],
                robot.data.joint_pos[:, self._arm_ids] - robot.data.default_joint_pos[:, self._arm_ids],
                robot.data.joint_vel[:, self._arm_ids],
                self._last,
            ],
            dim=-1,
        )
        with torch.inference_mode():
            out = self._policy(obs.to(self._asset.device))
        self._last = out
        targets = self._default14.clone()
        targets[:, :12] += 0.25 * out[:, :12]
        targets[:, 12:] += 0.5 * out[:, 12:14]
        for name in ("set_joint_position_target", "write_joint_position_target_to_sim"):
            fn = getattr(robot, name, None)
            if fn is not None:
                fn(targets, joint_ids=self._ids14)
                break

    def apply_actions(self) -> None:
        pass  # targets already in the articulation buffer

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._last.zero_()
        else:
            self._last[env_ids] = 0.0


@configclass
class FrozenBaseVelocityActionCfg(ActionTermCfg):
    """Velocity override for the frozen partial-control base policy."""

    class_type = FrozenBaseVelocityAction
    asset_name: str = "robot"
    base_policy_path: str = "models/k1_partialctrl_base.pt"


# ---------------------------------------------------------------------------
# cfg helpers re-exported for the env cfg
# ---------------------------------------------------------------------------
__all__ = [
    "K1_LEG_JOINTS", "K1_HEAD_JOINTS", "K1_LEFT_ARM_JOINTS", "K1_RIGHT_ARM_JOINTS",
    "LEFT_EE", "RIGHT_EE", "DifferentialIKControllerCfg", "DifferentialInverseKinematicsActionCfg",
    "WristTargetCommandCfg", "WristTargetCommand", "FrozenBaseVelocityAction", "FrozenBaseVelocityActionCfg",
    "randomize_box_geometry", "apply_box_green_alpha", "reset_box", "reset_wrist_targets", "advance_goal",
    "goal_dist_curriculum", "push_box_teacher", "corner_goal_tracking", "centroid_goal_tracking",
    "box_goal_progress", "box_vel_toward_goal", "box_spin_penalty", "wrist_target_tracking",
    "wrist_box_proximity", "track_cmd_lin_vel_exp", "track_cmd_ang_vel_exp",
    "vmdp",
]
