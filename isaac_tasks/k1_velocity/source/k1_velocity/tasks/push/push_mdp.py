"""MDP terms for the K1 box-push task (hierarchical: frozen locomotion base).

v3 (TRACK B P6, 2026-09-28) rewrites the goal machinery per the Phase-0
degenerate audit (TRAINING.md "P6 Phase-0 degenerate audit", findings A1-A15).

Architecture (v3 / blocker 2a - push freezes the NEW squat base teacher):
  action 10 = [vx, vy, wz, H* | left wrist EE delta (3) | right wrist EE delta (3)]
  - the first 4-dim slice drives the FROZEN squat-capable base teacher
    (velocity teacher exported to TorchScript). Its obs is assembled here in
    the EXACT squat TeacherCfg term order, noise-free (236 dims):
        lin_vel 3 | ang_vel 3 | gravity 3 | cmd 4 | leg_pos 12 |
        leg_vel 12 | last action 12 | height_scan 187 (clip +-1)
    and its 12 leg joint targets are written as default + clip(out, +-1)
    (scale 1.0 - identical to the velocity ActionsCfg the teacher trained
    in). H* (slice index 3) maps [-1,1] -> [0.40, 0.55] m into `st.h_cmd`,
    the reference for the command-relative height termination/reward.
    v2's velocity override becomes 4-dim, so the wrist slices moved to
    indices 4..9 (cmd slice 3 -> 4, blocker 2a).
  - legacy mode (base path names "partialctrl", or PUSH_BASE_MODE=partial)
    keeps the v2 partial-control base: 3-dim cmd, 68-dim obs, 14 targets
    (legs scale 0.25, head 0.5) - used for pre-Phase-1 A/B smokes only.
  - each wrist slice is a position delta consumed by a
    DifferentialInverseKinematicsAction term with EE body = the hand link
    (left_hand_link / right_hand_link - palm colliders, contact pushing).
  - the env's `wrist_target` command (2x3 contact points on the box's near
    face) is the task reference: obs + tracking/proximity rewards; the policy
    emits wrist deltas relative to its current EE pose (stock relative-mode IK).

Box state: 1 m prototype cube; per-env scale 0.7-1.5x, mass 3-25 kg,
friction 0.3-1.2 are USD-cooked so they are sampled at STARTUP (event mode
"prestartup", replicate_physics=False) and read back per prim into push_state.

Goals (v3, fixes A1-A3): a FIXED randomized pose sampled at reset:
  - box spawns in an annulus r in [0.9, 1.4] m with FULL direction + FULL yaw;
  - goal = box + dir*d, d in [0.3, curriculum cap 0.3 -> 1.5 m]; first
    feasible of 4 sampled directions keeps |goal_xy| <= GOAL_WORKSPACE_R
    (1.5 m, inside the 2.5 m env-cell footprint), else straight inward
    (always feasible: |r - d| <= 1.1 < 1.5);
  - goal yaw = box spawn yaw + delta, |delta| <= st.yaw_cap (curriculum
    +-30 deg -> +-180 deg, phased AFTER the distance curriculum), so goal
    corners carry BOTH translation and orientation signal;
  - goal corners come from the GOAL pose (own quat + current half-extents),
    never the current box pose - v2 anchored them to the box, so pushing the
    box moved the goal too and the heaviest reward term cancelled itself
    (finding A1).

Terminations (v3): time_out | command-relative floor (H* - 0.05, A12) |
bad_orientation | goal_reached (mean corner err < 0.08 m held 1 s, A10/A14)
| box OOB beyond OOB_RADIUS (A11) | box tip > 45 deg (A13). The -200
failure penalty sources from failure_terminated, which EXCLUDES success and
time_out (A9/A10 - vmdp.is_terminated only excluded time_out).
"""
from __future__ import annotations

import math
import os
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

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# height_scan lives in isaaclab.envs.mdp; the velocity mdp package usually
# re-exports it, but resolve it once here so the frozen obs never depends on
# package-level re-export differences between the dl image and the zz-bw SIF.
if hasattr(vmdp, "height_scan"):
    _height_scan = vmdp.height_scan
else:  # pragma: no cover - depends on image package layout
    from isaaclab.envs.mdp.observations import height_scan as _height_scan

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

# v3 task bounds (Phase-0 audit A10/A11/A13 + design decisions 2026-09-28)
GOAL_WORKSPACE_R = 1.5    # max |goal_xy| from env origin (env_spacing 2.5)
OOB_RADIUS = 1.9          # box centre beyond this from env origin = failure
SUCCESS_ERR_M = 0.08      # mean corner-error tolerance for success
SUCCESS_HOLD_S = 1.0      # seconds the error must stay under tolerance
H_CMD_MIN, H_CMD_MAX = 0.40, 0.55   # H* range the squat teacher trained on
H_CMD_DEFAULT = 0.55      # standing height until the first action lands


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def _state(env: ManagerBasedRLEnv) -> SimpleNamespace:
    st = getattr(env, "push_state", None)
    if st is None:
        n, dev = env.num_envs, env.device
        ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
        st = SimpleNamespace(
            half_extents=torch.full((n, 3), PROTO_HALF, device=dev),
            mass=torch.full((n,), 8.0, device=dev),
            # fixed-per-episode goal pose (world; replaces v2's goal_offset)
            goal_pos=torch.zeros((n, 3), device=dev),
            goal_quat=ident.repeat(n, 1).clone(),
            goal_dist_max=0.3,             # curriculum-owned (m)
            yaw_cap=math.radians(30.0),    # curriculum-owned (rad)
            # fresh spawn pose of the CURRENT episode (world), read by the
            # wrist reset - box.data may still hold the previous pose until
            # the next physics fetch
            spawn_pos=torch.zeros((n, 3), device=dev),
            spawn_quat=ident.repeat(n, 1).clone(),
            h_cmd=torch.full((n,), H_CMD_DEFAULT, device=dev),
            success_hold=torch.zeros(n, device=dev),
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
    USD once at startup). Values are read back per prim into push_state.

    Scale goes through IL's randomizer (it creates xformOp:scale with the right
    xformOpOrder); the mass is written straight to USD MassAPI because this
    build's randomize_rigid_body_mass is a class term (EventTermCfg protocol)
    whose operations exclude 'replace'.
    """
    import random as pyrandom

    from isaaclab.envs.mdp import events as il_events
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    st = _state(env)
    box = env.scene["box"]
    asset_cfg = SceneEntityCfg("box")
    asset_cfg.resolve(env.scene)
    env_ids = torch.arange(env.num_envs, device=env.device)

    il_events.randomize_rigid_body_scale(env, env_ids, scale_range, asset_cfg)

    scales, masses = [], []
    stage = env.sim.stage
    for path in sim_utils.find_matching_prim_paths(box.cfg.prim_path):
        prim = stage.GetPrimAtPath(path)
        s = prim.GetAttribute("xformOp:scale").Get()
        s = s if s is not None else (1.0, 1.0, 1.0)
        scales.append(torch.tensor([float(s[0])] * 3, device=env.device))
        # uniform mass in mass_range; inertia scaled with it (I ~ m for a cube)
        m = pyrandom.uniform(*mass_range)
        api = UsdPhysics.MassAPI.Apply(prim)
        old = float(api.GetMassAttr().Get() or 8.0)
        api.GetMassAttr().Set(m)
        diag = api.GetDiagonalInertiaAttr()
        d = diag.Get()
        if d is not None and len(d) == 3 and old > 0.0:
            diag.Set(Gf.Vec3f(*[float(v) * (m / old) for v in d]))
        masses.append(torch.tensor(m, device=env.device))
    st.half_extents = PROTO_HALF * torch.stack(scales)
    st.mass = torch.stack(masses)
    print(f"[push] box DR: edge {2 * st.half_extents[:, 0].min():.2f}-{2 * st.half_extents[:, 0].max():.2f} m, "
          f"mass {st.mass.min():.1f}-{st.mass.max():.1f} kg")


def apply_box_green_alpha(env: ManagerBasedRLEnv, env_ids: torch.Tensor = None) -> None:
    """Startup: semi-transparent green PBR material on the box (pxr binding).

    Event terms are invoked as func(env, env_ids, **params) in this Isaac Lab
    build, hence the (unused) env_ids argument.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    box = env.scene["box"]
    stage = env.sim.stage
    for path in sim_utils.find_matching_prim_paths(box.cfg.prim_path):
        prim = stage.GetPrimAtPath(path)
        stage = prim.GetStage()
        mat = UsdShade.Material.Define(stage, path + "/green_alpha")
        shader = UsdShade.Shader.Define(stage, path + "/green_alpha/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.8, 0.15))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _rot_batch(quat: torch.Tensor, pts: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate pts (n, ..., 3) by quats (n, 4) [xyzw]; shape-preserving.

    isaaclab's quat_apply/quat_apply_inverse reshape BOTH operands flat
    ((-1,4)/(-1,3)), so an (n,8,3) point batch desyncs from an (n,4) quat
    ("size of tensor a (4) ... b (32)" crash). Expand the quat to one row
    per point first; verified against per-env reference rotations.
    """
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    shape = pts.shape
    k = pts.numel() // (shape[0] * 3)
    fn = quat_apply_inverse if inverse else quat_apply
    return fn(quat.reshape(-1, 4).repeat_interleave(k, dim=0), pts.reshape(-1, 3)).view(shape)


def _corners_from_pose(pos: torch.Tensor, quat: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """(n,8,3) world corners of an oriented box."""
    signs = CORNER_SIGNS.to(pos.device)
    pts = signs[None] * half[:, None, :]                 # (n,8,3)
    return _rot_batch(quat, pts) + pos[:, None, :]


def box_corners_world(env: ManagerBasedRLEnv) -> torch.Tensor:
    st = _state(env)
    box = env.scene["box"]
    return _corners_from_pose(box.data.root_pos_w.torch, box.data.root_quat_w.torch, st.half_extents)


def _to_base(env: ManagerBasedRLEnv, pts_w: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """(...,3) world points -> robot base frame (handles n points per env).

    Pass ``env_ids`` when ``pts_w`` rows are a subset of envs (reset events);
    otherwise the full-batch robot state would be broadcast against it.
    """
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w.torch
    quat = robot.data.root_quat_w.torch
    if env_ids is not None:
        pos = pos[env_ids]
        quat = quat[env_ids]
    return _rot_batch(quat, pts_w - pos[:, None, :], inverse=True)


def box_corners_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n,8,3) current box corners in the robot base frame."""
    return _to_base(env, box_corners_world(env))


def goal_corners_world(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n,8,3) FIXED goal-pose corners (own quat, current half-extents)."""
    st = _state(env)
    return _corners_from_pose(st.goal_pos, st.goal_quat, st.half_extents)


def goal_corners_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n,8,3) goal corners in the robot base frame (fixed goal pose).

    v3 fix A1: v2 computed them as CURRENT box pose + goal_offset, so every
    corner term measured only the (env-owned) offset - pushing the box moved
    the goal along and the signal cancelled.
    """
    return _to_base(env, goal_corners_world(env))


def wrist_positions_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n,2,3) left/right wrist (hand-link) positions in the base frame."""
    robot = env.scene["robot"]
    names = [n.split("/")[-1] for n in robot.body_names]
    idx = [names.index(LEFT_EE), names.index(RIGHT_EE)]
    pos_w = robot.data.body_pos_w[:, idx, :]
    return _rot_batch(
        robot.data.root_quat_w.torch, pos_w - robot.data.root_pos_w.torch[:, None, :], inverse=True
    )


# ---------------------------------------------------------------------------
# reset + goal sampling (v3: fixed pose, no integrator)
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
    """v3: annulus spawn (full direction + full yaw) + FIXED episode goal.

    Fixes A1/A2 (goal was an env-owned integrator advanced under the box) and
    enables the orientation task (A3: goal yaw = spawn yaw +- st.yaw_cap).
    """
    st = _state(env)
    n, dev = len(env_ids), env.device
    # --- box spawn: annulus r in [0.9, 1.4], full circle, full yaw
    r = 0.9 + 0.5 * torch.rand(n, device=dev)
    phi = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi
    box_yaw = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi
    pos = torch.zeros((n, 3), device=dev)
    pos[:, 0] = r * torch.cos(phi)
    pos[:, 1] = r * torch.sin(phi)
    pos[:, 2] = st.half_extents[env_ids, 2]              # rest on the ground
    _write_box_pose(env, env_ids, pos, box_yaw)
    origins = env.scene.env_origins[env_ids]
    st.spawn_pos[env_ids] = pos + origins
    st.spawn_quat[env_ids] = torch.stack(
        [torch.zeros_like(box_yaw), torch.zeros_like(box_yaw),
         torch.sin(box_yaw / 2.0), torch.cos(box_yaw / 2.0)], dim=-1
    )
    # --- goal pose: fixed for the whole episode
    cap = st.goal_dist_max
    d = 0.3 + (cap - 0.3) * torch.rand(n, device=dev)    # [0.3, cap]
    k_dir = 4
    psi = (torch.rand(n, k_dir, device=dev) * 2.0 - 1.0) * math.pi
    gx = pos[:, 0:1] + d[:, None] * torch.cos(psi)       # (n,K)
    gy = pos[:, 1:2] + d[:, None] * torch.sin(psi)
    feas = (gx * gx + gy * gy).sqrt() <= GOAL_WORKSPACE_R
    idx = torch.where(
        feas.any(dim=1),
        feas.float().argmax(dim=1),                      # first feasible dir
        torch.zeros(n, dtype=torch.long, device=dev),
    )
    # fallback (no feasible direction): straight INWARD. |r - d| <= 1.1 m for
    # r in [0.9,1.4], d in [0.3,1.5] -> always inside the workspace.
    r_safe = r.clamp(min=1e-6)
    fb_x = pos[:, 0] - d * (pos[:, 0] / r_safe)
    fb_y = pos[:, 1] - d * (pos[:, 1] / r_safe)
    cand = torch.stack([gx, gy], dim=-1)                 # (n,K,2)
    picked = cand[torch.arange(n, device=dev), idx]      # (n,2)
    none = ~feas.any(dim=1)
    gx_f = torch.where(none, fb_x, picked[:, 0])
    gy_f = torch.where(none, fb_y, picked[:, 1])
    goal_yaw = box_yaw + (torch.rand(n, device=dev) * 2.0 - 1.0) * st.yaw_cap
    st.goal_pos[env_ids] = torch.stack(
        [gx_f, gy_f, st.half_extents[env_ids, 2]], dim=-1
    ) + origins
    st.goal_quat[env_ids] = torch.stack(
        [torch.zeros_like(goal_yaw), torch.zeros_like(goal_yaw),
         torch.sin(goal_yaw / 2.0), torch.cos(goal_yaw / 2.0)], dim=-1
    )
    # first-step baseline for box_goal_progress in its OWN metric:
    # corner-centroid distance == centre-to-centre distance (rotation-
    # invariant) == d. v2 seeded this with the robot->box x-distance (A5).
    st.prev_goal_dist[env_ids] = d
    st.success_hold[env_ids] = 0.0


def goal_dist_curriculum(env: ManagerBasedRLEnv, env_ids, start_iter: int = 200,
                         end_iter: int = 1800, d_max: float = 1.5,
                         steps_per_iter: int = 24) -> float:
    """Goal distance cap 0.3 -> d_max metres (iterations via physics steps)."""
    st = _state(env)
    iteration = (env.sim.get_physics_step_count() // env.cfg.decimation) / steps_per_iter
    span = max(end_iter - start_iter, 1)
    frac = min(max((iteration - start_iter) / span, 0.0), 1.0)
    st.goal_dist_max = 0.3 + (d_max - 0.3) * frac
    return st.goal_dist_max


def goal_yaw_curriculum(env: ManagerBasedRLEnv, env_ids, start_iter: int = 1800,
                        end_iter: int = 3000, yaw0_deg: float = 30.0,
                        yaw_max_deg: float = 180.0,
                        steps_per_iter: int = 24) -> float:
    """Phase 2 of the goal curriculum (A3): goal yaw delta +-30 -> +-180 deg,
    phased AFTER the distance walk (start_iter = its end) so orientation
    tracking enters only once distance pushing works. The goal's own yaw is
    what makes box rotation task-relevant (v2's spin penalty punished ALL
    rotation while goal yaw was pinned to box yaw - no orientation task).
    """
    st = _state(env)
    iteration = (env.sim.get_physics_step_count() // env.cfg.decimation) / steps_per_iter
    span = max(end_iter - start_iter, 1)
    frac = min(max((iteration - start_iter) / span, 0.0), 1.0)
    st.yaw_cap = math.radians(yaw0_deg + (yaw_max_deg - yaw0_deg) * frac)
    return math.degrees(st.yaw_cap)


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

    def _update_command(self) -> None:
        pass  # NOTE: no env_ids in this Isaac Lab build (see command_manager)

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
    """Writes 2x3 contact targets on the box's near face at mid height.

    v3 (annulus spawn): the near face is derived from the FRESH spawn pose
    stored by reset_box (box.data may still hold the previous episode's pose
    until the next physics fetch), for ANY spawn direction/yaw - v2 hard-coded
    the +x face, which points away from the robot for half the new spawns.
    """
    cmd_term = env.command_manager.get_term("wrist_target")
    if cmd_term is None:
        return
    st = _state(env)
    n = len(env_ids)
    pos = st.spawn_pos[env_ids]                       # (n,3) world
    quat = st.spawn_quat[env_ids]                     # (n,4) xyzw
    half = st.half_extents[env_ids]                   # (n,3)
    robot_pos = env.scene["robot"].data.root_pos_w.torch[env_ids]
    # box -> robot direction expressed in the BOX frame (horizontal)
    rel_box = _rot_batch(quat, robot_pos - pos, inverse=True)   # (n,3)
    dir_b = rel_box[:, :2]
    dir_b = dir_b / dir_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # distance from centre to the first face along dir_b (unit vector):
    # t = min(hx/|dx|, hy/|dy|)
    hx, hy, hz = half[:, 0], half[:, 1], half[:, 2]
    tx = torch.where(dir_b[:, 0:1].abs() > 1e-6,
                     hx.unsqueeze(1) / dir_b[:, 0:1].abs().clamp(min=1e-6),
                     torch.full((n, 1), float("inf"), device=dir_b.device))
    ty = torch.where(dir_b[:, 1:2].abs() > 1e-6,
                     hy.unsqueeze(1) / dir_b[:, 1:2].abs().clamp(min=1e-6),
                     torch.full((n, 1), float("inf"), device=dir_b.device))
    t = torch.minimum(tx, ty).squeeze(-1)                                  # (n,)
    face = torch.cat([dir_b * t[:, None], torch.zeros_like(t)[:, None]], dim=-1)  # (n,3)
    # two contact points +-0.35 half-extent along the horizontal tangent
    tang = torch.cat([-dir_b[:, 1:2], dir_b[:, 0:1]], dim=-1)              # (n,2)
    off = 0.35 * tang * torch.cat([hx[:, None], hy[:, None]], dim=-1)      # (n,2)
    p1 = face.clone()
    p1[:, 0:2] += off
    p2 = face.clone()
    p2[:, 0:2] -= off
    zt = 0.55 * hz                       # mid height above the box centre
    p1[:, 2] = zt
    p2[:, 2] = zt
    tgt_w = torch.stack([pos + _rot_batch(quat, p1), pos + _rot_batch(quat, p2)], dim=1)  # (n,2,3)
    # NOTE: pass the (n,2,3) batch directly - _to_base subtracts pos[:, None, :]
    # per env; flattening to (2n,3) first would broadcast against the (n,1,3)
    # robot state and produce (n,2n,3).
    tgt = _to_base(env, tgt_w, env_ids=env_ids)                                # (n,2,3)
    # reachability clamps in the robot base frame (unchanged from v2)
    tgt[..., 0] = tgt[..., 0].clamp(0.30, 0.75)
    tgt[..., 1] = tgt[..., 1].clamp(-0.45, 0.45)
    tgt[..., 2] = tgt[..., 2].clamp(0.35, 1.15)
    cmd_term.set_values(env_ids, tgt.reshape(n, 6))


# ---------------------------------------------------------------------------
# observations (teacher group = fully observable by design)
# ---------------------------------------------------------------------------
def push_box_teacher(env: ManagerBasedRLEnv) -> torch.Tensor:
    """mass 1 | half-extents 3 | corners 24 | vel 6 | goal pose 7 |
    goal-box delta 3 | goal corners 24 = 68 (width unchanged from v2; the
    delta replaces v2's goal_offset, which no longer exists)."""
    from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

    st = _state(env)
    robot = env.scene["robot"]
    box = env.scene["box"]
    corners_bf = box_corners_base(env).flatten(1)                            # (n,24)
    goal_corners_bf = goal_corners_base(env).flatten(1)                      # (n,24)
    vel_bf = quat_apply_inverse(robot.data.root_quat_w.torch, box.data.root_lin_vel_w.torch - robot.data.root_lin_vel_w.torch)
    ang_bf = quat_apply_inverse(robot.data.root_quat_w.torch, box.data.root_ang_vel_w.torch)
    goal_pos_bf = quat_apply_inverse(
        robot.data.root_quat_w.torch, st.goal_pos - robot.data.root_pos_w.torch
    )
    goal_quat_bf = quat_mul(quat_inv(robot.data.root_quat_w.torch), st.goal_quat)
    goal_box_delta_bf = quat_apply_inverse(
        robot.data.root_quat_w.torch, st.goal_pos - box.data.root_pos_w.torch
    )
    return torch.cat(
        [st.mass[:, None] / 25.0, st.half_extents, corners_bf, vel_bf, ang_bf,
         goal_pos_bf, goal_quat_bf, goal_box_delta_bf, goal_corners_bf], dim=-1
    )


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------
def corner_goal_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """-mean_i ||corner_i - goal_corner_i||, normalized by box size.

    v3 (A1): against the FIXED goal pose this is fully policy-controllable -
    the heaviest term finally teaches box motion.
    """
    st = _state(env)
    d = (box_corners_base(env) - goal_corners_base(env)).norm(dim=-1).mean(-1, keepdim=True)
    return (-d / st.half_extents.mean(-1, keepdim=True).clamp(min=0.2)).squeeze(-1)


def centroid_goal_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """-||mean(current corners) - mean(goal corners)|| (corner centroid)."""
    st = _state(env)
    d = (box_corners_base(env).mean(1) - goal_corners_base(env).mean(1)).norm(dim=-1, keepdim=True)
    return (-d / st.half_extents.mean(-1, keepdim=True).clamp(min=0.2)).squeeze(-1)


def box_goal_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Telescoping Delta(corner-centroid error); prev baseline is reset in
    the SAME metric/frame by reset_box (fixes A5)."""
    st = _state(env)
    cur = (box_corners_base(env).mean(1) - goal_corners_base(env).mean(1)).norm(dim=-1)
    prog = (st.prev_goal_dist - cur) / st.half_extents.mean(-1).clamp(min=0.2)
    st.prev_goal_dist = cur
    return prog


def box_vel_toward_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World-frame box velocity projected on the world direction
    (goal - box), recomputed every step (fix A4: v2 dotted a base-frame
    velocity with a world [1,0,0] that never updated - misdirected as soon
    as the robot yaws)."""
    st = _state(env)
    box = env.scene["box"]
    dir_w = st.goal_pos - box.data.root_pos_w.torch
    dir_w = dir_w / dir_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (box.data.root_lin_vel_w.torch * dir_w).sum(-1)


def box_spin_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Kept for A/B only - v3 sets the weight to None (A3): rotation is now
    REQUIRED when goal yaw differs from box yaw; corners track it."""
    w = env.scene["box"].data.root_ang_vel_w.torch
    return -w[:, :2].norm(dim=-1)


def wrist_target_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """-mean ||wrist_pos_base - commanded target|| (both wrists)."""
    pos_bf = wrist_positions_base(env)
    tgt = env.command_manager.get_term("wrist_target").command.reshape(-1, 2, 3)
    return -(pos_bf - tgt).norm(dim=-1).mean(-1)


def wrist_box_proximity(env: ManagerBasedRLEnv, coarse: float = 0.30, fine: float = 0.05) -> torch.Tensor:
    """two-scale exp(-mean wrist-to-box gap / scale), summed (0..2).

    Fix A7: v2's single 0.08 m scale had ~zero gradient beyond ~8 cm (no
    coarse approach pull); the 0.30 m term pulls the wrist across open space
    and the 0.05 m term keeps contact shaping sharp. Gap is distance from
    each wrist to the box surface (0 inside), computed in the BOX frame so
    the clamp uses the true half-extents regardless of yaw.
    """
    from isaaclab.utils.math import quat_inv, quat_mul

    st = _state(env)
    robot = env.scene["robot"]
    wrist_bf = wrist_positions_base(env)                     # (n,2,3)
    corners_bf = box_corners_base(env)                       # (n,8,3)
    center = corners_bf.mean(1, keepdim=True)                # (n,1,3)
    half = st.half_extents[:, None, :]                       # (n,1,3)
    # box orientation expressed in the robot base frame -> wrist in box frame
    box_q_bf = quat_mul(quat_inv(robot.data.root_quat_w.torch), env.scene["box"].data.root_quat_w.torch)
    wrist_box = _rot_batch(box_q_bf, wrist_bf - center, inverse=True)               # (n,2,3)
    inside = torch.maximum(torch.minimum(wrist_box, half), -half)
    gap = (wrist_box - inside).norm(dim=-1)                  # (n,2)
    g = gap.mean(-1)
    return torch.exp(-g / coarse) + torch.exp(-g / fine)


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


def base_height_command(env: ManagerBasedRLEnv, std: float = 0.05) -> torch.Tensor:
    """exp(-|z - H*| / std): trunk height tracks the CURRENT command.

    Fix A12/E1: v2 rewarded a fixed 0.57 target, which fights a squat-capable
    base; H* (st.h_cmd) is written each step by the frozen action's 4th
    slice (or stays at H_CMD_DEFAULT in legacy partial mode).
    """
    st = _state(env)
    z = env.scene["robot"].data.root_pos_w.torch[:, 2]
    return torch.exp(-torch.abs(z - st.h_cmd) / std)


def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse +1 exactly once per success.

    goal_reached (a termination term) updates st.success_hold FIRST in the
    control step - this Isaac Lab build computes terminations BEFORE
    rewards - and the env resets right after rewards, so the bonus lands on
    the firing step only. Fixes A10/A14: success was previously unmeasurable
    and unpaid.
    """
    st = _state(env)
    return (st.success_hold >= SUCCESS_HOLD_S).float()


def failure_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 on non-timeout terminations EXCLUDING goal_reached: the -200
    penalty must not charge success (fix A9/A10 - vmdp.is_terminated only
    excluded time_out; adding a success term would have made success net
    -150). Per-term dones are refreshed by the termination manager before
    rewards, so this reads the current step."""
    tm = env.termination_manager
    fail = tm.terminated.clone()
    fail &= ~tm.get_term("goal_reached").to(torch.bool)
    return fail.float()


# ---------------------------------------------------------------------------
# terminations (v3)
# ---------------------------------------------------------------------------
def goal_reached(env: ManagerBasedRLEnv, err_thresh: float = SUCCESS_ERR_M,
                 hold_s: float = SUCCESS_HOLD_S) -> torch.Tensor:
    """Success: mean corner error < err_thresh held for hold_s seconds.

    UPDATES st.success_hold here (termination manager runs before rewards in
    this build, so success_bonus sees the same-step value), then signals the
    episode end. Episode_Termination/goal_reached becomes the success-rate
    metric (fix A10).
    """
    st = _state(env)
    err = (box_corners_world(env) - goal_corners_world(env)).norm(dim=-1).mean(-1)
    within = err < err_thresh
    st.success_hold = torch.where(
        within, st.success_hold + env.step_dt, torch.zeros_like(st.success_hold)
    )
    return st.success_hold >= hold_s


def height_below_command(env: ManagerBasedRLEnv, margin: float = 0.05) -> torch.Tensor:
    """Trunk below H* - margin = failure (fix A12: v2's fixed 0.35 floor is
    meaningless once the policy commands squats at 0.40-0.55). Flat ground ->
    world z equals terrain-relative z."""
    st = _state(env)
    z = env.scene["robot"].data.root_pos_w.torch[:, 2]
    return z < (st.h_cmd - margin)


def box_out_of_bounds(env: ManagerBasedRLEnv, radius: float = OOB_RADIUS) -> torch.Tensor:
    """Box centre beyond `radius` from the env origin = failure (fix A11:
    v2 parked the box anywhere - no bounds term, no failure count). radius >
    GOAL_WORKSPACE_R keeps every legal goal inside bounds, with 0.4 m of
    overshoot margin before the failure fires."""
    box = env.scene["box"]
    xy = box.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return xy.norm(dim=-1) > radius


def box_tipped(env: ManagerBasedRLEnv, limit_deg: float = 45.0) -> torch.Tensor:
    """Box roll/pitch past limit_deg = failure (fix A13: v2 only had a spin
    penalty - a tipped box made the corner goal unreachable but the episode
    kept running). Box up-axis z < cos(limit) -> tipped."""
    from isaaclab.utils.math import quat_apply

    q = env.scene["box"].data.root_quat_w.torch
    up = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).repeat(q.shape[0], 1))
    return up[:, 2] < math.cos(math.radians(limit_deg))


# ---------------------------------------------------------------------------
# frozen-base velocity action: command slice -> frozen squat/partial policy -> legs
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
    """Command slice for the FROZEN base policy (v3 / blocker 2a).

    Modes (auto-derived from the resolved policy path basename unless
    PUSH_BASE_MODE=squat|partial overrides):

    squat (default, models/k1_push_base.pt -> exported squat teacher)
      action slice = 4 [vx, vy, wz, H*]; obs assembled in the EXACT squat
      TeacherCfg order, noise-free (236 dims): lin 3 | ang 3 | grav 3 |
      cmd 4 | leg_pos rel 12 | leg_vel 12 | last action 12 | height_scan 187
      (clip +-1). Output = 12 leg targets, target = default + clip(out, +-1)
      (velocity ActionsCfg: scale 1.0, clip +-1). vx/vy/wz clamp to the
      VR-parity training ranges +-0.5/+-0.3/+-0.8; H* maps [-1,1] ->
      [0.40, 0.55] m into st.h_cmd. Arms/head untouched (IK terms own the
      arms; the velocity teacher never actuated the head).

    partial (legacy v2, models/k1_partialctrl_base.pt)
      action slice = 3; 68-dim partial PolicyCfg obs (lin 3 | ang 3 | grav 3
      | cmd 3 | leg_pos 12 | leg_vel 12 | head_pos 2 | arm_pos 8 | arm_vel 8
      | last_base 14) -> 14 targets (legs scale 0.25, head 0.5). Kept for
      A/B smokes before the squat export exists.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._asset = env.scene[cfg.asset_name]
        path = cfg.base_policy_path
        mode = os.environ.get("PUSH_BASE_MODE", "").lower()
        if mode not in ("squat", "partial"):
            mode = "partial" if "partialctrl" in os.path.basename(os.path.realpath(path)) else "squat"
        self._mode = mode
        self._policy = torch.jit.load(path, map_location=self._asset.device)
        self._policy.eval()
        # diag switches (env vars; default = normal operation)
        #   PUSH_FROZEN_MODE=hold  -> write default leg targets, skip policy
        #                             (isolates env/physics from the frozen path)
        #   PUSH_FROZEN_DEBUG=1    -> print obs blocks + policy output (env 0)
        self._diag_mode = os.environ.get("PUSH_FROZEN_MODE", "")
        self._diag_debug = os.environ.get("PUSH_FROZEN_DEBUG", "") == "1"
        self._dbg_n = 0
        dev = self._asset.device
        # find_joints returns python lists in this Isaac Lab build -> tensors here.
        #
        # preserve_order=False = ARTICULATION joint order, mirroring BOTH
        # consumers in the envs the frozen policies trained in:
        #   JointActionCfg.preserve_order=False  (velocity joint_pos / partial terms)
        #   SceneEntityCfg.preserve_order=False  (obs joint_pos/vel terms)
        # The K1_*_JOINTS lists are left-then-right, but the articulation
        # interleaves L/R per joint type (legs resolve [3,4,8,9,12,13,16,17,
        # 18,19,20,21] = LHP,RHP,LHR,RHR,..., arms [1,2,6,7,10,11,14,15]).
        # With preserve_order=True this term wired 11/12 leg dims to the WRONG
        # joints (hip-roll cmds onto knees etc.) -> the frozen base tipped and
        # fell in ~17 steps (pass-B done_rate 0.043, immune to friction/reset/
        # arm-pose A/Bs; home stood on the identical first output).
        leg_ids, _ = self._asset.find_joints(K1_LEG_JOINTS, preserve_order=False)
        self._leg_ids = torch.as_tensor(leg_ids, dtype=torch.long, device=dev)
        self._default12 = self._asset.data.default_joint_pos[:, self._leg_ids].clone()
        if self._mode == "partial":
            head_ids, _ = self._asset.find_joints(K1_HEAD_JOINTS, preserve_order=False)
            # One resolution over all 8 arms: two per-side finds concatenated
            # would still be L-block-then-R-block instead of the interleaved
            # order the arm_pos/arm_vel obs blocks had during training.
            arm_ids, _ = self._asset.find_joints(
                K1_LEFT_ARM_JOINTS + K1_RIGHT_ARM_JOINTS, preserve_order=False)
            self._head_ids = torch.as_tensor(head_ids, dtype=torch.long, device=dev)
            self._arm_ids = torch.as_tensor(arm_ids, dtype=torch.long, device=dev)
            self._ids14 = torch.cat([self._leg_ids, self._head_ids], dim=0)
            self._default14 = self._asset.data.default_joint_pos[:, self._ids14].clone()
            self._cmd_dim, self._out_dim = 3, 14
        else:
            self._cmd_dim, self._out_dim = 4, 12
        self._last = torch.zeros((env.num_envs, self._out_dim), device=dev)
        self._vel = torch.zeros((env.num_envs, 3), device=dev)
        self._raw_actions = torch.zeros((env.num_envs, self._cmd_dim), device=dev)
        self._processed_actions = torch.zeros((env.num_envs, self._cmd_dim), device=dev)
        print(f"[push] frozen base loaded: {path} mode={self._mode} "
              f"cmd={self._cmd_dim} out={self._out_dim}")

    @property
    def action_dim(self) -> int:
        return self._cmd_dim

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._mode == "partial":
            self._process_partial(actions)
        else:
            self._process_squat(actions)

    def _process_squat(self, actions: torch.Tensor) -> None:
        """Blocker 2a: 4-dim command -> 236-dim teacher obs -> 12 leg targets."""
        st = _state(self._env)
        robot = self._asset
        raw = actions[:, :4]
        self._raw_actions[:] = raw
        vel = torch.stack(
            [raw[:, 0].clamp(-0.5, 0.5),
             raw[:, 1].clamp(-0.3, 0.3),
             raw[:, 2].clamp(-0.8, 0.8)], dim=-1
        )                                               # VR-parity ranges
        h = H_CMD_MIN + (H_CMD_MAX - H_CMD_MIN) * 0.5 * (raw[:, 3].clamp(-1.0, 1.0) + 1.0)
        self._vel = vel
        st.last_vel_cmd = vel
        st.h_cmd = h
        cmd4 = torch.cat([vel, h[:, None]], dim=-1)
        self._processed_actions[:] = cmd4
        lin = vmdp.base_lin_vel(self._env)
        ang = vmdp.base_ang_vel(self._env)
        grav = vmdp.projected_gravity(self._env)
        leg_pos = robot.data.joint_pos[:, self._leg_ids] - robot.data.default_joint_pos[:, self._leg_ids]
        # joint_vel_rel == raw: default_joint_vel is all-zero in this build
        leg_vel = robot.data.joint_vel[:, self._leg_ids]
        scan = _height_scan(self._env, sensor_cfg=SceneEntityCfg("height_scanner")).clamp(-1.0, 1.0)
        obs = torch.cat([lin, ang, grav, cmd4, leg_pos, leg_vel, self._last, scan], dim=-1)
        out = None
        if self._diag_mode != "hold":
            with torch.inference_mode():
                out = self._policy(obs.to(robot.device))
            self._last = out
        if self._diag_debug and self._dbg_n < 5:
            for name, blk in (
                ("lin_vel", lin), ("ang_vel", ang), ("gravity", grav), ("cmd4", cmd4),
                ("leg_pos", leg_pos), ("leg_vel", leg_vel), ("last_act", self._last),
                ("height_scan", scan),
            ):
                print(f"[frozenobs] step={self._dbg_n} {name}: "
                      f"mean={blk.mean().item():+.3f} absmax={blk.abs().max().item():.3f} "
                      f"v0={blk[0, :4].tolist()}")
            if out is not None:
                print(f"[frozenout] step={self._dbg_n} out0={out[0].tolist()}")
            else:
                print(f"[frozenout] step={self._dbg_n} HOLD (no policy)")
            self._dbg_n += 1
        targets = self._default12.clone()
        if out is not None:
            targets += out.clamp(-1.0, 1.0)      # scale 1.0 + clip +-1 (velocity)
        for name in ("set_joint_position_target", "write_joint_position_target_to_sim"):
            fn = getattr(robot, name, None)
            if fn is not None:
                # joint_ids must be int32: the physx warp kernel rejects int64
                fn(targets, joint_ids=self._leg_ids.to(torch.int32))
                break

    def _process_partial(self, actions: torch.Tensor) -> None:
        """Legacy v2 path: 3-dim cmd -> 68-dim partial obs -> 14 targets.

        h_cmd is pinned to H_CMD_MIN (0.40): the partial base has no H*
        input, and this gives the EXACT v2 termination floor (0.40 - 0.05
        = 0.35) while leaving the height reward near-dead (v2 push had no
        height term at all) - so the A/B differs from v2 only in the goal /
        reward machinery under test, not in floor or height dynamics."""
        st = _state(self._env)
        self._raw_actions[:] = actions
        vel = actions[:, :3].clamp(-0.8, 0.8)
        self._processed_actions[:] = vel
        self._vel = vel
        st.last_vel_cmd = vel
        st.h_cmd = H_CMD_MIN    # see docstring: exact v2 floor (0.35) in A/B
        robot = self._asset
        lin = vmdp.base_lin_vel(self._env)
        ang = vmdp.base_ang_vel(self._env)
        grav = vmdp.projected_gravity(self._env)
        leg_pos = robot.data.joint_pos[:, self._leg_ids] - robot.data.default_joint_pos[:, self._leg_ids]
        leg_vel = robot.data.joint_vel[:, self._leg_ids]
        head_pos = robot.data.joint_pos[:, self._head_ids] - robot.data.default_joint_pos[:, self._head_ids]
        arm_pos = robot.data.joint_pos[:, self._arm_ids] - robot.data.default_joint_pos[:, self._arm_ids]
        arm_vel = robot.data.joint_vel[:, self._arm_ids]
        obs = torch.cat(
            [
                lin,
                ang,
                grav,
                vel,
                leg_pos,
                leg_vel,
                head_pos,
                arm_pos,
                arm_vel,
                self._last,
            ],
            dim=-1,
        )
        out = None
        if self._diag_mode != "hold":
            with torch.inference_mode():
                out = self._policy(obs.to(self._asset.device))
            self._last = out
        if self._diag_debug and self._dbg_n < 5:
            for name, blk in (
                ("lin_vel", lin), ("ang_vel", ang), ("gravity", grav), ("cmd", vel),
                ("leg_pos", leg_pos), ("leg_vel", leg_vel), ("head_pos", head_pos),
                ("arm_pos", arm_pos), ("arm_vel", arm_vel), ("last_act", self._last),
            ):
                print(f"[frozenobs] step={self._dbg_n} {name}: "
                      f"mean={blk.mean().item():+.3f} absmax={blk.abs().max().item():.3f} "
                      f"v0={blk[0, :4].tolist()}")
            if out is not None:
                print(f"[frozenout] step={self._dbg_n} out0={out[0].tolist()}")
            else:
                print(f"[frozenout] step={self._dbg_n} HOLD (no policy)")
            self._dbg_n += 1
        targets = self._default14.clone()
        if out is not None:
            targets[:, :12] += 0.25 * out[:, :12]
            targets[:, 12:] += 0.5 * out[:, 12:14]
        for name in ("set_joint_position_target", "write_joint_position_target_to_sim"):
            fn = getattr(robot, name, None)
            if fn is not None:
                # joint_ids must be int32: the physx warp kernel rejects int64
                fn(targets, joint_ids=self._ids14.to(torch.int32))
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
    """Frozen squat base (v3) or legacy partial base.

    Default path models/k1_push_base.pt is a symlink the launch script points
    at k1_squat_base.pt (squat) or k1_partialctrl_base.pt (legacy); the mode
    is derived from the resolved basename ("partialctrl" -> partial). Set
    PUSH_BASE_POLICY / PUSH_BASE_MODE to override without touching the tree
    (env vars are stripped by singularity --containall, hence the symlink)."""

    class_type = FrozenBaseVelocityAction
    asset_name: str = "robot"
    base_policy_path: str = "models/k1_push_base.pt"


# ---------------------------------------------------------------------------
# cfg helpers re-exported for the env cfg
# ---------------------------------------------------------------------------
__all__ = [
    "K1_LEG_JOINTS", "K1_HEAD_JOINTS", "K1_LEFT_ARM_JOINTS", "K1_RIGHT_ARM_JOINTS",
    "LEFT_EE", "RIGHT_EE", "DifferentialIKControllerCfg", "DifferentialInverseKinematicsActionCfg",
    "WristTargetCommandCfg", "WristTargetCommand", "FrozenBaseVelocityAction", "FrozenBaseVelocityActionCfg",
    "randomize_box_geometry", "apply_box_green_alpha", "reset_box", "reset_wrist_targets",
    "goal_dist_curriculum", "goal_yaw_curriculum", "push_box_teacher",
    "corner_goal_tracking", "centroid_goal_tracking", "box_goal_progress", "box_vel_toward_goal",
    "box_spin_penalty", "wrist_target_tracking", "wrist_box_proximity",
    "track_cmd_lin_vel_exp", "track_cmd_ang_vel_exp", "base_height_command",
    "success_bonus", "failure_terminated", "goal_reached", "height_below_command",
    "box_out_of_bounds", "box_tipped",
    "vmdp",
]
