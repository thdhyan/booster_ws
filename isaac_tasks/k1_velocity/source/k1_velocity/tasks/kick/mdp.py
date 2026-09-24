"""MDP for the K1 kick task.

Extends isaaclab_tasks.core.velocity.mdp with ball-specific
observations (pose/velocity in robot frame), reward signals (goal-scored, ball-to-goal
progress), and reset-family samplers for curriculum learning.

Key design:
  - Ball is a moving RigidObject (not kinematic).
  - Goal posts are kinematic and replicated per environment at the +x field end.
  - Observations: policy obs for blind learner + teacher obs for privileged training.
  - Rewards: sparse goal_scored plus approach, goal-direction, stance, and
    ball-to-goal progress shaping with locomotion regularization.
  - Resets: OmniReset with three families (at_ball_shoot / stand_ready / walk_up),
    population-sampled with uniform distribution.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

# Re-export all standard mdp functions from isaaclab's velocity task.
# Isaac Lab 3.0-EA (dl) keeps this at isaaclab_tasks.core.velocity; the
# isaac-lab image renamed it to isaaclab_tasks.manager_based.locomotion.velocity
# (identical lazy_export module contents).
try:
    from isaaclab_tasks.core.velocity.mdp import (
        JointPositionActionCfg,
        UniformVelocityCommandCfg,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        generated_commands,
        joint_pos_rel,
        joint_vel_rel,
        last_action,
        height_scan,
        track_lin_vel_xy_yaw_frame_exp,
        track_ang_vel_z_world_exp,
        is_terminated,
        lin_vel_z_l2,
        flat_orientation_l2,
        action_rate_l2,
        joint_acc_l2,
        joint_torques_l2,
        joint_pos_limits,
        joint_deviation_l1,
        time_out,
        root_height_below_minimum,
        bad_orientation,
        reset_scene_to_default,
        reset_joints_by_scale,
        push_by_setting_velocity,
        randomize_rigid_body_mass,
        terrain_levels_vel,
    )
except (ImportError, ModuleNotFoundError):
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (
        JointPositionActionCfg,
        UniformVelocityCommandCfg,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        generated_commands,
        joint_pos_rel,
        joint_vel_rel,
        last_action,
        height_scan,
        track_lin_vel_xy_yaw_frame_exp,
        track_ang_vel_z_world_exp,
        is_terminated,
        lin_vel_z_l2,
        flat_orientation_l2,
        action_rate_l2,
        joint_acc_l2,
        joint_torques_l2,
        joint_pos_limits,
        joint_deviation_l1,
        time_out,
        root_height_below_minimum,
        bad_orientation,
        reset_scene_to_default,
        reset_joints_by_scale,
        push_by_setting_velocity,
        randomize_rigid_body_mass,
        terrain_levels_vel,
    )

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


# ============================================================================
# Kick Task Constants
# ============================================================================
# Field dimensions (parity with soccer_sim.py / MuJoCo robocup demo)
FIELD_L = 8.0
FIELD_W = 5.0
GOAL_W = 2.0
GOAL_H = 0.9
BALL_RADIUS = 0.11
BALL_MASS = 0.43

# Goal position in each environment's local frame (fixed at +x).
GOAL_POS = torch.tensor([FIELD_L / 2, 0.0, GOAL_H / 2], dtype=torch.float32)


def _goal_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the per-environment goal centres in world coordinates."""
    goal = GOAL_POS.to(device=env.device, dtype=env.scene.env_origins.dtype)
    return goal.expand(env.num_envs, -1) + env.scene.env_origins


def _progress_state(env: ManagerBasedRLEnv) -> SimpleNamespace:
    """Per-environment reset distances used by the two progress rewards."""
    state = getattr(env, "kick_progress", None)
    if state is None:
        state = SimpleNamespace(
            initial_goal_dist=torch.zeros(env.num_envs, device=env.device),
            initial_robot_ball_dist=torch.zeros(env.num_envs, device=env.device),
        )
        env.kick_progress = state
    return state


# ============================================================================
# Custom Kick Task Observations
# ============================================================================
def ball_pos_in_robot_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ball position in robot base frame (3-dim).

    Returns:
        torch.Tensor: (num_envs, 3) ball position in robot frame.
    """
    ball_pos_w = env.scene["ball"].data.root_pos_w.torch
    robot_pos_w = env.scene["robot"].data.root_pos_w.torch
    robot_quat_w = env.scene["robot"].data.root_quat_w.torch  # xyzw (IL 3.0)

    # Compute ball position relative to robot origin
    ball_pos_rel = ball_pos_w - robot_pos_w

    # Rotate using isaaclab's quat_apply_inverse (which expects torch tensors)
    from isaaclab.utils.math import quat_apply_inverse
    ball_pos_robot = quat_apply_inverse(robot_quat_w, ball_pos_rel)

    return ball_pos_robot


def ball_lin_vel_in_robot_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ball linear velocity in robot base frame (3-dim).

    Returns:
        torch.Tensor: (num_envs, 3) ball linear velocity in robot frame.
    """
    ball_lin_vel_w = env.scene["ball"].data.root_lin_vel_w.torch
    robot_quat_w = env.scene["robot"].data.root_quat_w.torch  # xyzw (IL 3.0)

    # Rotate into robot frame
    from isaaclab.utils.math import quat_apply_inverse
    ball_lin_vel_robot = quat_apply_inverse(robot_quat_w, ball_lin_vel_w)

    return ball_lin_vel_robot


def goal_pos_in_robot_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged goal centre (x, y) in the robot base frame."""
    from isaaclab.utils.math import quat_apply_inverse

    robot = env.scene["robot"]
    goal_rel_w = _goal_positions_w(env) - robot.data.root_pos_w.torch
    return quat_apply_inverse(robot.data.root_quat_w.torch, goal_rel_w)[:, :2]


def goal_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance from ball to the per-environment goal centre: shape ``(N,)``."""
    ball_pos_w = env.scene["ball"].data.root_pos_w.torch
    return torch.norm(ball_pos_w - _goal_positions_w(env), dim=-1)


def goal_scored(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary indicator: ball reached its environment's goal volume."""
    ball_pos_w = env.scene["ball"].data.root_pos_w.torch
    goal_pos_w = _goal_positions_w(env)
    rel = ball_pos_w - goal_pos_w
    return (
        (rel[:, 0] > -0.1)
        & (rel[:, 0] < 0.2)
        & (rel[:, 1].abs() < GOAL_W / 2)
        & (rel[:, 2].abs() < GOAL_H / 2)
    )


def ball_to_goal_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fraction of this reset's initial ball-to-goal distance closed."""
    state = _progress_state(env)
    current_dist = goal_distance(env)
    uninitialised = state.initial_goal_dist < 1e-6
    state.initial_goal_dist[uninitialised] = current_dist[uninitialised]
    initial_dist = state.initial_goal_dist.clamp_min(1e-6)
    return ((initial_dist - current_dist) / initial_dist).clamp(0.0, 1.0)


def approach_ball(env: ManagerBasedRLEnv, min_dist: float = 0.5) -> torch.Tensor:
    """Reward closing the initial robot-to-ball distance while still far away."""
    state = _progress_state(env)
    ball_pos_w = env.scene["ball"].data.root_pos_w.torch
    robot_pos_w = env.scene["robot"].data.root_pos_w.torch
    current_dist = torch.norm(ball_pos_w[:, :2] - robot_pos_w[:, :2], dim=-1)
    uninitialised = state.initial_robot_ball_dist < 1e-6
    state.initial_robot_ball_dist[uninitialised] = current_dist[uninitialised]
    initial_dist = state.initial_robot_ball_dist.clamp_min(1e-6)
    progress = ((initial_dist - current_dist) / initial_dist).clamp(0.0, 1.0)
    return progress * (current_dist > min_dist).float()


def kick_toward_goal(env: ManagerBasedRLEnv, speed_scale: float = 0.5) -> torch.Tensor:
    """Exponential reward for ball velocity directed toward the goal."""
    ball = env.scene["ball"]
    velocity = ball.data.root_lin_vel_w.torch
    speed = torch.norm(velocity, dim=-1)
    goal_axis = _goal_positions_w(env) - ball.data.root_pos_w.torch
    goal_axis = goal_axis / goal_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    along_goal = (velocity * goal_axis).sum(dim=-1)
    quality = torch.exp((along_goal / speed_scale).clamp(0.0, 2.0))
    return quality * (speed > 0.05).float()


def align_stance(env: ManagerBasedRLEnv, max_dist: float = 0.75) -> torch.Tensor:
    """Reward lining the robot up behind the ball relative to the goal."""
    ball_pos_w = env.scene["ball"].data.root_pos_w.torch
    robot_pos_w = env.scene["robot"].data.root_pos_w.torch
    robot_to_ball = ball_pos_w[:, :2] - robot_pos_w[:, :2]
    ball_to_goal = _goal_positions_w(env)[:, :2] - ball_pos_w[:, :2]
    robot_to_ball = robot_to_ball / robot_to_ball.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ball_to_goal = ball_to_goal / ball_to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cosine = (robot_to_ball * ball_to_goal).sum(dim=-1).clamp(-1.0, 1.0)
    distance = torch.norm(ball_pos_w[:, :2] - robot_pos_w[:, :2], dim=-1)
    return 0.5 * (cosine + 1.0) * (distance < max_dist).float()


# ============================================================================
# Kick Task Reset Samplers (OmniReset families)
# ============================================================================
def _set_ball_pos(env: ManagerBasedRLEnv, env_ids: torch.Tensor, ball_pos_w: torch.Tensor) -> None:
    """Write ball root state to sim for given env_ids."""
    ball = env.scene["ball"]
    # IL 3.0: .data.* returns ProxyArray → take .torch for tensor ops
    root_state = ball.data.default_root_state.torch[env_ids].clone()
    # world pos: default_root_state includes env origins; add env offset
    root_state[:, :3] = ball_pos_w + env.scene.env_origins[env_ids]
    root_state[:, 7:10] = 0.0   # zero lin vel
    root_state[:, 10:13] = 0.0  # zero ang vel
    # IL 3.0: write_root_state_to_sim removed → pose + velocity index variants
    # (root_state layout: pos xyz, quat xyzw | lin vel, ang vel)
    ball.write_root_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
    ball.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)


def _reset_progress(env: ManagerBasedRLEnv, env_ids: torch.Tensor, ball_pos: torch.Tensor) -> None:
    """Record reset distances before rewards can observe the new ball pose."""
    state = _progress_state(env)
    robot_local_xy = torch.tensor(env.scene["robot"].cfg.init_state.pos[:2], device=env.device)
    robot_pos_w = env.scene.env_origins[env_ids, :2] + robot_local_xy
    state.initial_goal_dist[env_ids] = torch.norm(ball_pos - GOAL_POS.to(env.device), dim=-1)
    state.initial_robot_ball_dist[env_ids] = torch.norm(ball_pos[:, :2] - robot_pos_w[:, :2], dim=-1)


def reset_ball_omnireset(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """OmniReset: population-sample a family per env and place ball accordingly.

    Ratios: at_ball_shoot 50%, stand_ready 30%, walk_up 20%.
    """
    num_reset = len(env_ids)
    device = env.device

    # Sample family per env
    rand = torch.rand(num_reset, device=device)
    mask_shoot = rand < 0.50
    mask_ready = (rand >= 0.50) & (rand < 0.80)
    mask_walk  = rand >= 0.80

    ball_pos = torch.zeros((num_reset, 3), device=device)
    ball_pos[:, 2] = BALL_RADIUS + 0.01  # default z

    # Family 1: at_ball_shoot — ball 0.3–0.6m ahead of robot, small y offset
    n = mask_shoot.sum()
    if n > 0:
        dist = 0.3 + 0.3 * torch.rand(n, device=device)
        y    = 0.10 * (2.0 * torch.rand(n, device=device) - 1.0)
        ball_pos[mask_shoot, 0] = dist
        ball_pos[mask_shoot, 1] = y

    # Family 2: stand_ready — ball at origin (robot at default, ~0.6–1.2m back)
    # Ball stays at (0,0,z); robot default pose is ~0 — that's fine

    # Family 3: walk_up — ball 2–3.5m ahead, wide y cone
    n = mask_walk.sum()
    if n > 0:
        dist = 2.0 + 1.5 * torch.rand(n, device=device)
        y    = 2.0 * (torch.rand(n, device=device) - 0.5)
        ball_pos[mask_walk, 0] = dist
        ball_pos[mask_walk, 1] = y

    _set_ball_pos(env, env_ids, ball_pos)
    _reset_progress(env, env_ids, ball_pos)


# ============================================================================
# Shared head-aiming helpers (P4 teacher reward, P3 detector proxy)
# ============================================================================
HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]


def ball_head_angles(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(yaw_err, pitch_err): ball bearing in the current head frame (radians)."""
    ball_rf = ball_pos_in_robot_frame(env)
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(HEAD_JOINTS, preserve_order=True)
    head_pos = robot.data.joint_pos[:, ids]
    yaw_q, pitch_q = head_pos[:, 0], head_pos[:, 1]
    yaw = torch.atan2(ball_rf[:, 1], ball_rf[:, 0]) - yaw_q
    pitch = torch.atan2(ball_rf[:, 2] - 0.55, torch.hypot(ball_rf[:, 0], ball_rf[:, 1])) - pitch_q
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    return yaw, pitch


def ball_fov_visible(env: ManagerBasedRLEnv, hfov_deg: float = 69.4) -> torch.Tensor:
    """Geometric FOV test through the current head pose (detector proxy)."""
    import math as _m

    yaw, pitch = ball_head_angles(env)
    hfov = _m.radians(hfov_deg)
    vfov = 2.0 * _m.atan(_m.tan(hfov / 2.0) * 240.0 / 320.0)
    ball_rf = ball_pos_in_robot_frame(env)
    return (ball_rf[:, 0] > 0.05) & (yaw.abs() < hfov / 2.0) & (pitch.abs() < vfov / 2.0)


def track_ball_head_exp(env: ManagerBasedRLEnv, std: float = 0.35) -> torch.Tensor:
    """exp(-(yaw_err^2 + pitch_err^2)/std^2) — head points at the ball (GT, reward-time)."""
    yaw, pitch = ball_head_angles(env)
    return torch.exp(-(yaw * yaw + pitch * pitch) / (std * std))


def ball_in_frame(env: ManagerBasedRLEnv, min_dist: float = 0.75) -> torch.Tensor:
    """Ball inside the camera FOV while the robot is still far (>min_dist)."""
    visible = ball_fov_visible(env).float()
    dist = torch.norm(ball_pos_in_robot_frame(env), dim=-1)
    return visible * (dist > min_dist).float()
