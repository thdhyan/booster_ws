"""MDP for the K1 kick task.

Extends isaaclab_tasks.core.velocity.mdp with ball-specific
observations (pose/velocity in robot frame), reward signals (goal-scored, ball-to-goal
progress), and reset-family samplers for curriculum learning.

Key design:
  - Ball is a moving RigidObject (not kinematic).
  - Goal posts are kinematic (just for contact detection in future; currently
    ball-to-goal progress uses a fixed goal position at +x end of field).
  - Observations: policy obs for blind learner + teacher obs for privileged training.
  - Rewards: sparse goal_scored (dominant) + light ball_to_goal_progress +
    locomotion regularization (no contact shaping).
  - Resets: OmniReset with three families (at_ball_shoot / stand_ready / walk_up),
    population-sampled with uniform distribution.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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

# Goal position (fixed, at +x end of field)
GOAL_POS = torch.tensor([FIELD_L / 2, 0.0, GOAL_H / 2], dtype=torch.float32)


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


def goal_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance from ball to goal (scalar per env).

    Returns:
        torch.Tensor: (num_envs, 1) Euclidean distance from ball to goal.
    """
    ball_pos_w = env.scene["ball"].data.root_pos_w.torch
    goal_pos = GOAL_POS.to(ball_pos_w.device)
    dist = torch.norm(ball_pos_w - goal_pos, dim=-1, keepdim=True)
    return dist


def goal_scored(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary indicator: ball reached goal (within goal volume).

    The goal is a box at (+x, y=±1m, z=0.45±0.45m).
    Simplified: score if ball is within goal volume (x>FIELD_L/2-0.1, |y|<GOAL_W/2, 0<z<GOAL_H).

    Returns:
        torch.Tensor: (num_envs,) binary score indicator as bool.
    """
    ball_pos_w = env.scene["ball"].data.root_pos_w.torch
    goal_x_threshold = FIELD_L / 2 - 0.1  # 0.1m margin for goal line

    scored = (
        (ball_pos_w[:, 0] > goal_x_threshold) &  # past goal line
        (torch.abs(ball_pos_w[:, 1]) < GOAL_W / 2) &  # within goal width
        (ball_pos_w[:, 2] > 0.0) &  # above ground
        (ball_pos_w[:, 2] < GOAL_H)  # below crossbar
    )

    return scored


def ball_to_goal_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalized progress toward goal (distance decrease).

    Reward for reducing ball-to-goal distance. Measured as fraction of initial
    distance closed (capped at 1.0 for overshooting).

    Returns:
        torch.Tensor: (num_envs, 1) progress in range [0, 1].
    """
    ball_pos_w = env.scene["ball"].data.root_pos_w.torch
    goal_pos = GOAL_POS.to(ball_pos_w.device)

    # Current distance to goal
    current_dist = torch.norm(ball_pos_w - goal_pos, dim=-1)  # (num_envs,)

    # Initial distance (ball spawn at origin, goal at +x)
    init_ball_pos = torch.zeros_like(ball_pos_w)
    init_ball_pos[:, 2] = BALL_RADIUS  # z
    init_dist = torch.norm(init_ball_pos - goal_pos, dim=-1)  # (num_envs,)

    # Progress: (init_dist - current_dist) / init_dist, clamped to [0, 1]
    progress = torch.clamp((init_dist - current_dist) / (init_dist + 1e-6), 0.0, 1.0)

    return progress


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
    return torch.exp(-(yaw * yaw + pitch * pitch) / (std * std)).unsqueeze(-1)


def ball_in_frame(env: ManagerBasedRLEnv, min_dist: float = 0.75) -> torch.Tensor:
    """Ball inside the camera FOV while the robot is still far (>min_dist)."""
    visible = ball_fov_visible(env).float().unsqueeze(-1)
    dist = torch.norm(ball_pos_in_robot_frame(env), dim=-1, keepdim=True)
    return visible * (dist > min_dist).float()
