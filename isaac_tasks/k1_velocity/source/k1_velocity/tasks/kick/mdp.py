"""MDP for the K1 kick task.

Extends isaaclab_tasks.manager_based.locomotion.velocity.mdp with ball-specific
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

# Re-export all standard mdp functions from isaaclab's velocity task
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
    ball_pos_w = env.scene["ball"].data.root_pos_w
    robot_pos_w = env.scene["robot"].data.root_pos_w
    robot_quat_w = env.scene["robot"].data.root_quat_w

    # Convert warp tensors to torch if needed
    if not isinstance(ball_pos_w, torch.Tensor):
        ball_pos_w = torch.as_tensor(ball_pos_w)
    if not isinstance(robot_pos_w, torch.Tensor):
        robot_pos_w = torch.as_tensor(robot_pos_w)
    if not isinstance(robot_quat_w, torch.Tensor):
        robot_quat_w = torch.as_tensor(robot_quat_w)

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
    ball_lin_vel_w = env.scene["ball"].data.root_lin_vel_w
    robot_quat_w = env.scene["robot"].data.root_quat_w

    # Convert warp tensors to torch if needed
    if not isinstance(robot_quat_w, torch.Tensor):
        robot_quat_w = torch.as_tensor(robot_quat_w)
    if not isinstance(ball_lin_vel_w, torch.Tensor):
        ball_lin_vel_w = torch.as_tensor(ball_lin_vel_w)

    # Rotate using isaaclab's quat_apply_inverse (which expects torch tensors)
    from isaaclab.utils.math import quat_apply_inverse
    ball_lin_vel_robot = quat_apply_inverse(robot_quat_w, ball_lin_vel_w)

    return ball_lin_vel_robot


def goal_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance from ball to goal (scalar per env).

    Returns:
        torch.Tensor: (num_envs, 1) Euclidean distance from ball to goal.
    """
    ball_pos_w = env.scene["ball"].data.root_pos_w
    if not isinstance(ball_pos_w, torch.Tensor):
        ball_pos_w = torch.as_tensor(ball_pos_w)
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
    ball_pos_w = env.scene["ball"].data.root_pos_w
    if not isinstance(ball_pos_w, torch.Tensor):
        ball_pos_w = torch.as_tensor(ball_pos_w)
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
    ball_pos_w = env.scene["ball"].data.root_pos_w
    if not isinstance(ball_pos_w, torch.Tensor):
        ball_pos_w = torch.as_tensor(ball_pos_w)
    goal_pos = GOAL_POS.to(ball_pos_w.device)

    # Current distance to goal
    current_dist = torch.norm(ball_pos_w - goal_pos, dim=-1, keepdim=True)

    # Initial distance (ball spawn at origin, goal at +x)
    init_ball_pos = torch.zeros_like(ball_pos_w)
    init_ball_pos[:, 0] = 0.0  # x
    init_ball_pos[:, 1] = 0.0  # y
    init_ball_pos[:, 2] = BALL_RADIUS  # z
    init_dist = torch.norm(init_ball_pos - goal_pos, dim=-1, keepdim=True)

    # Progress: (init_dist - current_dist) / init_dist, clamped to [0, 1]
    progress = torch.clamp((init_dist - current_dist) / (init_dist + 1e-6), 0.0, 1.0)

    return progress


# ============================================================================
# Kick Task Reset Samplers (OmniReset families)
# ============================================================================
def reset_at_ball_shoot(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset: ball 0.3–0.6m in front of one foot, robot ready to kick.

    Family 1: at_ball_shoot — robot close to ball, prepared to kick.
    """
    num_reset = len(env_ids)

    # Random ball distance: 0.3–0.6m
    ball_dist = 0.3 + 0.3 * torch.rand(num_reset, device=env.device)

    # Random heading / y offset for the ball
    ball_y = (BALL_RADIUS + 0.05) * (2.0 * torch.rand(num_reset, device=env.device) - 1.0)

    # Ball position: (0.5 m from robot origin in +x, y offset, z=radius+margin)
    ball_pos_w = torch.zeros((num_reset, 3), device=env.device)
    ball_pos_w[:, 0] = ball_dist
    ball_pos_w[:, 1] = ball_y
    ball_pos_w[:, 2] = BALL_RADIUS + 0.01

    # Robot at default stance (handled by reset_scene_to_default)
    env.scene["ball"].data.root_pos_w[env_ids] = ball_pos_w
    env.scene["ball"].data.root_lin_vel_w[env_ids] = 0.0
    env.scene["ball"].data.root_ang_vel_w[env_ids] = 0.0


def reset_stand_ready(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset: robot 0.6–1.2m behind ball, ball at origin, ready to approach.

    Family 2: stand_ready — robot standing back, preparing approach.
    """
    num_reset = len(env_ids)

    # Ball at origin (0, 0, radius+margin)
    ball_pos_w = torch.zeros((num_reset, 3), device=env.device)
    ball_pos_w[:, 2] = BALL_RADIUS + 0.01

    # Robot pushed back: -0.6 to -1.2m (via velocity perturbation or direct reset)
    # For now, rely on reset_scene_to_default + ball placement
    env.scene["ball"].data.root_pos_w[env_ids] = ball_pos_w
    env.scene["ball"].data.root_lin_vel_w[env_ids] = 0.0
    env.scene["ball"].data.root_ang_vel_w[env_ids] = 0.0


def reset_walk_up(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset: robot 2–3.5m from ball, ball in wide cone, approach from distance.

    Family 3: walk_up — robot far from ball, must approach and kick.
    """
    num_reset = len(env_ids)

    # Random ball distance: 2–3.5m
    ball_dist = 2.0 + 1.5 * torch.rand(num_reset, device=env.device)

    # Random y offset: wide cone (±1m)
    ball_y = 2.0 * (torch.rand(num_reset, device=env.device) - 0.5)

    # Ball position
    ball_pos_w = torch.zeros((num_reset, 3), device=env.device)
    ball_pos_w[:, 0] = ball_dist
    ball_pos_w[:, 1] = ball_y
    ball_pos_w[:, 2] = BALL_RADIUS + 0.01

    env.scene["ball"].data.root_pos_w[env_ids] = ball_pos_w
    env.scene["ball"].data.root_lin_vel_w[env_ids] = 0.0
    env.scene["ball"].data.root_ang_vel_w[env_ids] = 0.0
