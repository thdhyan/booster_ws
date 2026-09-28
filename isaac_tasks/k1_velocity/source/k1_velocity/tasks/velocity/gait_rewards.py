"""AGILE-derived gait-shaping rewards for the K1 velocity task.

Is Isaac Lab's locomotion ``mdp`` has no foot-posture or foot-spacing terms, and
NVIDIA's AGILE velocity task for the Booster T1 (``agile/rl_env/mdp/rewards``,
``agile/rl_env/tasks/locomotion/t1/velocity_env_cfg.py``) leans on exactly
those instead of an air-time reward.  AGILE deliberately omits ``feet_air_time``
-- air-time is a classic degenerate attractor (hop, or never load one foot) --
and replaces it with foot slip / roll / yaw / spacing shaping.

Every function here returns a *penalty* (>= 0, 0 = ideal) so it can be wired
straight into ``RewTerm`` with a negative weight, matching AGILE's convention.
"""
from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Planar yaw (rotation about world +z) of a wxyz quaternion."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _roll_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Roll (rotation about the body's forward axis) of a wxyz quaternion."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def feet_yaw_mean_vs_base(
    env,
    feet_asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise feet that drift out of the base's heading (AGILE ``feet_yaw_mean``).

    Without this a policy can turn a foot sideways, or track with one foot
    rotated, and still satisfy the velocity tracker.
    """
    robot = env.scene[asset_cfg.name]
    foot_quat = robot.data.body_quat_w[:, feet_asset_cfg.body_ids, :]
    base_quat = robot.data.body_quat_w[:, base_body_cfg.body_ids, :]
    rel = _yaw_from_quat(foot_quat) - _yaw_from_quat(base_quat)
    rel = torch.atan2(torch.sin(rel), torch.cos(rel))  # wrap to (-pi, pi]
    return rel.abs().mean(dim=1)


def feet_yaw_diff_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared difference between the two feet's world yaw (AGILE ``feet_yaw_diff``)."""
    robot = env.scene[asset_cfg.name]
    yaw = _yaw_from_quat(robot.data.body_quat_w[:, asset_cfg.body_ids, :])
    return torch.square(yaw[:, 0] - yaw[:, 1])


def feet_roll_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared foot roll -- keeps the sole flat instead of toe-walking (AGILE ``feet_roll``)."""
    robot = env.scene[asset_cfg.name]
    roll = _roll_from_quat(robot.data.body_quat_w[:, asset_cfg.body_ids, :])
    return torch.square(roll).mean(dim=1)


def feet_distance_from_ref(env, asset_cfg: SceneEntityCfg, ref_distance: float) -> torch.Tensor:
    """Squared error of the two feet's horizontal separation from ``ref_distance``.

    Too narrow -> the legs scissor and collide; too wide -> a wide, splayed,
    robot-waddle gait.  AGILE uses 0.2 m on the T1.
    """
    robot = env.scene[asset_cfg.name]
    pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    sep = torch.linalg.norm(pos[:, 0, :2] - pos[:, 1, :2], dim=-1)
    return torch.square(sep - ref_distance)
