"""MDP terms for the residual skills on the frozen velocity student (ball track, kick).

Ball/goal geometry and the kick shaping come from the kick task (kick/mdp.py); this
module adds the rolling-ball tracking task and the residual-specific penalties.
Observations are privileged (GT ball state); a detector-fed student comes later.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from k1_velocity.tasks.kick import mdp as kmdp

try:
    from isaaclab_tasks.core.velocity.mdp import is_alive, reset_root_state_uniform  # noqa: F401
except ImportError:  # isaac-lab image layout
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import is_alive, reset_root_state_uniform  # noqa: F401
from k1_velocity.tasks.kick.mdp import (  # noqa: F401 (re-exported for the cfgs)
    BALL_RADIUS, ball_pos_in_robot_frame, ball_lin_vel_in_robot_frame, goal_pos_in_robot_frame,
    track_ball_head_exp, ball_fov_visible,
)


def _action_term(env, name: str = "skill"):
    return env.action_manager.get_term(name)


def _track_state(env) -> SimpleNamespace:
    st = getattr(env, "ball_track_state", None)
    if st is None:
        st = SimpleNamespace(
            ball_dir=torch.zeros(env.num_envs, 3, device=env.device),
            ball_speed=0.0,
        )
        env.ball_track_state = st
    return st


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
def head_ball_error(env) -> torch.Tensor:
    """(n, 2) ball bearing error in the head frame (yaw, pitch), radians."""
    yaw, pitch = kmdp.ball_head_angles(env)
    return torch.stack([yaw, pitch], dim=-1)


def student_command(env) -> torch.Tensor:
    """(n, 3) velocity command the skill is currently sending the frozen student."""
    return _action_term(env).cmd


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------
def ball_distance_keep(env, target: float = 0.8, std: float = 0.4) -> torch.Tensor:
    """exp(-(d_xy - target)^2 / std^2): stay at the tracking distance from the ball."""
    d = ball_pos_in_robot_frame(env)[:, :2].norm(dim=-1)
    return torch.exp(-((d - target) / std) ** 2)


def face_ball(env, std: float = 0.5) -> torch.Tensor:
    """exp(-bearing^2 / std^2): trunk faces the ball (bearing in the root frame)."""
    b = ball_pos_in_robot_frame(env)
    bearing = torch.atan2(b[:, 1], b[:, 0])
    return torch.exp(-(bearing / std) ** 2)


def ball_visible(env) -> torch.Tensor:
    """1 while the ball is inside the head camera FOV (geometric proxy)."""
    return ball_fov_visible(env).float()


def residual_l2(env, action_name: str = "skill") -> torch.Tensor:
    """|residual|^2 actually added to the student's leg output (keeps the skill a residual)."""
    return _action_term(env, action_name).residual.square().sum(-1)


def term_fired(env, term_names: list[str]) -> torch.Tensor:
    """1.0 where any of the named termination terms fired this step (failures only,
    unlike ``is_terminated``, which also fires on a success termination)."""
    tm = env.termination_manager
    out = torch.zeros(env.num_envs, device=env.device)
    for name in term_names:
        out = torch.maximum(out, tm.get_term(name).float())
    return out


def command_rate_l2(env) -> torch.Tensor:
    """|d cmd|^2 between control steps (smooth commands to the student)."""
    term = _action_term(env)
    prev = getattr(term, "_prev_cmd_for_rate", None)
    cur = term.cmd.clone()
    term._prev_cmd_for_rate = cur
    if prev is None or prev.shape != cur.shape:
        return torch.zeros(env.num_envs, device=env.device)
    return (cur - prev).square().sum(-1)


# ---------------------------------------------------------------------------
# ball-track events + curriculum
# ---------------------------------------------------------------------------
def reset_ball_rolling(env, env_ids: torch.Tensor, dist_range=(1.0, 2.5), lat: float = 1.0) -> None:
    """Ball 1-2.5 m away in a wide cone, rolling in a random direction at the curriculum speed."""
    st = _track_state(env)
    n, dev = len(env_ids), env.device
    d = dist_range[0] + (dist_range[1] - dist_range[0]) * torch.rand(n, device=dev)
    pos = torch.zeros(n, 3, device=dev)
    pos[:, 0] = d
    pos[:, 1] = lat * (2.0 * torch.rand(n, device=dev) - 1.0)
    pos[:, 2] = BALL_RADIUS + 0.01
    theta = (2.0 * torch.rand(n, device=dev) - 1.0) * math.pi
    direction = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros(n, device=dev)], dim=-1)
    st.ball_dir[env_ids] = direction
    ball = env.scene["ball"]
    root = ball.data.default_root_state.torch[env_ids].clone()
    root[:, :3] = pos + env.scene.env_origins[env_ids]
    root[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)   # xyzw identity (IL 3.0)
    root[:, 7:10] = direction * st.ball_speed
    root[:, 10:13] = 0.0
    ball.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    ball.write_root_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)


def steer_ball(env, env_ids: torch.Tensor, turn_prob: float = 0.3, max_turn: float = 1.2) -> None:
    """Interval: re-assert the roll speed and sometimes turn the ball (keeps it moving)."""
    st = _track_state(env)
    dev = env.device
    turn = (torch.rand(len(env_ids), device=dev) < turn_prob).float()
    ang = turn * (2.0 * torch.rand(len(env_ids), device=dev) - 1.0) * max_turn
    d = st.ball_dir[env_ids]
    c, s = torch.cos(ang), torch.sin(ang)
    st.ball_dir[env_ids] = torch.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], dim=-1)
    ball = env.scene["ball"]
    vel = ball.data.root_lin_vel_w.torch[env_ids].clone()
    vel[:, :2] = st.ball_dir[env_ids, :2] * st.ball_speed
    vel[:, 2] = 0.0
    ball.write_root_velocity_to_sim_index(root_velocity=torch.cat((vel, torch.zeros_like(vel)), -1), env_ids=env_ids)


def keep_ball_in_cell(env, env_ids: torch.Tensor, max_r: float = 4.0) -> None:
    """Interval: a ball that leaves the cell is respawned near the robot (no cross-env balls)."""
    ball = env.scene["ball"]
    rel = ball.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    out = env_ids[rel.norm(dim=-1) > max_r]
    if len(out):
        reset_ball_rolling(env, out)


def ball_speed_curriculum(env, env_ids, start_step: int = 24 * 200, end_step: int = 24 * 1500,
                          v_max: float = 0.6) -> float:
    """0 -> v_max roll speed over [start_step, end_step] control steps."""
    st = _track_state(env)
    frac = (env.common_step_counter - start_step) / max(end_step - start_step, 1)
    st.ball_speed = v_max * min(max(frac, 0.0), 1.0)
    return st.ball_speed


def ball_lost(env, max_dist: float = 3.5) -> torch.Tensor:
    """Ball further than max_dist from the robot = lost track (termination)."""
    return ball_pos_in_robot_frame(env)[:, :2].norm(dim=-1) > max_dist
