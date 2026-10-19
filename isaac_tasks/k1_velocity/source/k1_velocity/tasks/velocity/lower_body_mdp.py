"""Curriculum terms for the Stage-0 lower-body base (lower_body_env_cfg.py).

Both ramps are TIME-based on purpose. The squat teacher's height-gated depth curriculum
(squat_mdp.squat_depth_curriculum) never got past the 0.50-0.55 m band because its 2.5 cm
height-error gate never passed, so the policy learned a 2-7 cm dip that reads as standing.

Teacher v4 (2026-10-16): pure time ramps broke the other way. In v2 the fall rate climbed
23 % -> 66 % exactly while upper pose / stub force / H* ramped to full (it 2000-4000) and
never came back (time_out 0 % for the last 6000 it). With ``gate`` set, the ramps run on
GATED iterations: the clock only advances while the time-out fraction of finished episodes
(EMA) is >= gate, so the disturbances wait for the policy. ``gate=None`` keeps v2 behaviour.
"""
from __future__ import annotations

import torch


def _iteration(env, steps_per_iter: int, gate: float | None = None) -> float:
    if gate is None:
        return (env.sim.get_physics_step_count() // env.cfg.decimation) / steps_per_iter
    # one shared clock per env; several curriculum terms call this on the same reset
    st = env.__dict__.setdefault("_lb_gate", {"step": 0, "it": 0.0, "ema": 0.0})
    step = env.sim.get_physics_step_count() // env.cfg.decimation
    if step != st["step"]:
        if st["ema"] >= gate:
            st["it"] += (step - st["step"]) / steps_per_iter
        st["step"] = step
    return st["it"]


def _frac(env, start_iter: int, end_iter: int, steps_per_iter: int, gate: float | None = None) -> float:
    it = _iteration(env, steps_per_iter, gate)
    return min(max((it - start_iter) / max(end_iter - start_iter, 1), 0.0), 1.0)


def survival_gate(env, env_ids, beta: float = 0.98) -> float:
    """Update the time-out EMA from the episodes ending now; logged as ``Curriculum/survival``.

    Must be declared BEFORE the gated ramps in the curriculum cfg (terms run in field order).
    """
    st = env.__dict__.setdefault("_lb_gate", {"step": 0, "it": 0.0, "ema": 0.0})
    if env_ids is not None and len(env_ids) > 0:
        frac = float(env.termination_manager.time_outs[env_ids].float().mean())
        st["ema"] = beta * st["ema"] + (1.0 - beta) * frac
    return st["ema"]


def gated_iteration(env, env_ids, steps_per_iter: int = 24, gate: float = 0.6) -> float:
    """Logged as ``Curriculum/gated_iter``: how far the gated ramps have advanced."""
    return _iteration(env, steps_per_iter, gate)


def upper_pose_ramp(
    env,
    env_ids,
    event_term: str,
    start_scale: float = 0.15,
    end_scale: float = 1.0,
    ramp_start_iter: int = 300,
    ramp_end_iter: int = 2000,
    steps_per_iter: int = 24,
    gate: float | None = None,
) -> float:
    """partial_mdp.arm_pose_curriculum with an optional survival gate."""
    s = start_scale + (end_scale - start_scale) * _frac(env, ramp_start_iter, ramp_end_iter, steps_per_iter, gate)
    env.event_manager.get_term_cfg(event_term).params["curriculum_scale"] = s
    return s


def root_height_below_terrain(env, minimum_height: float = 0.30, sensor_name: str = "height_scanner") -> torch.Tensor:
    """Fall check relative to the ground under the trunk (stairs/pits make world z useless)."""
    from .squat_mdp import _ray_mean_z

    z_w = env.scene["robot"].data.root_pos_w[:, 2]
    return (z_w - _ray_mean_z(env, sensor_name)) < float(minimum_height)


def squat_height_ramp(
    env,
    env_ids,
    command_name: str = "base_velocity",
    rel_squat: float = 0.5,
    low_start: float = 0.50,
    low_end: float = 0.40,
    high: float = 0.55,
    start_iter: int = 200,
    end_iter: int = 3000,
    steps_per_iter: int = 24,
    gate: float | None = None,
) -> float:
    """H* ~ U(low, high) for ``rel_squat`` of the envs; ``low`` ramps low_start -> low_end.

    Returns the current lower bound (logged as ``Curriculum/squat_height``).
    """
    low = low_start + (low_end - low_start) * _frac(env, start_iter, end_iter, steps_per_iter, gate)
    cfg = env.command_manager.get_term(command_name).cfg
    cfg.rel_squat_envs = rel_squat
    cfg.squat_height_range = (low, high)
    return low


def force_range_ramp(
    env,
    env_ids,
    event_term: str = "stub_force",
    max_force: float = 20.0,
    max_torque: float = 0.0,
    start_iter: int = 1000,
    end_iter: int = 4000,
    steps_per_iter: int = 24,
    gate: float | None = None,
) -> float:
    """Ramp the stub-tip wrench range of ``event_term`` from 0 to +-max_force N.

    Returns the current force bound (logged as ``Curriculum/stub_force``).
    """
    s = _frac(env, start_iter, end_iter, steps_per_iter, gate)
    params = env.event_manager.get_term_cfg(event_term).params
    params["force_range"] = (-s * max_force, s * max_force)
    params["torque_range"] = (-s * max_torque, s * max_torque)
    return s * max_force
