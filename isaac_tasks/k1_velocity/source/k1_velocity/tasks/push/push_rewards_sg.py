# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""SG-style + Jeon reward terms for the K1 box-push task, ARM-CONTACT variant.

WHY THIS EXISTS
---------------
The reference Push-Things box-pushing task (checked out at
``~/Projects/ebasa/Push-Things``, branch ``dhyan/obs_groups``) is **base-push**: an
ANYmal-C pushes with its body, and the high-level action is only a base velocity
command fed to a frozen blind locomotion policy. Seven of its shaping terms --
``approach_object_recip``, ``fast_approach``, ``task_box_to_goal_recip``,
``stop_at_goal_box_still``, ``heading_progress_combo``, ``push_pull_etiquette`` and
``parked_bonus`` -- are not present in that checkout at all; the revision that has
them is newer than the clone on disk, so the bodies could not be copied even in
principle. They are therefore written here from their documented signatures.

K1 pushes with its ARMS, not its body, so these are NOT a transcription. Three
quantities are reinterpreted:

===========================  ==========================  ==========================
term                         Push-Things (base-push)     here (arm-contact)
===========================  ==========================  ==========================
approach / contact distance  robot base -> box          nearest WRIST -> box face
"standoff"                   base distance from box     wrist withdrawal from box
"pulling"                    box dragged by the base    box dragged by the wrists
===========================  ==========================  ==========================

Using base distance for an arm-contact task is not a detail: a K1 can stand 0.4 m
from a box with both hands on it, so a base-distance approach term reads "arrived"
while the task is not started, and a base-distance standoff term fights the very
posture the push needs.

SELF-CONTAINED BY DESIGN
------------------------
No ``push_things`` import. That repo is a separate checkout on a feature branch and
must not become a runtime dependency of this one; everything here reads the scene
through our own ``push_mdp`` helpers.

The pure arithmetic lives in the ``_*`` helpers, which take plain tensors and touch
no Isaac symbol, so the shapes and the sign conventions are unit-tested on a laptop
(``tests/test_push_rewards_sg.py``). The public reward functions are thin adapters.
"""
from __future__ import annotations

import torch

from .push_mdp import _state, box_corners_base, wrist_box_gap, wrist_positions_base

# ---------------------------------------------------------------------------
# pure helpers (Isaac-free; unit-tested directly)
# ---------------------------------------------------------------------------


def recip(d: torch.Tensor, scale: float) -> torch.Tensor:
    """``scale / (scale + d)`` -- 1 at d=0, ->0 far away, and never negative.

    Used instead of ``1/d`` because the raw reciprocal is unbounded: at d=0.02 m it
    returns 50 and a single term swamps every other reward in the manager, which is
    the same failure the earlier ``wrist_box_proximity`` single-scale note describes.
    """
    return scale / (scale + d.clamp_min(0.0))


def inside_radius(dist: torch.Tensor, radius: float, smooth: float = 0.0) -> torch.Tensor:
    """1 inside ``radius``, ->0 outside, optionally with a linear ramp of width ``smooth``."""
    if smooth <= 0.0:
        return (dist <= radius).to(dist.dtype)
    return (1.0 - (dist - radius) / smooth).clamp(0.0, 1.0)


def heading_cos(fwd_xy: torch.Tensor, to_goal_xy: torch.Tensor) -> torch.Tensor:
    """cosine between the robot's forward axis and the box->goal direction, in [-1, 1]."""
    f = fwd_xy / fwd_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    t = to_goal_xy / to_goal_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (f * t).sum(-1).clamp(-1.0, 1.0)


def pull_fraction(box_v_xy: torch.Tensor, to_goal_unit: torch.Tensor) -> torch.Tensor:
    """Signed box velocity along the goal direction, in m/s.

    Positive is pushing (good). Negative is PULLING the box away from the goal,
    which for an arm-contact task means the wrists hooked the box and dragged it
    backwards -- a real failure mode that a pure distance-to-goal term rewards,
    because dragging the box toward the robot can transiently shorten the distance.
    """
    return (box_v_xy * to_goal_unit).sum(-1)


def engagement(gap: torch.Tensor, contact_dist: float) -> torch.Tensor:
    """1 when the wrists are still short of the box, 0 once within ``contact_dist``.

    Gates the goal-progress terms so they cannot be collected by shoving the box
    from a distance, and gates the approach term so it stops competing once the
    hands are on it.
    """
    return (gap > contact_dist).to(gap.dtype)


# ---------------------------------------------------------------------------
# public reward terms
# ---------------------------------------------------------------------------


def approach_object_recip(env, d_min: float = 0.40) -> torch.Tensor:
    """Arm-contact approach: reciprocal gap from the wrists to the box face.

    ``d_min`` is the scale, defaulting to 0.40 m rather than the base-push 2.0 m --
    with an arm the whole approach spans about one arm's reach, so a 2 m scale is
    nearly flat over the region the term is supposed to shape.
    """
    # max, not mean: scored on the FARTHER wrist, so one arm on the box earns little.
    return recip(wrist_box_gap(env).max(-1).values, d_min)


def fast_approach(env, contact_dist: float = 0.60) -> torch.Tensor:
    """Reward closing the wrist->box gap, but only while still out of contact.

    Uses the per-env gap delta rather than wrist speed so it cannot be farmed by
    swinging the arms in free space away from the box.
    """
    st = _state(env)
    gap = wrist_box_gap(env).max(-1).values   # farther wrist (both arms must close)
    prev = getattr(st, "sg_prev_gap", None)
    if prev is None or prev.shape != gap.shape:
        st.sg_prev_gap = gap.clone()
        return torch.zeros_like(gap)
    closing = (prev - gap) / max(1e-3, env.step_dt)
    st.sg_prev_gap = gap.clone()
    return closing.clamp(min=0.0) * engagement(gap, contact_dist)


def task_box_to_goal_recip(env, goal_radius_m: float = 0.35) -> torch.Tensor:
    """Reciprocal box-centre to fixed-goal distance (the task objective proper).

    Arm-agnostic: the box has to reach the goal whoever pushes it. Measured on the
    XY plane only -- the goal pose carries a yaw, and vertical error is the box
    tipping, which ``pushable_fallen``/``box_tipped`` already terminate on.
    """
    st = _state(env)
    box = env.scene["box"]
    d = (st.goal_pos[:, :2] - box.data.root_pos_w[:, :2]).norm(dim=-1)
    return recip(d, goal_radius_m)


def stop_at_goal_box_still(env, goal_radius_m: float = 0.35, v_box_thresh: float = 0.04) -> torch.Tensor:
    """Bonus for the box being inside the goal AND stationary.

    Without the stillness half this pays out for a box sweeping through the goal at
    speed, which is not the behaviour the task wants and which the success
    termination (``box_inside_and_parked``) explicitly forbids.
    """
    st = _state(env)
    box = env.scene["box"]
    d = (st.goal_pos[:, :2] - box.data.root_pos_w[:, :2]).norm(dim=-1)
    speed = box.data.root_lin_vel_w[:, :2].norm(dim=-1)
    return inside_radius(d, goal_radius_m) * (speed < v_box_thresh).to(d.dtype)


def heading_progress_combo(env, contact_dist: float = 0.70) -> torch.Tensor:
    """Heading alignment with the goal, gated to only count while in contact.

    Multiplied by the engagement gate so it shapes *how* the robot squares up to
    the push, and is silent while it is still walking over. Reported in [0, 1]: the
    raw cosine is negative when facing away, and a negative reward for merely
    approaching would fight the approach term.
    """
    from isaaclab.utils.math import quat_apply

    st = _state(env)
    robot = env.scene["robot"]
    box = env.scene["box"]
    fwd = quat_apply(robot.data.root_quat_w.torch, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
    to_goal = st.goal_pos[:, :2] - box.data.root_pos_w[:, :2]
    cos = heading_cos(fwd[:, :2], to_goal)
    gap = wrist_box_gap(env).mean(-1)
    return 0.5 * (cos + 1.0) * engagement(gap, contact_dist)


def push_pull_etiquette(env, pull_scale: float = 0.10, standoff_m: float = 0.50) -> torch.Tensor:
    """Penalise dragging the box backwards; reward keeping the wrists off it.

    Two failure modes in one term, both specific to pushing with arms:

    * the wrists hook the box and drag it toward the robot, which shortens the
      box->goal distance at times -- ``pull_fraction`` goes negative and is charged;
    * the robot leans on the box with the arms locked, which stalls progress and
      topples it -- ``standoff_m`` rewards withdrawing the wrists.

    Returns a SIGNED value: positive for clean pushing, negative for either fault.
    """
    st = _state(env)
    box = env.scene["box"]
    to_goal = st.goal_pos[:, :2] - box.data.root_pos_w[:, :2]
    along = pull_fraction(box.data.root_lin_vel_w[:, :2], to_goal)
    standoff = (wrist_box_gap(env).mean(-1) > standoff_m).to(along.dtype)
    return along / pull_scale + standoff


def parked_bonus(env, standoff_m: float = 0.50, standoff_tol: float = 0.07,
                 err_thresh: float | None = None, hold_s: float | None = None) -> torch.Tensor:
    """Dense reward for the actual success event, plus arm withdrawal.

    Deliberately NOT the base-push ``box_inside_and_parked`` shape (box centre inside
    a 0.35 m radius, still, robot at standoff). That termination does not exist in
    this task: ours is ``goal_reached`` -- mean CORNER error under 0.08 m held for
    1.0 s -- because v3 gives the box a goal *pose* (fix A1), so centre-inside-a-
    circle would pay out for a box in the right place at the wrong yaw and would
    disagree with the terminator that actually ends the episode.

    So this mirrors ``goal_reached`` exactly (same tolerance, same hold, reusing the
    ``success_hold`` counter it maintains) and adds the arm-specific part the
    base-push version has no way to express: the wrists must come off the box once
    the pose is right, or the robot is still leaning on it and has not parked it.
    """
    from .push_mdp import SUCCESS_ERR_M, SUCCESS_HOLD_S, goal_corners_base

    err_thresh = SUCCESS_ERR_M if err_thresh is None else err_thresh
    hold_s = SUCCESS_HOLD_S if hold_s is None else hold_s
    st = _state(env)
    corner_err = (box_corners_base(env) - goal_corners_base(env)).norm(dim=-1).mean(-1)
    posed = (corner_err < err_thresh).to(corner_err.dtype)
    held = (getattr(st, "success_hold", torch.zeros_like(corner_err)) >= hold_s).to(corner_err.dtype)
    gap = wrist_box_gap(env).mean(-1)
    withdrawn = inside_radius(gap, standoff_m, smooth=standoff_tol)
    return posed * held * withdrawn