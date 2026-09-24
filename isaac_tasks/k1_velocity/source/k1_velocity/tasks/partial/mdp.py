# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""Custom MDP terms for the K1 hierarchical partial-control velocity task.

The policy owns a **14-dim action** (12 leg + 2 head DoF). The 8 arm joints are
deliberately *outside* the action space: on every reset they are placed at a
curriculum-scaled random pose and held there by their delayed-PD actuators, so
the base policy must balance and track velocity commands under **variable arm
configurations** — the precondition for putting an upper-body/arm controller on
top later (carrying, pushing) without retraining locomotion.

Standard terms (velocity tracking, feet, terminations, terrain curriculum) come
from ``isaaclab_tasks.core.velocity.mdp`` (or the ``manager_based`` fallback on
older layouts) and are referenced there directly by the env cfg; only the
arm-pose machinery lives here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    # NOTE: keep these under TYPE_CHECKING — importing the runtime classes eagerly
    # pulls pxr into sys.modules before SimulationApp starts, which breaks Kit's
    # extension loading at launch (same gotcha as k1_velocity/tasks/basic/mdp.py).
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Arm-pose randomization (reset event)
# ---------------------------------------------------------------------------
def randomize_arm_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    offset_range: tuple[float, float] = (-1.0, 1.0),
    curriculum_scale: float = 1.0,
):
    """Place the arms at a curriculum-scaled random pose and hold it for the episode.

    ``mode="reset"`` event, declared AFTER ``reset_scene``/``reset_robot_joints``
    (config field order), so the sampled pose survives the generic joint resets.

    Sampled pose = ``default_joint_pos + curriculum_scale * U(offset_range)``,
    clamped to the soft joint limits (same clamp as ``reset_joints_by_scale``).
    ``curriculum_scale`` is written each reset by the ``arm_pose`` curriculum
    term (0.15 → 1.0 over training), i.e. small perturbations first, full random
    arm poses later.

    Hold mechanism: the arms are driven by no action term, so their entries in
    the articulation joint-position-target buffer persist across steps — writing
    the target once here is exactly what the delayed-PD arm actuators track for
    the whole episode (same mechanism the plain velocity task uses to keep arms
    at default, but re-randomized per episode).

    All params must appear in this signature: the event manager invokes class/
    function terms as ``func(env, env_ids, **cfg.params)`` and statically checks
    the signature against ``cfg.params``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    # broadcast dims: (num_reset_envs, num_arm_joints) — mirrors reset_joints_by_scale
    iter_env_ids = env_ids[:, None] if joint_ids != slice(None) else env_ids

    default = asset.data.default_joint_pos.torch[iter_env_ids, joint_ids]
    lo, hi = offset_range
    offsets = math_utils.sample_uniform(lo, hi, default.shape, default.device) * curriculum_scale
    arm_pos = default + offsets
    # clamp to soft joint limits
    limits = asset.data.soft_joint_pos_limits.torch[iter_env_ids, joint_ids]
    arm_pos = arm_pos.clamp_(limits[..., 0], limits[..., 1])

    # reset joint state ...
    asset.write_joint_position_to_sim_index(position=arm_pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(arm_pos), joint_ids=joint_ids, env_ids=env_ids
    )
    # ... and make that pose the episode's PD hold target (arms have no action term)
    asset.set_joint_position_target_index(target=arm_pos, joint_ids=joint_ids, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Arm-pose delta randomization (interval event — mid-episode re-shuffles)
# ---------------------------------------------------------------------------
def randomize_arm_pose_delta(
    env: ManagerBasedEnv,
    env_ids,
    asset_cfg: SceneEntityCfg,
    offset_range: tuple[float, float] = (-1.0, 1.0),
    curriculum_scale: float = 0.0,
):
    """Add a curriculum-scaled random delta to the CURRENT arm PD hold target.

    ``mode="interval"`` event (~2–5 s): every triggered env shifts each arm
    joint's *existing* ``joint_pos_target`` by
    ``curriculum_scale * U(offset_range)`` (clamped to the soft joint limits).
    The delayed-PD actuators chase the new target, so the arms glide to a new
    pose mid-episode while the policy is walking — and
    ``arm_pose_deviation`` (reward) automatically tracks the moved target.

    Unlike :func:`randomize_arm_pose` (reset-time re-pose), this does NOT touch
    the joint state — it only re-targets, keeping the disturbance smooth.

    ``curriculum_scale`` starts at **0.0** (no-ops — nothing moves) and is
    ramped by the ``arm_delta`` curriculum term as training progresses.

    All params must appear in the signature: the event manager invokes terms as
    ``func(env, env_ids, **cfg.params)`` and statically checks the signature.
    """
    if curriculum_scale == 0.0:
        return  # zero-delta phase: nothing changes at all
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device)

    # (num_triggered_envs, num_arm_joints) — current hold targets
    cur = asset.data.joint_pos_target.torch[env_ids][:, joint_ids]
    lo, hi = offset_range
    delta = math_utils.sample_uniform(lo, hi, cur.shape, cur.device) * curriculum_scale
    new_target = cur + delta
    # clamp to soft joint limits (same clamp as randomize_arm_pose)
    limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, joint_ids]
    new_target = new_target.clamp_(limits[..., 0], limits[..., 1])

    asset.set_joint_position_target_index(target=new_target, joint_ids=joint_ids, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Arm-pose hold reward
# ---------------------------------------------------------------------------
def joint_deviation_from_target_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sum |joint_pos - joint_pos_target| over the listed joints, per env (weight ~ -0.05).

    Unlike ``mdp.joint_deviation_l1`` (deviation from the FIXED default pose) this
    measures deviation from the *episode hold target* written by
    :func:`randomize_arm_pose` — it keeps the arms wherever they were randomized
    to instead of dragging them back to the default pose.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    # (num_instances, num_joints) in global joint order; deprecated alias of
    # actuators.target_command.position, identical on the beta2 image and EA.
    target = asset.data.joint_pos_target.torch[:, asset_cfg.joint_ids]
    return torch.abs(pos - target).sum(dim=-1)


# ---------------------------------------------------------------------------
# Curriculum: small arm perturbations -> full random arm poses
# ---------------------------------------------------------------------------
def arm_pose_curriculum(
    env: ManagerBasedRLEnv,
    env_ids,
    event_term: str = "arm_pose_random",
    start_scale: float = 0.15,
    end_scale: float = 1.0,
    ramp_start_iter: int = 300,
    ramp_end_iter: int = 2000,
    steps_per_iter: int = 24,
) -> float:
    """Ramp the arm-pose randomization range from small perturbations to full poses.

    Progress is measured in *training iterations* (global env steps /
    ``steps_per_iter``, where 24 = the runner's ``num_steps_per_env``): the first
    ``ramp_start_iter`` iterations train with ±0.15 rad perturbations around the
    default arm pose, then the range linearly opens to ``end_scale`` (±1.0 rad,
    clamped to joint limits) by ``ramp_end_iter`` — after that the policy just
    balances and follows velocity commands at fully random arm poses.

    The CurriculumManager runs this from ``_reset_idx`` BEFORE the reset events,
    and the event manager forwards live ``cfg.params`` on every apply — so
    writing the scale into the event term's params here takes effect on the very
    resets that follow (including the current batch).

    Returns the current scale (logged by the CurriculumManager as
    ``Curriculum/arm_pose``).
    """
    iteration = (env.sim.get_physics_step_count() // env.cfg.decimation) / steps_per_iter
    span = max(ramp_end_iter - ramp_start_iter, 1)
    frac = min(max((iteration - ramp_start_iter) / span, 0.0), 1.0)
    scale = start_scale + (end_scale - start_scale) * frac
    env.event_manager.get_term_cfg(event_term).params["curriculum_scale"] = scale
    return scale
