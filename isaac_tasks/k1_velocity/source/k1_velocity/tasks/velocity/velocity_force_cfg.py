# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 **P2f** — velocity locomotion + Isaac Lab's built-in random force/torque shoves.

P2 (``velocity/velocity_env_cfg.py``) shipped without the PLAN §3.2 wrench shove
(it only had velocity impulses ``push_robot`` and ±2 kg mass randomization).
P2f closes that gap with Isaac Lab's **native**
``envs.mdp.apply_external_force_torque`` event (user request 2026-09-23:
"IsaacLab already has those things — research and call them P1f/P2f"):

* ``shove_force_torque`` (interval, 4–8 s per env): random 3-D force (±30 N) +
  torque (±10 N·m) composed onto the Trunk's permanent-wrench buffer, held
  until the next resample — sustained random shoves on top of P2's velocity
  impulses and mass randomization.
* Teacher obs group gains ``shove_wrench`` (6-dim): **teacher knows the shove,
  student doesn't** — the deployable ``policy`` group is untouched.

Runs: P2f teacher ``Isaac-Velocity-Rough-K1-Teacher-F-v0`` (exp
``p2f_move_teacher``) -> P2f student ``Isaac-Velocity-Distill-K1-F-v0`` (exp
``p2f_move_student``).
"""
from __future__ import annotations

from isaaclab.envs.mdp import apply_external_force_torque as _apply_shove_wrench
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.utils.configclass import configclass

from k1_velocity.shove_mdp import applied_shove_wrench

from .velocity_env_cfg import EventCfg as VelocityEventCfg, K1VelocityRoughEnvCfg, ObservationsCfg as VelocityObsCfg
from .velocity_env_distill import K1VelocityDistillEnvCfg

# The single body the P2f shoves act on (fused chest+waist+pelvis link).
K1_SHOVE_BODIES = ["Trunk"]


# ---------------------------------------------------------------------------
# MDP — teacher obs group + shove wrench (privileged; student group untouched)
# ---------------------------------------------------------------------------
@configclass
class ForceTeacherCfg(VelocityObsCfg.TeacherCfg):
    """P2 teacher obs (235) + applied shove wrench (6) = 241."""

    shove_wrench = ObsTerm(
        func=applied_shove_wrench,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=K1_SHOVE_BODIES)},
    )


@configclass
class ForceObservationsCfg(VelocityObsCfg):
    """P2 observations with the force-aware teacher group."""

    teacher = ForceTeacherCfg()


# ---------------------------------------------------------------------------
# Events — P2 set + native Isaac Lab force/torque shoves
# ---------------------------------------------------------------------------
@configclass
class ForceEventCfg(VelocityEventCfg):
    shove_force_torque = EventTerm(
        func=_apply_shove_wrench,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=K1_SHOVE_BODIES),
            "force_range": (-30.0, 30.0),   # per-axis [N], world frame
            "torque_range": (-10.0, 10.0),  # per-axis [N·m], world frame
        },
    )


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------
@configclass
class K1VelocityRoughForceEnvCfg(K1VelocityRoughEnvCfg):
    """P2f teacher env: rough velocity locomotion + native force/torque shoves + shove-aware teacher."""

    observations: ForceObservationsCfg = ForceObservationsCfg()
    events: ForceEventCfg = ForceEventCfg()


@configclass
class K1VelocityDistillForceEnvCfg(K1VelocityDistillEnvCfg):
    """P2f student distillation env: history-stacked policy obs (unchanged) + shove-aware teacher."""

    observations: ForceObservationsCfg = ForceObservationsCfg()
    events: ForceEventCfg = ForceEventCfg()
