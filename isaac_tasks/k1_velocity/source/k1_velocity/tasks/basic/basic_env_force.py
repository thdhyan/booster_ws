# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 **P1f** — P1 BASIC + Isaac Lab's built-in random force/torque shoves.

Diff vs P1 (identical for the teacher and student variants — they must share
obs groups for distillation):

* ``shove_force_torque`` event — Isaac Lab **native**
  ``envs.mdp.apply_external_force_torque`` (interval mode, 4–8 s per env):
  samples a random 3-D force (±30 N) and torque (±10 N·m) on the K1 Trunk and
  *sets* them into the permanent-wrench composer, where they are applied every
  physics step until the next resample — sustained random shoves, complementary
  to P1's velocity impulses (``push_robot``) and ±2 kg mass randomization.
  (User request 2026-09-23: use Isaac Lab's built-in force/torque events.)
* The custom ``random_body_push`` impulse shove is **disabled** here: it drives
  the SAME permanent-wrench buffer — its ``composer.reset()`` would clobber the
  native event's sustained wrench mid-shove (and vice versa). One wrench
  authority per task: P1 keeps the custom impulse, P1f uses the native event.
* Teacher obs group gains ``shove_wrench`` (6-dim applied force+torque on the
  Trunk): **teacher knows the shove, student doesn't** — the ``policy`` group is
  untouched, so the deployable student stays blind (distillation transfers the
  anticipation).

Runs: P1f teacher ``Isaac-Basic-Teacher-K1-F-v0`` (exp ``p1f_basic_teacher``)
-> P1f student ``Isaac-Basic-Student-K1-F-v0`` (exp ``p1f_basic_student``).
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

from .basic_env_cfg import EventCfg as BasicEventCfg, K1BasicTeacherEnvCfg, ObservationsCfg as BasicObsCfg
from .basic_env_student import K1BasicDistillEnvCfg

# The single body the P1f shoves act on (fused chest+waaist+pelvis link).
K1_SHOVE_BODIES = ["Trunk"]


# ---------------------------------------------------------------------------
# MDP — teacher obs group + shove wrench (privileged; student group untouched)
# ---------------------------------------------------------------------------
@configclass
class ForceTeacherCfg(BasicObsCfg.TeacherCfg):
    """P1 teacher obs (233) + applied shove wrench (6) = 239."""

    shove_wrench = ObsTerm(
        func=applied_shove_wrench,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=K1_SHOVE_BODIES)},
    )


@configclass
class ForceObservationsCfg(BasicObsCfg):
    """P1 observations with the force-aware teacher group."""

    teacher = ForceTeacherCfg()


# ---------------------------------------------------------------------------
# Events — P1 set with the custom impulse shove swapped for Isaac Lab's native
# force/torque event (same permanent-wrench buffer; see module docstring).
# ---------------------------------------------------------------------------
@configclass
class ForceEventCfg(BasicEventCfg):
    random_body_push = None  # replaced: buffer conflict with the native wrench event
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
class K1BasicTeacherForceEnvCfg(K1BasicTeacherEnvCfg):
    """P1f teacher env: P1 + native force/torque shoves + shove-aware teacher."""

    observations: ForceObservationsCfg = ForceObservationsCfg()
    events: ForceEventCfg = ForceEventCfg()


@configclass
class K1BasicDistillForceEnvCfg(K1BasicDistillEnvCfg):
    """P1f student distillation env: history-stacked policy obs (unchanged) + shove-aware teacher."""

    observations: ForceObservationsCfg = ForceObservationsCfg()
    events: ForceEventCfg = ForceEventCfg()
