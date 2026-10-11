# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""Single-robot push with the SIMPLIFIED reward set (``Isaac-Push-Simple-K1-v0``).

Three ideas, nothing else (plus light regularization):

1. hands to the box     -> ``wrist_box_proximity`` (farther wrist, two-scale)
2. box closer to goal   -> ``box_goal_progress_rate`` (metres closed / s) + ``box_goal_near``
                           + a success bonus when the box centre sits on the goal
3. do not fall          -> ``robot_fallen`` termination with a penalty

Why a new reward set: the SG set stood still (see REWARDS_NOTES.md audit). Etiquette paid
+1 for a standoff, fast_approach was farmable, stop_bonus paid at reset, and the only
progress term paid 0.02 x weight per metre because the reward manager scales by dt.

Curriculum: box spawn radius 0.75-0.9 -> 0.9-1.8 m and goal distance 0.3 -> 2.5 m, both
between iterations 200 and 1800. The near face of the smallest start box is ~0.4 m from
the robot, outside the base's 0.35 m stop radius. Success is on box POSITION (0.15 m, 1 s hold,
matching success_bonus); the goal yaw is not part of this task.
"""
from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import push_env_cfg as base
from . import push_mdp as mdp

GOAL_D_MAX = 2.5        # final goal distance from the box (m)
WORKSPACE_R = 2.5       # goals stay within this radius of the env origin
OOB_R = 3.0             # box beyond this = failure (env_spacing 8 keeps cells apart)


@configclass
class K1PushSimpleRewardsCfg:
    # 1. hands to the box (0..2 per s)
    hands_to_box = RewTerm(func=mdp.wrist_box_proximity, weight=0.5)
    # 2. box closer to the goal
    box_progress = RewTerm(func=mdp.box_goal_progress_rate, weight=5.0)   # 5 per metre closed
    box_near_goal = RewTerm(func=mdp.box_goal_near, weight=2.0, params={"std": 0.5})
    success_bonus = RewTerm(func=mdp.success_bonus, weight=500.0)        # x dt = +10 once
    # 3. do not fall (x dt = -10); box knocked over / out of bounds costs the same
    termination_penalty = RewTerm(func=mdp.term_fired, weight=-500.0,
                                  params={"term_names": ["fallen", "box_oob", "box_tip"]})
    # light regularization
    flat_orientation_l2 = RewTerm(func=mdp.vmdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.vmdp.action_rate_l2, weight=-0.005)
    joint_pos_limits = RewTerm(
        func=mdp.vmdp.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(mdp.K1_LEFT_ARM_JOINTS)
                                            + list(mdp.K1_RIGHT_ARM_JOINTS))},
    )


@configclass
class K1PushSimpleTerminationsCfg:
    time_out = DoneTerm(func=mdp.vmdp.time_out, time_out=True)
    fallen = DoneTerm(func=mdp.robot_fallen, params={"min_z": 0.30, "limit_angle": 0.8})
    goal_reached = DoneTerm(func=mdp.goal_reached_centroid, params={"err_thresh": 0.15, "hold_s": 1.0})
    box_oob = DoneTerm(func=mdp.box_out_of_bounds, params={"radius": OOB_R})
    box_tip = DoneTerm(func=mdp.box_tipped, params={"limit_deg": 45.0})


@configclass
class K1PushSimpleCurriculumCfg:
    box_dist = CurrTerm(func=mdp.box_dist_curriculum, params={
        "start_iter": 200, "end_iter": 1800, "r0": (0.75, 0.9), "r1": (0.9, 1.8)})
    goal_dist = CurrTerm(func=mdp.goal_dist_curriculum, params={
        "start_iter": 200, "end_iter": 1800, "d_max": GOAL_D_MAX})


@configclass
class K1PushSimpleEnvCfg(base.K1PushEnvCfg):
    rewards: K1PushSimpleRewardsCfg = K1PushSimpleRewardsCfg()
    terminations: K1PushSimpleTerminationsCfg = K1PushSimpleTerminationsCfg()
    curriculum: K1PushSimpleCurriculumCfg = K1PushSimpleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_box.params = {"workspace_r": WORKSPACE_R}


@configclass
class K1PushSimplePlayEnvCfg(K1PushSimpleEnvCfg):
    """Play at the FINAL curriculum values (start == end)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.teacher.enable_corruption = False
        self.curriculum.box_dist.params["start_iter"] = -2
        self.curriculum.box_dist.params["end_iter"] = -1
        self.curriculum.goal_dist.params["start_iter"] = -2
        self.curriculum.goal_dist.params["end_iter"] = -1
