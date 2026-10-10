# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""Arm-contact push env cfgs using the ported SG reward set.

Registered as a SEPARATE task pair (``Isaac-Push-Reach-SG-K1-v0`` /
``Isaac-Push-SG-K1-v0``) rather than by editing ``push_env_cfg``. The v3
corner-goal reward set is the one every recorded run used; overwriting it would make
the two indistinguishable and there would be no way back. A sibling cfg keeps both
runnable so the reward set can be A/B'd on the same frozen base.

Only the push-shaping block is swapped. Height tracking, commanded-velocity
tracking, success bonus, regularization and the failure penalty are inherited from
``K1PushRewardsCfg`` unchanged -- they are orthogonal to how the box is shaped
toward the goal, and re-tuning them here would confound the comparison.

SIGN CONVENTION
---------------
Every SG term returns a NON-NEGATIVE value except ``push_pull_etiquette``, which is
signed with positive meaning "pushing cleanly". All weights are therefore POSITIVE.
This is stated explicitly because ``push_env_cfg`` documents a past bug from the
opposite convention (negative weights on functions that already return negative
error double-negated, and the policies were paid to keep their wrists away from the
box -- 0% contact in v1 runs). Do not copy weights across from the v3 block without
checking which side of zero the function returns.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass  # NOT "from isaaclab.utils import": that binds the MODULE

from . import push_env_cfg as base
from . import push_rewards_sg as sg


@configclass
class K1PushSGRewardsCfg(base.K1PushRewardsCfg):
    """SG push shaping on top of v3's inherited height/velocity/regularization."""

    # --- retired: the SG terms below replace the corner-goal block -------------
    # The goal is still a POSE (goal_reached scores 8-corner error), and
    # task_box_to_goal_recip tracks the centroid; the SG set simply reaches it
    # through contact and etiquette instead of raw corner error.
    corner_goal_tracking = None
    centroid_goal_tracking = None
    box_goal_progress = None
    box_vel_toward_goal = None

    # --- approach: wrists to the box face -------------------------------------
    # d_min defaults to 0.40 m (an arm's reach) rather than the reference 2.0 m,
    # which is nearly flat over the span an arm actually has to cross.
    approach_object = RewTerm(func=sg.approach_object_recip, weight=2.0)
    fast_approach = RewTerm(func=sg.fast_approach, weight=1.5, params={"contact_dist": 0.60})
    # v3's own two-scale proximity term is kept: the SG approach is a bounded
    # reciprocal and gives no sharp gradient in the last few cm of contact.
    wrist_box_proximity = RewTerm(func=base.mdp.wrist_box_proximity, weight=0.5)

    # --- task progress --------------------------------------------------------
    task_box_goal = RewTerm(func=sg.task_box_to_goal_recip, weight=1.5)
    heading_progress = RewTerm(func=sg.heading_progress_combo, weight=0.6, params={"contact_dist": 0.70})

    # --- settling and manners -------------------------------------------------
    # stop_at_goal_box_still gates on the box being STILL, so it cannot be farmed
    # by a box sweeping through the goal (which goal_reached's 1 s hold also forbids).
    stop_bonus = RewTerm(func=sg.stop_at_goal_box_still, weight=5.0)
    etiquette = RewTerm(func=sg.push_pull_etiquette, weight=1.0)
    parked = RewTerm(func=sg.parked_bonus, weight=0.3)


@configclass
class K1PushReachSGRewardsCfg(K1PushSGRewardsCfg):
    """Reach stage for the SG set: approach only, no push shaping.

    The box is the heavier reach range (6-12 kg) so bumping it cannot slide it to the
    0.3+ m goal, which makes the success machinery inert here and keeps this stage
    purely about walking up and putting the hands on the box face.
    """

    task_box_goal = None
    heading_progress = None
    stop_bonus = None
    etiquette = None
    parked = None
    # contact is the whole point of this stage, so it is worth more than in push
    wrist_box_proximity = RewTerm(func=base.mdp.wrist_box_proximity, weight=1.0)
    approach_object = RewTerm(func=sg.approach_object_recip, weight=3.0)
    fast_approach = RewTerm(func=sg.fast_approach, weight=2.0, params={"contact_dist": 0.60})


@configclass
class K1PushSGEnvCfg(base.K1PushEnvCfg):
    rewards: K1PushSGRewardsCfg = K1PushSGRewardsCfg()


@configclass
class K1PushReachSGEnvCfg(base.K1PushReachEnvCfg):
    rewards: K1PushReachSGRewardsCfg = K1PushReachSGRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # K1PushReachEnvCfg.__post_init__ assigns the v3 reach rewards, which silently
        # replaced the SG set: runs of this task logged no approach_object/fast_approach.
        self.rewards = K1PushReachSGRewardsCfg()

@configclass
class K1PushSGPlayEnvCfg(K1PushSGEnvCfg):
    """Small-env, no-corruption variant for rendering and for zero-actor smoke runs.

    Deliberately does NOT drop the curriculum's siblings: the scene, the frozen-base
    wiring, the per-env ground patches and the 10-dim action layout are all identical to
    the training cfg, so a video recorded here shows the environment the policy will
    actually be trained in rather than a simplified stand-in.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.teacher.enable_corruption = False
        self.curriculum = None


@configclass
class K1PushReachSGPlayEnvCfg(K1PushReachSGEnvCfg):
    """Reach-stage play variant; see K1PushSGPlayEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.teacher.enable_corruption = False
        self.curriculum = None
