# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 hierarchical partial-control velocity locomotion — the base policy.

Base policy for the hierarchical scheme (legs+head policy under a future
upper-body/arm controller):

* **Action (14-dim)**: 12 leg DoF (scale 0.25, as the proven velocity task) +
  2 head DoF (``AAHead_yaw``, ``Head_pitch``, scale 0.5).
* **Arms (8 DoF) are NOT in the action space.** Each reset places them at a
  curriculum-scaled random pose (clamped to soft joint limits) and holds it via
  their delayed-PD actuators. The policy *observes* the resulting arm pose and
  velocity, so the same controller balances and tracks velocity commands under
  variable arm configurations — the precondition for carrying/pushing with
  freely-moving arms later.
* **Curriculum** (``arm_pose``): small perturbations (±0.15 rad) first → full
  random arm poses (±1.0 rad) by iteration ~2000. Everything else — rough
  terrain scene, velocity commands, pushes, rewards, terminations, PPO runner —
  is inherited from the proven ``Isaac-Velocity-Rough-K1-v0`` task so the two
  policies stay directly comparable.

Requires the same environment as the velocity task:
  ``pip install -e isaac_tasks/booster_train_ref/source/booster_train``
  ``pip install -e isaac_tasks/k1_velocity``
"""
from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

try:  # Isaac Lab 3.0-EA layout (dl/laptop); the isaac-lab image renamed this package
    import isaaclab_tasks.core.velocity.mdp as mdp
except (ImportError, ModuleNotFoundError):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from k1_velocity.tasks.partial import mdp as k1_mdp
from k1_velocity.tasks.velocity.velocity_env_cfg import (
    CurriculumCfg as VelocityCurriculumCfg,
    EventCfg as VelocityEventCfg,
    K1VelocityRoughEnvCfg,
    RewardsCfg as VelocityRewardsCfg,
    K1_ARM_HEAD_JOINTS,
    K1_LEG_JOINTS,
)

# K1 head joints — the 2 extra action DoFs (gaze) the policy owns.
K1_HEAD_JOINTS = K1_ARM_HEAD_JOINTS[:2]  # AAHead_yaw, Head_pitch
# K1 arm joints — outside the action space; randomized + PD-held each episode.
K1_ARM_JOINTS = K1_ARM_HEAD_JOINTS[2:]   # 4 DoF per arm


# ---------------------------------------------------------------------------
# MDP — Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Blind proprioceptive observations for the 14-dim partial-control policy.

        Identical to the plain velocity task PLUS the arm/head state: the policy
        must see the current (randomized) arm configuration to balance under it.
        """
        # Base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Commands
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # Leg joints (the policy's 12 controlled DoF)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # Head joints (the policy's 2 extra controlled DoF)
        joint_pos_head = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_HEAD_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # Arm joints (NOT controlled — observed so the policy conditions on the
        # randomized arm pose; joint_pos_rel = offset from the fixed default pose)
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_ARM_JOINTS)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # Last action (14-dim)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# MDP — Actions
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """14-dim: 12 leg position targets (scale 0.25) + 2 head position targets (scale 0.5)."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=K1_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )
    head_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=K1_HEAD_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# MDP — Rewards (velocity task's locked set, minus default-pose arm penalty)
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg(VelocityRewardsCfg):
    # Arms are held at a randomized pose each episode: penalising deviation from
    # the FIXED default pose would fight the randomization. Replaced below with
    # deviation from the episode hold target. (None = term disabled; the reward
    # manager skips None terms.)
    joint_deviation_arms = None
    arm_pose_deviation = RewTerm(
        func=k1_mdp.joint_deviation_from_target_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_ARM_JOINTS)},
    )


# ---------------------------------------------------------------------------
# Events (randomization) — velocity task's set + per-episode arm pose
# ---------------------------------------------------------------------------
@configclass
class EventCfg(VelocityEventCfg):
    # Declared LAST = applied after reset_scene/reset_robot_joints (config field
    # order), so the sampled arm pose survives the generic joint resets.
    arm_pose_random = EventTerm(
        func=k1_mdp.randomize_arm_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=K1_ARM_JOINTS),
            "offset_range": (-1.0, 1.0),
            # overwritten every reset by the arm_pose curriculum term
            "curriculum_scale": 0.15,
        },
    )
    # Mid-episode arm re-target: every ~2-5 s add a curriculum-scaled delta to
    # the CURRENT arm hold target (PD chases it; reward tracks it). Starts at
    # scale 0.0 (zero changes) and is ramped by the arm_delta curriculum term
    # as the model improves (user request 2026-09-23).
    arm_delta_change = EventTerm(
        func=k1_mdp.randomize_arm_pose_delta,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=K1_ARM_JOINTS),
            "offset_range": (-1.0, 1.0),
            # overwritten every curriculum step by the arm_delta term
            "curriculum_scale": 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Curriculum — velocity task's terrain ramp + arm-pose ramp
# ---------------------------------------------------------------------------
@configclass
class CurriculumCfg(VelocityCurriculumCfg):
    arm_pose = CurrTerm(
        func=k1_mdp.arm_pose_curriculum,
        params={
            "event_term": "arm_pose_random",
            "start_scale": 0.15,   # small perturbations (~±0.15 rad)
            "end_scale": 1.0,      # full random poses (~±1.0 rad, clamped)
            "ramp_start_iter": 300,
            "ramp_end_iter": 2000,
            "steps_per_iter": 24,  # runner's num_steps_per_env
        },
    )
    # Mid-episode arm DELTA changes: zero (0.0) until iteration ~600, then ramp
    # linearly to full ±1.0 rad re-targets by ~iteration 2500 (staggered after
    # the reset-pose ramp so the policy first masters static random poses,
    # then learns to cope with arms that move WHILE walking). Same ramp helper.
    arm_delta = CurrTerm(
        func=k1_mdp.arm_pose_curriculum,
        params={
            "event_term": "arm_delta_change",
            "start_scale": 0.0,    # starts with ZERO mid-episode changes
            "end_scale": 1.0,      # full ±1.0 rad deltas (clamped), by iter ~2500
            "ramp_start_iter": 600,
            "ramp_end_iter": 2500,
            "steps_per_iter": 24,  # runner's num_steps_per_env
        },
    )


# ---------------------------------------------------------------------------
# Main env config
# ---------------------------------------------------------------------------
@configclass
class K1PartialCtrlEnvCfg(K1VelocityRoughEnvCfg):
    """K1 velocity locomotion with legs+head actions and randomized held arms.

    Inherits scene / commands / terminations / sim settings from
    :class:`K1VelocityRoughEnvCfg` (PhysX backend, 200 Hz physics, 50 Hz
    control, 20 s episodes, rough-terrain curriculum, pushes, mass rand).
    """

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # physics backend, dt/decimation, episode length, scanner period — all
        # identical to the inherited velocity task
        super().__post_init__()
