# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 P1 BASIC — stand / balance teacher environment (PLAN §4 P1).

Teacher (privileged) obs ≈ 233: base 42 (gravity 3 + ang_vel 3 + joint_pos 12 +
joint_vel 12 + last_action 12, noise-free) + height scan 187 + foot contact/slip 4.
Student (deployable, distill run): the same 42-dim "policy" group, blind, K10 history.

Rewards are the LOCKED 9-term P1 table (PLAN §4) — do not alter weights without
re-approval. No velocity commands and no terrain curriculum (stand task); rough
terrain comes from the same generator as P2 with level mixing at reset.
"""
from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_visualizers.kit import KitVisualizerCfg

import isaaclab_tasks.core.velocity.mdp as mdp

from k1_velocity.tasks.basic.mdp import foot_contact_slip, still_ang_vel, still_lin_vel
from k1_velocity.tasks.basic.mdp import random_body_push as _random_body_push
from k1_velocity.tasks.velocity.velocity_env_cfg import K1RoughSceneCfg, K1_LEG_JOINTS
from k1_velocity.sim_backend import apply_physics_backend

# Feet (exact URDF link names — bare, no Robot/ prefix, IL 3.0 SceneEntityCfg contract)
K1_FEET = ["left_foot_link", "right_foot_link"]


# ---------------------------------------------------------------------------
# Video (PLAN §3.5) — headless Kit viewport recording of env_0
# ---------------------------------------------------------------------------
# NOTE: recording deliberately uses the EA-native *visualizer* source, not scene
# camera sensors.  IL 3.0's ``Camera`` requires one prim per environment, so a
# scene-camera recorder forces num_envs render products — at 256 envs that added
# ~1-1.5 GB VRAM and OOM'd the 8 GB GPU at the first PPO update.  The headless
# Kit visualizer renders the single viewport **on demand** at capture time
# (``render_rgb_array``), frames env_0 via ``origin_type="env"``, and its cfg
# module is pxr-free (kit_visualizer.py's module-level ``from pxr import ...``
# only loads when the visualizer instantiates, i.e. after SimulationApp starts).
# ``sim.visualizer_cfgs`` pre-set below also skips the launcher's auto-injection.
# PLAN §3.5's "4 dedicated low-res video envs" is superseded by this: one framed
# view of env_0, zero per-env camera cost.


# ---------------------------------------------------------------------------
# MDP — Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable student observations — 42-dim blind proprioception.

        gravity 3 + ang_vel 3 + joint_pos 12 + joint_vel 12 + last_action 12 (PLAN §4 P1).
        No base lin vel, no commands, no height scan.
        """
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
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
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged teacher observations (PPO run) — 42 clean + 187 scan + 4 foot = 233."""
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        foot_contact_slip = ObsTerm(
            func=foot_contact_slip,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=K1_FEET),
                "asset_cfg": SceneEntityCfg("robot", body_names=K1_FEET),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False  # privileged: no sensor noise
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


# ---------------------------------------------------------------------------
# MDP — Actions (12 leg joint position targets, same as P2)
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=K1_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# MDP — Rewards (LOCKED P1 table, PLAN §4 — 9 terms)
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # R1 stand upright
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # R2/R3 stay still
    still_lin_vel = RewTerm(func=still_lin_vel, weight=-1.0)
    still_ang_vel = RewTerm(func=still_ang_vel, weight=-0.5)
    # R4 hold the default stance (L1 over the 12 leg joints)
    joint_deviation_default = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
    )
    # R5 smooth actions
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # R6/R7 regularization
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
    )
    # R8 no foot skating
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=K1_FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=K1_FEET),
        },
    )
    # R9 fall penalty (h<0.35 / tilt>0.8 via TerminationsCfg)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


# ---------------------------------------------------------------------------
# MDP — Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35},
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},
    )


# ---------------------------------------------------------------------------
# Events (randomization)
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    # Auto-stability shove (PLAN §3.2) — force/duration/interval scheduling lives in
    # the term itself; the manager interval is just its 100 Hz tick.
    random_body_push = EventTerm(
        func=_random_body_push,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={
            # K1 has one fused Trunk (chest+waist+pelvis); upper legs = Hip_Pitch links
            # (URDF chain: Hip_Yaw -> Hip_Roll -> Hip_Pitch -> Shank).
            "body_names": ["Trunk", "Left_Hip_Pitch", "Right_Hip_Pitch"],
            "force_range": (20.0, 80.0),
            "duration_range": (0.05, 0.15),
            "interval_range": (3.0, 8.0),
            "active_env_fraction": 0.6,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
                "mass_distribution_params": (-2.0, 2.0),
                "operation": "add"},
    )


# ---------------------------------------------------------------------------
# Main env config (no commands, no curriculum — stand task, like the kick env)
# ---------------------------------------------------------------------------
@configclass
class K1BasicTeacherEnvCfg(ManagerBasedRLEnvCfg):
    """K1 P1 BASIC stand/balance — rough terrain teacher training."""

    scene: K1RoughSceneCfg = K1RoughSceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        # IL 3.0: custom BoosterDelayedPDActuator needs the Isaac Lab execution path
        # (see SimulationCfg.use_newton_actuators).
        self.sim.use_newton_actuators = False
        # Physics backend: Newton/warp by default (K1_PHYSICS=newton|physx).
        apply_physics_backend(self)
        self.sim.dt = 0.005          # 200 Hz physics
        self.decimation = 4          # 50 Hz control
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # Video (PLAN §3.5): pre-declare the headless Kit visualizer so the CLI's
        # pre_launch_video_config skips its auto-injection (our cfg is concrete) and
        # apply_video_recording wires the --video recorder with source='visualizer'.
        # Framing: env_0's robot, front-right ~3.2 m, look-at base height 0.5 m.
        # Defaults would frame the world origin; envs sit on random terrain tiles.
        self.sim.visualizer_cfgs = [
            KitVisualizerCfg(
                headless=True,
                origin_type="env",
                origin_env_index=0,
                eye=(2.2, -2.2, 1.3),
                lookat=(0.0, 0.0, 0.5),
                focal_length=17.0,      # ~70 deg hfov (matches the earlier sensor cam)
                window_width=640,       # low-res clip per PLAN §3.5
                window_height=360,
            )
        ]
