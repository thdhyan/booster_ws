# Copyright (c) 2025-2026, Booster Robotics + thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 Kicking — train a policy to kick a ball forward.

Observation space (48-dim):
  ball_distance(1) + ball_angle(2) + base_lin_vel(3) + base_ang_vel(3) +
  projected_gravity(3) + joint_pos(12) + joint_vel(12) + last_action(12)

Action space (12-dim):
  12 leg joint positions — kick motion

Reward:
  - Ball forward velocity (primary)
  - Ball contact force
  - Stable stance during approach
  - Penalize falls, energy, action rate
"""

from __future__ import annotations
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.core.velocity.mdp as mdp
from booster_train.assets.robots.booster import BOOSTER_K1_CFG

K1_LEG_JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

K1_ARM_HEAD_JOINTS = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
]

K1_ARTICULATION_CFG = BOOSTER_K1_CFG


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class KickingSceneCfg(InteractiveSceneCfg):
    """Scene: K1 on flat ground with ball."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
    )
    robot: ArticulationCfg = K1_ARTICULATION_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Soccer ball — dynamic sphere
    ball = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),  # FIFA ball mass
        ),
    )


# ---------------------------------------------------------------------------
# MDP — Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Kicking observations (48-dim)."""

        # Ball state relative to robot
        ball_distance = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "kick_target"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # Base state
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # Joint state (12 leg joints)
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
        # Last action
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
    """Joint position targets for 12 leg joints."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=K1_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# MDP — Commands
# ---------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """Kick target — ball position and desired kick direction."""
    kick_target = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),  # ball distance forward
            lin_vel_y=(-0.5, 0.5),  # ball lateral offset
            ang_vel_z=(-0.5, 0.5),  # kick direction angle
            heading=(-math.pi / 4, math.pi / 4),
        ),
    )


# ---------------------------------------------------------------------------
# MDP — Rewards
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # Ball forward velocity — primary reward
    ball_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={"command_name": "kick_target", "std": math.sqrt(0.5)},
    )
    # Ball contact — reward foot hitting ball
    ball_contact = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "kick_target",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Robot/left_foot_link", "Robot/right_foot_link"],
            ),
            "threshold": 0.2,
        },
    )
    # Stable stance
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )
    # Penalize falls
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # Regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*"])},
    )
    dof_torques = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])},
    )
    # Keep arms/head still during kick
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_ARM_HEAD_JOINTS)},
    )


# ---------------------------------------------------------------------------
# MDP — Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.illegal_height,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.3},
    )


# ---------------------------------------------------------------------------
# MDP — Events
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 8.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (0.0, 0.0),
                                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.3, 0.3)}},
    )


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
@configclass
class KickingEnvCfg(ManagerBasedRLEnvCfg):
    """Kicking environment for K1."""

    scene: KickingSceneCfg = KickingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurrTerm()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
