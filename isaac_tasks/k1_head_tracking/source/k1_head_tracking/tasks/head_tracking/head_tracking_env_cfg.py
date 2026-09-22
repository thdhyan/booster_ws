# Copyright (c) 2025-2026, Booster Robotics + thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 Head Tracking — train a policy to turn head toward a ball target.

Observation space (11-dim):
  ball_angle_to_head(2) + head_joint_pos(2) + head_joint_vel(2) +
  base_ang_vel(3) + last_action(2)

Action space (2-dim):
  AAHead_yaw, Head_pitch — joint position targets

Reward:
  - Track ball angle with exponential kernel
  - Penalize action rate, joint limits
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from booster_train.assets.robots.booster import BOOSTER_K1_CFG

# Head joints only
K1_HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]

# All joints for full state
K1_ALL_JOINTS = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

K1_ARTICULATION_CFG = BOOSTER_K1_CFG


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class HeadTrackingSceneCfg(InteractiveSceneCfg):
    """Scene: K1 on flat ground with ball target."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
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

    # Ball target — a small sphere that the head should track
    ball = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


# ---------------------------------------------------------------------------
# MDP — Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Head tracking observations (11-dim)."""

        # Ball angle relative to head (yaw, pitch) — from ball position
        ball_angle = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ball_target"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # Head joint state
        head_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_HEAD_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        head_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_HEAD_JOINTS)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        # Base angular velocity (for stabilization)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
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
    """Head joint position targets (2-dim)."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=K1_HEAD_JOINTS,
        scale=0.5,  # head moves more freely
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# MDP — Commands
# ---------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """Ball target command — random angle for head to track."""
    ball_target = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),  # ball distance
            lin_vel_y=(-0.5, 0.5),  # ball lateral offset
            ang_vel_z=(-1.5, 1.5),  # ball yaw angle
            heading=(-math.pi, math.pi),
        ),
    )


# ---------------------------------------------------------------------------
# MDP — Rewards
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # Track ball angle — primary reward
    track_ball_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "ball_target", "std": math.sqrt(0.25)},
    )
    # Penalize head action rate (smooth tracking)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.5,
    )
    # Penalize joint limits
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_HEAD_JOINTS)},
    )
    # Termination penalty
    termination = RewTerm(func=mdp.is_terminated, weight=-100.0)


# ---------------------------------------------------------------------------
# MDP — Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ---------------------------------------------------------------------------
# MDP — Events
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    # Randomize physics
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
    # Push robot occasionally
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0),
                                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.2, 0.2)}},
    )


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
@configclass
class HeadTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Head tracking environment for K1."""

    scene: HeadTrackingSceneCfg = HeadTrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurrTerm()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
