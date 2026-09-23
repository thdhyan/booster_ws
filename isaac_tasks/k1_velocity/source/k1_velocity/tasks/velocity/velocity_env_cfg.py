# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Modifications for Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""
K1 velocity locomotion — rough terrain training environment.
Ported from:
  isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py

Key differences from G1:
  - Robot: Booster K1 (22 DoF, 12 leg joints for policy)
  - Asset: BOOSTER_K1_CFG from booster_train (UrdfFileCfg, real actuator models)
  - K1 init height: 0.57m (from booster_train)
  - Real PD gains per joint from booster_train actuator specs

Requires:
  pip install -e isaac_tasks/booster_train_ref/source/booster_train
  pip install -e isaac_tasks/k1_velocity
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
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

try:  # Isaac Lab 3.0-EA layout (dl); isaac-lab image renamed this package
    import isaaclab_tasks.core.velocity.mdp as mdp
except (ImportError, ModuleNotFoundError):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Use the real K1 articulation config from booster_train (correct actuators, URDF path, PD gains)
from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from k1_velocity.sim_backend import apply_physics_backend

# ---------------------------------------------------------------------------
# K1 joint name constants
# ---------------------------------------------------------------------------
# 12 leg joints controlled by locomotion policy
K1_LEG_JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

# 10 arm+head joints — penalise deviation from default, regulated separately.
# NOTE: exact URDF names — Isaac Sim 6 importer preserves them verbatim:
# head yaw is 'AAHead_yaw', shoulder pitch joints carry an 'A' prefix.
K1_ARM_HEAD_JOINTS = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
]

# K1 articulation — imported from booster_train (real actuators, K1_22dof.urdf, init_pos z=0.57)
K1_ARTICULATION_CFG = BOOSTER_K1_CFG


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class K1RoughSceneCfg(InteractiveSceneCfg):
    """Scene: K1 on rough terrain."""

    # Rough terrain: low-angle slopes + random bumps only (no stairs).
    # Blind policy — no raycaster, proprioception only.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(10.0, 10.0),
            border_width=20.0,
            num_rows=6,
            num_cols=12,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains={
                # 40% flat — for curriculum start
                "flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.4,
                    noise_range=(0.0, 0.01),
                    noise_step=0.005,
                    border_width=0.25,
                ),
                # 40% mild rough — bumps up to ±4 cm
                "rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.4,
                    noise_range=(-0.04, 0.04),
                    noise_step=0.01,
                    border_width=0.25,
                ),
                # 20% low-slope — max 12° incline
                "slope": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                    proportion=0.2,
                    num_obstacles=5,
                    obstacle_height_mode="fixed",
                    obstacle_height_range=(0.0, 0.05),
                    obstacle_width_range=(0.5, 1.5),
                    platform_width=2.0,
                    border_width=0.25,
                ),
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    robot: ArticulationCfg = K1_ARTICULATION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Height scanner for the PRIVILEGED (teacher) observation group. The blind
    # student policy never sees it; the teacher uses it during PPO training and
    # the student recovers the behavior via distillation.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 17x11 = 187 pts
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Contact sensor for foot forces (feet_air_time, feet_slide rewards).
    # Fixed via scripts/flatten_k1_usd.py which applies PhysxContactReportAPI to all rigid bodies.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=["/World/ground"],
    )


# ---------------------------------------------------------------------------
# MDP — Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for locomotion policy (72-dim per step)."""
        # Base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Commands
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # Joint state (12 leg joints only)
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
        # No height scan — blind policy, proprioception only
        # Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged observations for the TEACHER policy (PPO training only).

        Noise-free proprioception + terrain height scan. Foot contact forces are now
        available via scripts/flatten_k1_usd.py which applies PhysxContactReportAPI to
        all rigid bodies, enabling contact sensors to see nested feet.
        """
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
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

        def __post_init__(self):
            self.enable_corruption = False  # privileged: no sensor noise
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


# ---------------------------------------------------------------------------
# MDP — Actions
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """Joint position targets for 12 leg joints (offsets from default)."""
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
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


# ---------------------------------------------------------------------------
# MDP — Rewards
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # Velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Gait shaping via contacts — contact sensors re-enabled via flatten_k1_usd.py
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
        },
    )
    # Termination
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # Regularization
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_.*"])},
    )
    # Arm/head joint deviation — keep near default
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
    # Contact-free fall detection (height-based, not contact-based).
    # K1 trunk stands at ~0.57 m; 0.35 m means the robot has collapsed.
    # Contact-based termination is now possible via contact_forces sensor.
    root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35},
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},
    )


# ---------------------------------------------------------------------------
# MDP — Curriculum
# ---------------------------------------------------------------------------
@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel,
                               params={"asset_cfg": SceneEntityCfg("robot")})


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
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
                "mass_distribution_params": (-2.0, 2.0),
                "operation": "add"},
    )


# ---------------------------------------------------------------------------
# Main env config
# ---------------------------------------------------------------------------
@configclass
class K1VelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """K1 velocity locomotion — rough terrain training."""

    scene: K1RoughSceneCfg = K1RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # IL 3.0: custom BoosterDelayedPDActuator cannot run through Newton-native
        # actuator authoring (see SimulationCfg.use_newton_actuators) → use the
        # Isaac Lab execution path that supports custom actuator configs.
        self.sim.use_newton_actuators = False
        # Physics backend: Newton/warp by default (K1_PHYSICS=newton|physx).
        apply_physics_backend(self)
        self.sim.dt = 0.005          # 200 Hz physics
        self.decimation = 4          # 50 Hz control
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
