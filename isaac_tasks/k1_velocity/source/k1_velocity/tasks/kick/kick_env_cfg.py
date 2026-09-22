"""K1 kick task environment configuration.

Scene: K1 robot, soccer ball, goal posts/crossbar on flat ground.

The ball is a dynamic RigidObject (not kinematic) with realistic physics.
The goal structure is kinematic (for future contact-based scoring; currently
unused in reward logic). Observations include ball pose and velocity in robot
frame, enabling a blind proprioceptive policy to learn ball manipulation.

Reset curriculum via OmniReset families:
  1. at_ball_shoot (50%): ball 0.3–0.6m in front of one foot
  2. stand_ready (30%): robot 0.6–1.2m behind ball at origin
  3. walk_up (20%): robot 2–3.5m from ball in wide cone

No terrain randomization; focus on ball dynamics and gait adaptation.
"""

from __future__ import annotations
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
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
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

# Kick task MDP (imports ball-specific functions)
from . import mdp

# Use the real K1 articulation config from booster_train (correct actuators, URDF path, PD gains)
from booster_train.assets.robots.booster import BOOSTER_K1_CFG

# ============================================================================
# K1 Joint Constants
# ============================================================================
# 12 leg joints controlled by locomotion policy
K1_LEG_JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

# 10 arm+head joints — penalise deviation from default, regulated separately
K1_ARM_HEAD_JOINTS = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
]

K1_ARTICULATION_CFG = BOOSTER_K1_CFG


# ============================================================================
# Scene: K1 + Ball + Goal Posts/Crossbar
# ============================================================================
@configclass
class K1KickSceneCfg(InteractiveSceneCfg):
    """Scene: K1 on flat ground, soccer ball, kinematic goal structure."""

    # Flat ground (no terrain randomization)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Sky light
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )

    # K1 robot
    robot: ArticulationCfg = K1_ARTICULATION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Soccer ball: dynamic RigidObject with realistic physics
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=sim_utils.SphereCfg(
            radius=mdp.BALL_RADIUS,
            mass_props=sim_utils.MassPropertiesCfg(mass=mdp.BALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.85, 0.1)  # yellow
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, mdp.BALL_RADIUS + 0.01),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # Goal structure (kinematic, visual + collision for future contact detection)
    # Posts and crossbar at +x end (FIELD_L/2 = 4.0m)
    # Positive side goal (at x = FIELD_L/2 = 4.0m)
    goal_post_pos_0 = AssetBaseCfg(
        prim_path="/World/goal_post_pos_0",
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=mdp.GOAL_H,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(mdp.FIELD_L / 2, mdp.GOAL_W / 2, mdp.GOAL_H / 2)),
    )
    goal_post_pos_1 = AssetBaseCfg(
        prim_path="/World/goal_post_pos_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=mdp.GOAL_H,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(mdp.FIELD_L / 2, -mdp.GOAL_W / 2, mdp.GOAL_H / 2)),
    )
    goal_bar_pos = AssetBaseCfg(
        prim_path="/World/goal_bar_pos",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, mdp.GOAL_W + 0.08, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(mdp.FIELD_L / 2, 0.0, mdp.GOAL_H)),
    )

    # Negative side goal (at x = -FIELD_L/2 = -4.0m) — optional, not used in base kick task
    goal_post_neg_0 = AssetBaseCfg(
        prim_path="/World/goal_post_neg_0",
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=mdp.GOAL_H,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-mdp.FIELD_L / 2, mdp.GOAL_W / 2, mdp.GOAL_H / 2)),
    )
    goal_post_neg_1 = AssetBaseCfg(
        prim_path="/World/goal_post_neg_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=mdp.GOAL_H,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-mdp.FIELD_L / 2, -mdp.GOAL_W / 2, mdp.GOAL_H / 2)),
    )
    goal_bar_neg = AssetBaseCfg(
        prim_path="/World/goal_bar_neg",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, mdp.GOAL_W + 0.08, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-mdp.FIELD_L / 2, 0.0, mdp.GOAL_H)),
    )


# ============================================================================
# MDP — Observations
# ============================================================================
@configclass
class ObservationsCfg:
    """Observations for blind K1 kick policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Blind proprioceptive observations (48-dim base + no height scan)."""
        # Base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
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
        # Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged observations for teacher (noise-free + ball state)."""
        # Base state (noise-free)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # Joint state (noise-free)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
        )
        # Last action
        actions = ObsTerm(func=mdp.last_action)
        # Ball state in robot frame (privileged)
        ball_pos = ObsTerm(func=mdp.ball_pos_in_robot_frame)
        ball_lin_vel = ObsTerm(func=mdp.ball_lin_vel_in_robot_frame)

        def __post_init__(self):
            self.enable_corruption = False  # privileged: no sensor noise
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


# ============================================================================
# MDP — Actions
# ============================================================================
@configclass
class ActionsCfg:
    """Joint position targets for 12 leg joints (offsets from default)."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=K1_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


# ============================================================================
# MDP — Rewards
# ============================================================================
@configclass
class RewardsCfg:
    """Rewards: sparse goal_scored + light progress + locomotion regularization."""

    # PRIMARY REWARD: goal scored (sparse, dominant)
    goal_scored = RewTerm(
        func=mdp.goal_scored,
        weight=100.0,
        params={},
    )

    # SECONDARY: ball-to-goal progress (light shaping)
    ball_to_goal_progress = RewTerm(
        func=mdp.ball_to_goal_progress,
        weight=1.0,
        params={},
    )
    # Distance from spawn (encourages ball movement away from start)
    ball_dist_from_spawn = RewTerm(
        func=mdp.ball_dist_from_spawn,
        weight=0.1,
        params={},
    )
    # REGULARIZATION: velocity, action, joint constraints
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_ARM_HEAD_JOINTS)},
    )

    # Termination penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


# ============================================================================
# MDP — Terminations
# ============================================================================
@configclass
class TerminationsCfg:
    """Episode termination conditions."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35},
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},
    )
    goal_scored = DoneTerm(
        func=mdp.goal_scored,
        params={},
    )


# ============================================================================
# Events (randomization & resets)
# ============================================================================
@configclass
class EventCfg:
    """Environment randomization and reset events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    # OmniReset: population-sample ball placement family (shoot / ready / walk-up)
    reset_ball = EventTerm(func=mdp.reset_ball_omnireset, mode="reset")
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


# ============================================================================
# Main Environment Config
# ============================================================================
@configclass
class K1KickEnvCfg(ManagerBasedRLEnvCfg):
    """K1 kick task environment configuration."""

    scene: K1KickSceneCfg = K1KickSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005          # 200 Hz physics
        self.decimation = 4          # 50 Hz control
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
