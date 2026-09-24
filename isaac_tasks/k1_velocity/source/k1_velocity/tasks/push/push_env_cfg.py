"""P6 box-push envs: frozen locomotion base + wrist IK, corner-goal rewards.

Two ids share this cfg:
  Isaac-Push-Reach-K1-v0 : walk up + extend wrists to contact (heavy box, no
                           push rewards) -> warm-start for the push task.
  Isaac-Push-K1-v0       : full corner-goal pushing (box DR 0.7-1.5x size,
                           3-25 kg, friction 0.3-1.2; cumulative goal walks
                           away 0.3 -> 1.5 m on a curriculum).

Action (9) = velocity override (3) | left wrist EE delta (3) | right (3).
The velocity slice feeds the FROZEN Run-11 partial-control TorchScript policy
(legs+head); the wrist slices feed DifferentialIK position terms whose EE is
the hand link. Obs group is TeacherCfg = fully observable by design (mass,
size, current corners, goal pose + corners, cumulative offset).
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import push_mdp as mdp
from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from k1_velocity.sim_backend import apply_physics_backend


@configclass
class K1PushSceneCfg(InteractiveSceneCfg):
    """K1 + one box; replicate_physics=False for per-env USD DR."""

    replicate_physics = False

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),  # prototype; per-env scale 0.7-1.5x
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.0, 0.5)),
    )


@configclass
class ObservationsCfg:
    @configclass
    class TeacherCfg(ObsGroup):
        """Fully observable (privileged by design): box mass/size/corners/goal
        + goal pose & corners + cumulative offset, wrist targets, proprio."""

        box_state = ObsTerm(func=mdp.push_box_teacher)          # 68
        base_lin_vel = ObsTerm(func=mdp.vmdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.vmdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.vmdp.projected_gravity)
        wrist_targets = ObsTerm(
            func=mdp.vmdp.generated_commands, params={"command_name": "wrist_target"}
        )                                                        # 6
        arm_joint_pos = ObsTerm(
            func=mdp.vmdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.K1_LEFT_ARM_JOINTS + mdp.K1_RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )                                                        # 8
        arm_joint_vel = ObsTerm(
            func=mdp.vmdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.K1_LEFT_ARM_JOINTS + mdp.K1_RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )                                                        # 8
        actions = ObsTerm(func=mdp.vmdp.last_action)             # 9

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


@configclass
class K1PushActionsCfg:
    """9-dim: velocity override (frozen base) + two wrist IK position terms."""

    base_velocity = mdp.FrozenBaseVelocityActionCfg(
        asset_name="robot",
        base_policy_path="models/k1_partialctrl_base.pt",
    )
    wrist_left = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=mdp.K1_LEFT_ARM_JOINTS,
        body_name=mdp.LEFT_EE,
        scale=0.05,  # max 5 cm wrist delta per step
        controller=mdp.DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=True, ik_method="dls",
        ),
    )
    wrist_right = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=mdp.K1_RIGHT_ARM_JOINTS,
        body_name=mdp.RIGHT_EE,
        scale=0.05,
        controller=mdp.DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=True, ik_method="dls",
        ),
    )


@configclass
class K1PushCommandsCfg:
    wrist_target = mdp.WristTargetCommandCfg()


@configclass
class K1PushRewardsCfg:
    """Corner-goal pushing rewards + reach shaping + regularization."""

    # primary: corners -> goal corners (normalized), centroid ("cumulative sum")
    corner_goal_tracking = RewTerm(func=mdp.corner_goal_tracking, weight=-1.0)
    centroid_goal_tracking = RewTerm(func=mdp.centroid_goal_tracking, weight=-0.5)
    box_goal_progress = RewTerm(func=mdp.box_goal_progress, weight=2.0)
    box_vel_toward_goal = RewTerm(func=mdp.box_vel_toward_goal, weight=0.5)
    box_spin_penalty = RewTerm(func=mdp.box_spin_penalty, weight=-0.1)
    # reach/contact shaping
    wrist_target_tracking = RewTerm(func=mdp.wrist_target_tracking, weight=-0.3)
    wrist_box_proximity = RewTerm(func=mdp.wrist_box_proximity, weight=0.5)
    # commanded-velocity tracking (command = action's velocity slice)
    track_cmd_lin_vel = RewTerm(func=mdp.track_cmd_lin_vel_exp, weight=0.5, params={"std": 0.5})
    track_cmd_ang_vel = RewTerm(func=mdp.track_cmd_ang_vel_exp, weight=0.25, params={"std": 0.5})
    # regularization
    flat_orientation_l2 = RewTerm(func=mdp.vmdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.vmdp.action_rate_l2, weight=-0.005)
    dof_torques_l2 = RewTerm(
        func=mdp.vmdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.K1_LEFT_ARM_JOINTS + mdp.K1_RIGHT_ARM_JOINTS)},
    )
    joint_pos_limits = RewTerm(
        func=mdp.vmdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.K1_LEFT_ARM_JOINTS + mdp.K1_RIGHT_ARM_JOINTS)},
    )
    termination_penalty = RewTerm(func=mdp.vmdp.is_terminated, weight=-200.0)


@configclass
class K1PushReachRewardsCfg(K1PushRewardsCfg):
    """Reach stage: no push rewards; box is heavy so bumps do not slide it."""

    corner_goal_tracking = None
    centroid_goal_tracking = None
    box_goal_progress = None
    box_vel_toward_goal = None
    box_spin_penalty = None
    wrist_target_tracking = RewTerm(func=mdp.wrist_target_tracking, weight=-1.0)
    wrist_box_proximity = RewTerm(func=mdp.wrist_box_proximity, weight=1.0)
    track_cmd_lin_vel = RewTerm(func=mdp.track_cmd_lin_vel_exp, weight=0.25, params={"std": 0.5})
    track_cmd_ang_vel = RewTerm(func=mdp.track_cmd_ang_vel_exp, weight=0.1, params={"std": 0.5})


@configclass
class K1PushTerminationsCfg:
    time_out = DoneTerm(func=mdp.vmdp.time_out, time_out=True)
    root_height = DoneTerm(func=mdp.vmdp.root_height_below_minimum, params={"minimum_height": 0.35})
    bad_orientation = DoneTerm(func=mdp.vmdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class K1PushEventCfg:
    # pre-startup USD-level DR (this build has no "usd" event mode; prestartup
    # fires before sim start / PhysX parse)
    randomize_box_geometry = EventTerm(
        func=mdp.randomize_box_geometry,
        mode="prestartup",
        params={"scale_range": (0.7, 1.5), "mass_range": (3.0, 25.0)},
    )
    randomize_friction = EventTerm(
        func=mdp.vmdp.randomize_rigid_body_material,
        # NOT prestartup: the material impls need asset.root_view, which only
        # exists after sim play; startup fires right after play.
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
        },
    )
    green_alpha = EventTerm(func=mdp.apply_box_green_alpha, mode="startup")
    # per-episode
    reset_scene = EventTerm(func=mdp.vmdp.reset_scene_to_default, mode="reset")
    reset_box = EventTerm(func=mdp.reset_box, mode="reset")
    reset_wrist_targets = EventTerm(func=mdp.reset_wrist_targets, mode="reset")
    advance_goal = EventTerm(
        func=mdp.advance_goal, mode="interval", interval_range_s=(0.24, 0.26), params={"dt": 0.25}
    )
    push_robot = EventTerm(
        func=mdp.vmdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class K1PushCurriculumCfg:
    goal_dist = CurrTerm(
        func=mdp.goal_dist_curriculum,
        params={"start_iter": 200, "end_iter": 1800, "d_max": 1.5},
    )


@configclass
class K1PushEnvCfg(ManagerBasedRLEnvCfg):
    """Box-push env (push stage)."""

    scene: K1PushSceneCfg = K1PushSceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: K1PushActionsCfg = K1PushActionsCfg()
    commands: K1PushCommandsCfg = K1PushCommandsCfg()
    rewards: K1PushRewardsCfg = K1PushRewardsCfg()
    terminations: K1PushTerminationsCfg = K1PushTerminationsCfg()
    events: K1PushEventCfg = K1PushEventCfg()
    curriculum: K1PushCurriculumCfg = K1PushCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.use_newton_actuators = False
        apply_physics_backend(self)
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation


@configclass
class K1PushReachEnvCfg(K1PushEnvCfg):
    """Reach stage: same world, heavy box (12-25 kg), reach-only rewards."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards = K1PushReachRewardsCfg()
        self.events.randomize_box_geometry.params["mass_range"] = (12.0, 25.0)
