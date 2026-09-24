"""P3 head tracking: keep the *detected* ball centred in the head camera.

- Action: 2 head joints (AAHead_yaw, Head_pitch) @ scale 0.5.
- Legs: frozen at the standing default (P1-equivalent static stance).
- Obs (12): detection (visible, du, dv) + head pos/vel (2+2) + base ang vel
  (3) + last action (2).  No true ball pose in the policy group.
- Rewards: centredness of the detection (primary), geometric pointing (GT,
  reward-time), in-frame bonus, smoothness + joint-limit penalties.
- Curriculum: ball roll speed 0 -> 0.8 m/s over iters 200..1200.
- Head camera is mounted on Head_2 (pitch link) so it moves with the head.
"""
from __future__ import annotations

import math

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
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import head_mdp as mdp
try:  # Isaac Lab 3.0-EA layout (dl); isaac-lab image renamed this package
    import isaaclab_tasks.core.velocity.mdp as vmdp
except (ImportError, ModuleNotFoundError):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vmdp
from ..kick.mdp import BALL_MASS, BALL_RADIUS
from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from k1_velocity.sim_backend import apply_physics_backend

K1_HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]


@configclass
class K1HeadTrackSceneCfg(InteractiveSceneCfg):
    """K1 standing on flat ground, one ball, head camera."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.0, BALL_RADIUS + 0.01),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # Head camera on the pitch link (Head_2): moves with the head joints.
    head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Head_2",
        offset=CameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.10),
            rot=(0.5, -0.5, 0.5, -0.5),  # ROS optical, +x forward (wxyz)
        ),
        height=240,
        width=320,
        update_period=20,  # sim steps @200 Hz -> 10 Hz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=52.76,       # ZED 2i-class optics (soccer_sim.py)
            horizontal_aperture=128.0,
            clipping_range=(0.1, 30.0),
        ),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """12-dim deployable obs: detector + head state + base ang vel + action."""

        detection = ObsTerm(func=mdp.ball_detection)  # (visible, du, dv) — no GT
        head_pos = ObsTerm(
            func=vmdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_HEAD_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        head_vel = ObsTerm(
            func=vmdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_HEAD_JOINTS)},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        base_ang_vel = ObsTerm(func=vmdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=vmdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    joint_pos = vmdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=K1_HEAD_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Centredness of the DETECTION is primary; geometric term shapes early."""

    ball_centered = RewTerm(func=mdp.ball_centered, weight=2.0, params={"std": 0.35})
    track_ball_angle = RewTerm(func=mdp.track_ball_angle_exp, weight=1.0, params={"std": 0.35})
    ball_in_frame = RewTerm(func=mdp.ball_in_frame, weight=0.5)
    action_rate_l2 = RewTerm(func=mdp.guarded_action_rate_l2, weight=-0.1)
    joint_pos_limits = RewTerm(
        func=mdp.guarded_joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_HEAD_JOINTS)},
    )
    time_penalty = RewTerm(func=mdp.guarded_time_penalty, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=vmdp.time_out, time_out=True)


@configclass
class EventCfg:
    reset_scene = EventTerm(func=vmdp.reset_scene_to_default, mode="reset")
    reset_ball = EventTerm(func=mdp.reset_ball_for_tracking, mode="reset")
    maintain_ball_speed = EventTerm(
        func=mdp.maintain_ball_velocity,
        mode="interval",
        interval_range_s=(0.4, 0.6),
    )


@configclass
class CurriculumCfg:
    ball_speed = CurrTerm(
        func=mdp.ball_speed_curriculum,
        params={"start_iter": 200, "end_iter": 1200, "v_max": 0.8},
    )


@configclass
class K1HeadTrackEnvCfg(ManagerBasedRLEnvCfg):
    """P3 head tracking env (see module docstring)."""

    scene: K1HeadTrackSceneCfg = K1HeadTrackSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.use_newton_actuators = False
        apply_physics_backend(self)
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
