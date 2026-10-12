"""Residual skill envs on the frozen velocity student: ball tracking and kicking.

Both share one action term (``skill``: [vx, vy, wz | 12 leg residuals | 2 head]),
see residual_action.py. The student walks; the skill steers it with the command,
nudges the legs with a small residual and aims the head.

* BallTrack: a ball rolls (speed curriculum 0 -> 0.6 m/s, random turns); stay ~0.8 m
  from it, face it, keep it centred in the head camera.
* Kick: kick task scene (ball + goal at +4 m, OmniReset ball families); walk up and
  kick it into the goal. Larger residual authority (0.3) so a swing leg can strike.

Ground friction is the velocity terrain's (1.0/1.0, multiply): the frozen student
trained on it, and the push hunt measured what a 0.5 default plane does to a
frozen base.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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
from isaaclab.utils.configclass import configclass

from k1_velocity.sim_backend import apply_physics_backend, ensure_physx_gpu_capacity
from k1_velocity.tasks.kick import mdp as kmdp
from k1_velocity.tasks.kick.kick_env_cfg import K1KickSceneCfg

from . import skills_mdp as smdp
from .residual_action import K1_HEAD_JOINTS, K1_LEG_JOINTS, FrozenStudentResidualActionCfg

import os

STUDENT_PATH = os.environ.get("SKILL_STUDENT_POLICY", "models/k1_velocity_student_gaitfix.pt")


@configclass
class K1SkillSceneCfg(K1KickSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            )
        ),
    )


@configclass
class SkillActionsCfg:
    # cmd_pad: the walk student has no H* input, so nothing is padded. When the squat
    # student replaces it, set cmd_pad=(0.57,) (standing H*) and nothing else changes.
    skill = FrozenStudentResidualActionCfg(student_path=STUDENT_PATH, residual_scale=0.1, head=True, cmd_pad=())


@configclass
class TrackObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=kmdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=kmdp.base_ang_vel)
        projected_gravity = ObsTerm(func=kmdp.projected_gravity)
        leg_pos = ObsTerm(func=kmdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)})
        leg_vel = ObsTerm(func=kmdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)})
        head_pos = ObsTerm(func=kmdp.joint_pos_rel,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_HEAD_JOINTS, preserve_order=True)})
        actions = ObsTerm(func=kmdp.last_action)
        ball_pos = ObsTerm(func=smdp.ball_pos_in_robot_frame)
        ball_vel = ObsTerm(func=smdp.ball_lin_vel_in_robot_frame)
        head_err = ObsTerm(func=smdp.head_ball_error)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class KickObsCfg(TrackObsCfg):
    @configclass
    class PolicyCfg(TrackObsCfg.PolicyCfg):
        goal_pos = ObsTerm(func=smdp.goal_pos_in_robot_frame)

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommonRewardsCfg:
    # 2026-10-11 simplified set (REWARDS_SIMPLE_PROPOSAL.md): task terms + one fall
    # penalty (x dt = -10) + light regularisation. The frozen student owns the gait.
    action_rate_l2 = RewTerm(func=kmdp.action_rate_l2, weight=-0.01)
    residual_l2 = RewTerm(func=smdp.residual_l2, weight=-0.5)


@configclass
class TrackRewardsCfg(CommonRewardsCfg):
    ball_distance = RewTerm(func=smdp.ball_distance_keep, weight=1.5, params={"target": 0.8, "std": 0.4})
    head_track = RewTerm(func=smdp.track_ball_head_exp, weight=1.0, params={"std": 0.35})
    termination_penalty = RewTerm(func=smdp.term_fired, weight=-500.0,
                                  params={"term_names": ["root_height", "base_orientation", "ball_lost"]})


@configclass
class KickRewardsCfg(CommonRewardsCfg):
    approach_ball = RewTerm(func=kmdp.approach_ball, weight=1.0, params={"min_dist": 0.5})
    ball_progress = RewTerm(func=kmdp.ball_to_goal_progress_rate, weight=5.0)   # 5 per metre closed
    goal_scored = RewTerm(func=kmdp.goal_scored, weight=500.0)                  # x dt = +10 once
    head_track = RewTerm(func=smdp.track_ball_head_exp, weight=0.5, params={"std": 0.35})
    # 2026-10-12: v1 scored 6.6 % with ball_progress < 0 - dense terms for lining up,
    # touching the ball with a foot and sending it goalward
    align_stance = RewTerm(func=kmdp.align_stance, weight=0.5, params={"max_dist": 0.75})
    foot_ball = RewTerm(func=smdp.foot_ball_proximity, weight=1.0, params={"std": 0.3})
    ball_vel_goal = RewTerm(func=smdp.ball_vel_to_goal, weight=2.0, params={"max_speed": 3.0})
    # failures only: the old is_terminated also fired on the goal_scored termination
    termination_penalty = RewTerm(func=smdp.term_fired, weight=-500.0,
                                  params={"term_names": ["root_height", "base_orientation"]})


@configclass
class TrackTerminationsCfg:
    time_out = DoneTerm(func=kmdp.time_out, time_out=True)
    root_height = DoneTerm(func=kmdp.root_height_below_minimum, params={"minimum_height": 0.35})
    base_orientation = DoneTerm(func=kmdp.bad_orientation, params={"limit_angle": 0.8})
    ball_lost = DoneTerm(func=smdp.ball_lost, params={"max_dist": 3.5})


@configclass
class KickTerminationsCfg:
    time_out = DoneTerm(func=kmdp.time_out, time_out=True)
    root_height = DoneTerm(func=kmdp.root_height_below_minimum, params={"minimum_height": 0.35})
    base_orientation = DoneTerm(func=kmdp.bad_orientation, params={"limit_angle": 0.8})
    goal_scored = DoneTerm(func=kmdp.goal_scored)


@configclass
class CommonEventsCfg:
    reset_scene = EventTerm(func=kmdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=kmdp.reset_joints_by_scale, mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=kmdp.push_by_setting_velocity, mode="interval", interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class TrackEventsCfg(CommonEventsCfg):
    reset_base = EventTerm(
        func=smdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-math.pi, math.pi)},
                "velocity_range": {}},
    )
    reset_ball = EventTerm(func=smdp.reset_ball_rolling, mode="reset")
    steer_ball = EventTerm(func=smdp.steer_ball, mode="interval", interval_range_s=(1.0, 3.0))
    keep_ball = EventTerm(func=smdp.keep_ball_in_cell, mode="interval", interval_range_s=(0.5, 0.5))


@configclass
class KickEventsCfg(CommonEventsCfg):
    reset_base = EventTerm(
        func=smdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)},
                "velocity_range": {}},
    )
    reset_ball = EventTerm(func=kmdp.reset_ball_omnireset, mode="reset")


@configclass
class TrackCurriculumCfg:
    ball_speed = CurrTerm(func=smdp.ball_speed_curriculum)


@configclass
class K1BallTrackResidualEnvCfg(ManagerBasedRLEnvCfg):
    scene: K1SkillSceneCfg = K1SkillSceneCfg(num_envs=4096, env_spacing=9.0)
    observations: TrackObsCfg = TrackObsCfg()
    actions: SkillActionsCfg = SkillActionsCfg()
    rewards: TrackRewardsCfg = TrackRewardsCfg()
    terminations: TrackTerminationsCfg = TrackTerminationsCfg()
    events: TrackEventsCfg = TrackEventsCfg()
    curriculum: TrackCurriculumCfg = TrackCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.use_newton_actuators = False
        apply_physics_backend(self)
        ensure_physx_gpu_capacity(self)
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation


@configclass
class K1KickResidualEnvCfg(K1BallTrackResidualEnvCfg):
    observations: KickObsCfg = KickObsCfg()
    rewards: KickRewardsCfg = KickRewardsCfg()
    terminations: KickTerminationsCfg = KickTerminationsCfg()
    events: KickEventsCfg = KickEventsCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 10.0      # goal sits 4 m out: keep neighbours' balls out
        self.actions.skill.residual_scale = 0.3
        self.rewards.residual_l2.weight = -0.2
        self.episode_length_s = 15.0


@configclass
class K1BallTrackResidualPlayEnvCfg(K1BallTrackResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.events.push_robot = None
        # play starts at step 0: give the ball its full roll speed right away
        self.curriculum.ball_speed.params = {"start_step": -1, "end_step": 0}


@configclass
class K1KickResidualPlayEnvCfg(K1KickResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.events.push_robot = None
