"""Squat teacher as a RESIDUAL over the frozen velocity student (``Isaac-Velocity-Squat-Residual-K1-v0``).

Same world, commands, observations, terminations and squat curriculum as
``K1VelocitySquatEnvCfg``. The action and the rewards change:

    leg target = default + clip(student(vx, vy, wz) + residual_scale * a, +-1)

The frozen velocity student (``models/k1_velocity_student_gaitfix.pt``, 500 -> 12) walks
from the env's own ``base_velocity`` command (vx, vy, wz, clipped to its training ranges);
the policy sees the full squat teacher obs (238, incl. H*) and adds a 12-dim leg residual
that has to produce the height change. ``residual_scale`` is 0.5 rad: a squat to H* 0.40
needs a large knee/hip offset, which the 0.1 used by track/kick could not reach.

Rewards (2026-10-11, REWARDS_SIMPLE_PROPOSAL.md): 8 terms instead of the teacher's 33.
The frozen student already walks with a clean gait, so the AGILE gait/torque stack is
dropped; ``residual_l2`` keeps the policy near the student's gait.
"""
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from k1_velocity.tasks.skills import skills_mdp as smdp
from k1_velocity.tasks.skills.residual_action import FrozenStudentResidualActionCfg

from . import agile_rewards as agile
from . import gait_rewards as gait
from . import squat_mdp
from .squat_env_cfg import K1VelocitySquatEnvCfg, K1VelocitySquatPlayEnvCfg
from .velocity_env_cfg import mdp

STUDENT_PATH = os.environ.get("SKILL_STUDENT_POLICY", "models/k1_velocity_student_gaitfix.pt")
RESIDUAL_SCALE = 0.5


@configclass
class SquatResidualActionsCfg:
    # named joint_pos so action-based rewards/obs keep their term names (12 dims, as before)
    joint_pos = FrozenStudentResidualActionCfg(
        student_path=STUDENT_PATH, residual_scale=RESIDUAL_SCALE, head=False, command_name="base_velocity"
    )


@configclass
class SquatResidualRewardsCfg:
    track_lin_vel_xy_exp = RewTerm(func=agile.track_lin_vel_xy_exp_weighted, weight=10.0,
                                   params={"command_name": "base_velocity", "std": 0.15})
    track_ang_vel_z_exp = RewTerm(func=agile.track_ang_vel_z_world_exp_weighted, weight=5.0,
                                  params={"command_name": "base_velocity", "std": 0.25})
    # the squat term: trunk at H*
    base_height = RewTerm(func=squat_mdp.base_height_cmd_exp, weight=4.0,
                          params={"command_name": "base_velocity", "std": 0.05, "sensor_name": "height_scanner"})
    feet_flat = RewTerm(func=gait.feet_flat_contact, weight=1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
        "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
        "std": 0.15})
    # do not fall (x dt = -10); no success termination here, so is_terminated is safe
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"])})
    residual_l2 = RewTerm(func=smdp.residual_l2, weight=-0.2, params={"action_name": "joint_pos"})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class K1VelocitySquatResidualEnvCfg(K1VelocitySquatEnvCfg):
    actions: SquatResidualActionsCfg = SquatResidualActionsCfg()
    rewards: SquatResidualRewardsCfg = SquatResidualRewardsCfg()


@configclass
class K1VelocitySquatResidualPlayEnvCfg(K1VelocitySquatPlayEnvCfg):
    actions: SquatResidualActionsCfg = SquatResidualActionsCfg()
    rewards: SquatResidualRewardsCfg = SquatResidualRewardsCfg()
