"""Squat teacher as a RESIDUAL over the frozen velocity student (``Isaac-Velocity-Squat-Residual-K1-v0``).

Same world, commands, observations, rewards (incl. ``feet_flat``), terminations and squat
curriculum as ``K1VelocitySquatEnvCfg``. Only the action changes:

    leg target = default + clip(student(vx, vy, wz) + residual_scale * a, +-1)

The frozen velocity student (``models/k1_velocity_student_gaitfix.pt``, 500 -> 12) walks
from the env's own ``base_velocity`` command (vx, vy, wz, clipped to its training ranges);
the policy sees the full squat teacher obs (238, incl. H*) and adds a 12-dim leg residual
that has to produce the height change. ``residual_scale`` is 0.5 rad: a squat to H* 0.40
needs a large knee/hip offset, which the 0.1 used by track/kick could not reach.
"""
import os

from isaaclab.utils.configclass import configclass

from k1_velocity.tasks.skills.residual_action import FrozenStudentResidualActionCfg

from .squat_env_cfg import K1VelocitySquatEnvCfg, K1VelocitySquatPlayEnvCfg

STUDENT_PATH = os.environ.get("SKILL_STUDENT_POLICY", "models/k1_velocity_student_gaitfix.pt")
RESIDUAL_SCALE = 0.5


@configclass
class SquatResidualActionsCfg:
    # named joint_pos so action-based rewards/obs keep their term names (12 dims, as before)
    joint_pos = FrozenStudentResidualActionCfg(
        student_path=STUDENT_PATH, residual_scale=RESIDUAL_SCALE, head=False, command_name="base_velocity"
    )


@configclass
class K1VelocitySquatResidualEnvCfg(K1VelocitySquatEnvCfg):
    actions: SquatResidualActionsCfg = SquatResidualActionsCfg()


@configclass
class K1VelocitySquatResidualPlayEnvCfg(K1VelocitySquatPlayEnvCfg):
    actions: SquatResidualActionsCfg = SquatResidualActionsCfg()
