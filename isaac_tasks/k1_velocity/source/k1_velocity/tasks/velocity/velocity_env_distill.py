# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Modifications for Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 velocity env variant for STUDENT DISTILLATION.

Identical to the rough training env except the student "policy" obs group is
wrapped with a 10-step history buffer (ObservationManager group-level
history_length): the group becomes 50 x 10 = 500-dim, laid out term-major
([lin_vel x10, ang_vel x10, ..., last_action x10, phase_clock x10], oldest ->
newest).

The teacher group (privileged, 237-dim: P2's 235 plus the 2-dim phase clock)
is otherwise unchanged.
"""

from isaaclab.utils.configclass import configclass

from .squat_env_cfg import K1VelocitySquatEnvCfg
from .velocity_env_cfg import K1VelocityRoughEnvCfg

STUDENT_HISTORY_LEN = 10  # 0.2 s at 50 Hz — matches deployment node buffer


@configclass
class K1VelocityDistillEnvCfg(K1VelocityRoughEnvCfg):
    """Rough env with history-stacked student observations (500-dim)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = STUDENT_HISTORY_LEN
        # history stacking must see clean measurements — corruption is applied
        # per-term before stacking, keep it (matches teacher-free deployment
        # noise); disable only if distillation proves unstable.


@configclass
class K1VelocitySquatDistillEnvCfg(K1VelocitySquatEnvCfg):
    """SQUAT teacher -> deployable student, with the same 10-step history stacking.

    WHY THIS NEEDS ITS OWN TASK
    The teacher's input width is set by its env, not by the runner cfg. Distilling the
    squat teacher through Isaac-Velocity-Distill-K1-v0 fails at the first policy call:

        size mismatch for mlp.0.weight: checkpoint [512, 238], model [512, 237]

    The squat teacher is 238-dim because the squat command adds H* (commanded trunk
    height) to the observation; the plain velocity teacher is 237. Feeding a squat
    checkpoint to the velocity distill env is not a config nicety -- it is a shape error,
    and it is the same class of mismatch as the 236/238 one in push_mdp.

    Everything else is inherited from the squat cfg, so the student is distilled against
    exactly the terrain, commands, rewards and terminations its teacher was trained on.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = STUDENT_HISTORY_LEN


@configclass
class K1VelocitySquatDistillPlayEnvCfg(K1VelocitySquatDistillEnvCfg):
    """Play/render variant of the squat distillation env, so the STUDENT can be shown.

    There was no way to render the squat student at all: the only squat distill task is
    the training one, which carries observation corruption and a full env count. The
    question "what does the student actually do" was unanswerable without this.

    The history stacking is inherited untouched -- the student's 500-dim input is 10
    stacked 50-dim frames, so disabling corruption here does NOT change the student's
    input width. Only the training-time sensor noise is removed, which is what makes the
    clip legible.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
