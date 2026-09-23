# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Modifications for Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 velocity env variant for STUDENT DISTILLATION.

Identical to the rough training env except the student "policy" obs group is
wrapped with a 10-step history buffer (ObservationManager group-level
history_length): the group becomes 48 x 10 = 480-dim, laid out term-major
([lin_vel x10, ang_vel x10, ..., last_action x10], oldest -> newest).

The teacher group (privileged, 235-dim) is unchanged.
"""

from isaaclab.utils.configclass import configclass

from .velocity_env_cfg import K1VelocityRoughEnvCfg

STUDENT_HISTORY_LEN = 10  # 0.2 s at 50 Hz — matches deployment node buffer


@configclass
class K1VelocityDistillEnvCfg(K1VelocityRoughEnvCfg):
    """Rough env with history-stacked student observations (480-dim)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = STUDENT_HISTORY_LEN
        # history stacking must see clean measurements — corruption is applied
        # per-term before stacking, keep it (matches teacher-free deployment
        # noise); disable only if distillation proves unstable.
