# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 P1 BASIC env variant for STUDENT DISTILLATION.

Identical to the teacher env except the deployable "policy" obs group carries a
10-step history buffer (group-level history_length): 42 x 10 = 420-dim, term-major
([gravity x10, ang_vel x10, ...], oldest -> newest) — matches the node-side
deployment buffer (0.2 s at 50 Hz). The privileged "teacher" group (233-dim) is
unchanged and supplies the frozen teacher's observations.
"""

from isaaclab.utils.configclass import configclass

from .basic_env_cfg import K1BasicTeacherEnvCfg

STUDENT_HISTORY_LEN = 10  # 0.2 s at 50 Hz


@configclass
class K1BasicDistillEnvCfg(K1BasicTeacherEnvCfg):
    """Teacher env with history-stacked student observations (420-dim)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = STUDENT_HISTORY_LEN
