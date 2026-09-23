# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL distillation runner config for the K1 P1 BASIC student (PLAN row 2).

Teacher (frozen): P1 teacher checkpoint, 233-dim privileged obs.
Student (deployable): blind 42-dim obs x 10-step history = 420, MLP (512, 256, 128)
tanh — runs on the robot with joint encoders + IMU only (PLAN §4 P1 student column).
Launch: ``--task Isaac-Basic-Student-K1-v0 --checkpoint <teacher model.pt>``.
"""
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class K1BasicDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner for the K1 P1 BASIC student."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "p1_basic_student"
    run_name = "p1_basic_student"
    logger = "wandb"
    wandb_project = "booster_k1_soccer_hrl"
    wandb_entity = "thakk100-dhyan-home"
    # student <- blind "policy" group (history-stacked by the env cfg),
    # teacher <- privileged "teacher" group (height scan + foot contact/slip)
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}

    student = RslRlMLPModelCfg(
        # input: 42-dim obs x 10-step history = 420
        hidden_dims=[512, 256, 128],
        activation="tanh",
        obs_normalization=False,
    )
    teacher = RslRlMLPModelCfg(
        # input: 42 proprio + 187 height scan + 4 foot contact/slip = 233 (must equal teacher env)
        hidden_dims=[512, 256, 128],
        activation="tanh",
        obs_normalization=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=5.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
    )
