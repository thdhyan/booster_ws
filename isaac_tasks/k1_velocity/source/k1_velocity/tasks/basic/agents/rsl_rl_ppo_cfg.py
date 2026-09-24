# Copyright (c) 2026, The Isaac Lab Project Developers. Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL PPO runner config for the K1 P1 BASIC teacher (PLAN §4 P1, run table row 1).

Runs on the privileged "teacher" obs group (42 noise-free proprio + 187 height scan
+ 4 foot contact/slip = 233). The checkpoint feeds the P1 student distillation run.
Net spec per PLAN: MLP (512, 256, 128), tanh.
"""
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class K1BasicPPOTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO training config for the P1 BASIC teacher."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "p1_basic_teacher"
    run_name = "p1_basic_teacher"
    logger = "wandb"
    wandb_project = "booster_k1_soccer_hrl"
    # Verified via `wandb.Api().default_entity` — 'thakk100'/'thdhyan' are not valid entities.
    wandb_entity = "thakk100-dhyan-home"
    # Privileged teacher obs feeds both actor and critic.
    obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="tanh",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="tanh",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class K1BasicPPOTeacherForceRunnerCfg(K1BasicPPOTeacherRunnerCfg):
    """PPO runner for the **P1f** teacher: P1 hyperparams + shove-aware teacher obs (239-dim).

    Experiment renamed so runs land under ``logs/rsl_rl/p1f_basic_teacher``.
    """

    experiment_name = "p1f_basic_teacher"
    run_name = "p1f_basic_teacher"
