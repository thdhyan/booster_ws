"""RSL-RL PPO runner config for K1 kick task.

Uses the rsl-rl >= 4.0 model-based schema (actor/critic as RslRlMLPModelCfg)
matching Isaac Lab 3.0.
"""
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class K1KickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO training config for K1 kick task."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "p4_kick"
    logger = "wandb"
    wandb_project = "booster_k1_soccer_hrl"
    wandb_entity = "thakk100-dhyan-home"
    # Blind proprioceptive policy: single "policy" obs group feeds both actor and critic.
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
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
class K1KickPPOTeacherRunnerCfg(K1KickPPORunnerCfg):
    """PPO runner for the PRIVILEGED teacher (ball state + noise-free obs).

    Trains on the "teacher" obs group (privileged observations including
    ball pose and velocity). The checkpoint produced is consumed by the
    distillation runner as the frozen teacher.
    """

    experiment_name = "p4_kick_teacher"
    run_name = "p4_chase_kick"
    obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
