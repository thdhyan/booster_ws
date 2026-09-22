"""RSL-RL PPO runner config for K1 velocity task.

Uses the rsl-rl >= 4.0 model-based schema (actor/critic as RslRlMLPModelCfg)
matching Isaac Lab 3.0; the old `policy: RslRlPpoActorCriticCfg` field is deprecated.
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class K1VelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO training config for K1 velocity locomotion."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "k1_velocity_rough"
    logger = "wandb"
    wandb_project = "booster_k1_soccer_hrl"
    # Verified via `wandb.Api().default_entity` — 'thakk100'/'thdhyan' are not valid entities.
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
class K1VelocityPPOTeacherRunnerCfg(K1VelocityPPORunnerCfg):
    """PPO runner for the PRIVILEGED teacher (height-scan + noise-free obs).

    Trains on the "teacher" obs group (235-dim: 48 proprio + 187 height scan).
    The checkpoint this produces is consumed by the distillation runner as the
    frozen teacher. Everything else (rewards, terrain, PPO hyperparams) is
    identical to the blind-policy run for comparability.
    """

    experiment_name = "k1_velocity_teacher"
    obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
