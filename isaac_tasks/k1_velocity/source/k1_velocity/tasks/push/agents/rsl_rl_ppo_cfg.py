"""RSL-RL PPO runners for box pushing (reach, then push)."""
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class K1PushReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 1: walk up, extend wrists, make contact (no push rewards)."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "p6_push_reach"
    run_name = "p6_push_reach"
    logger = "wandb"
    wandb_project = "booster_k1_soccer_hrl"
    wandb_entity = "thakk100-dhyan-home"
    obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
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
class K1PushPPORunnerCfg(K1PushReachPPORunnerCfg):
    """Stage 2: corner-goal pushing (warm-start from the reach ckpt)."""

    max_iterations = 3000
    experiment_name = "p6_push"
    run_name = "p6_push"

    def __post_init__(self):
        # Actions are clamped to +-1 in FrozenBaseVelocityAction, so a wider std costs
        # nothing and the entropy bonus drove it 0.8 -> 3.2 (2026-10-10 run, reward
        # 125 -> 32, goal never reached). No entropy bonus for this stage.
        self.algorithm.entropy_coef = 0.0


@configclass
class K1PushMultiPPORunnerCfg(K1PushPPORunnerCfg):
    """M2 multi-robot push: shared actor over both robots, warm-started from p6_push."""

    max_iterations = 3000
    experiment_name = "p6_push_multi"
    run_name = "p6_push_multi"


@configclass
class K1PushSimplePPORunnerCfg(K1PushPPORunnerCfg):
    """Simplified push rewards (push_simple_env_cfg)."""

    experiment_name = "p6_push_simple"
    run_name = "p6_push_simple"
