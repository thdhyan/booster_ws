"""PPO runner for the Phase-1 squat base teacher (experiment: k1_squat_teacher)."""

import os

import torch
from isaaclab.utils.configclass import configclass
from torch.distributions import Normal

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg
from rsl_rl.modules.distribution import GaussianDistribution

from .rsl_rl_ppo_cfg import K1VelocityPPOTeacherRunnerCfg


class CappedGaussianDistribution(GaussianDistribution):
    """Gaussian head with std clamped to [``std_min``, ``std_max``]. Own class because the zz-bw SIF's
    rsl_rl has no ``std_range`` kwarg (dl's has); clamping after update() works on both."""

    def __init__(self, output_dim: int, init_std: float = 1.0, std_type: str = "scalar",
                 std_min: float = 0.0, std_max: float = 1.0):
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        self.std_min = std_min
        self.std_max = std_max

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(mlp_output)
        d = self._distribution
        self._distribution = Normal(d.mean, d.stddev.clamp(min=self.std_min, max=self.std_max))


@configclass
class CappedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    class_name: str = f"{__name__}:CappedGaussianDistribution"
    std_min: float = 0.0
    std_max: float = 1.0


@configclass
class K1SquatPPOTeacherRunnerCfg(K1VelocityPPOTeacherRunnerCfg):
    """Same recipe as the validated zz-bw teacher, renamed so logs cannot mix.

    Teacher obs group (noise-free proprio + height scan), 512-256-128 nets and
    all PPO hparams are inherited unchanged so the Phase-1 velocity-parity
    gate compares like with like. The obs dim grows 237 -> 238 by itself: the
    H* column rides into the ``velocity_commands`` term of both obs groups.
    Log dir becomes ``logs/rsl_rl/k1_squat_teacher/{time-stamp}_{run_name}``
    and the wandb run is named accordingly.
    """

    experiment_name = "k1_squat_teacher"
    run_name = "k1_squat_teacher"


@configclass
class K1SquatResidualPPORunnerCfg(K1SquatPPOTeacherRunnerCfg):
    """Squat residual over the frozen velocity student (squat_residual_env_cfg).

    Smaller initial std: the residual is added to a working gait, so std 1.0 would
    start by kicking the legs 0.5 rad at random.
    """

    max_iterations = 6000
    experiment_name = "k1_squat_residual"
    run_name = "k1_squat_residual"

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3, std_type="log"),
    )

    # actions are clamped to +-1 in the residual action, so std above ~1 is free
    # noise: 0.01 grew std 0.3 -> 5.2 in the 2026-10-11 run (skills use 0.001 too)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
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
class K1LowerBodyPPORunnerCfg(K1SquatPPOTeacherRunnerCfg):
    """Stage-0 lower-body base (lower_body_env_cfg). Long run on zz-bw; same PPO recipe."""

    max_iterations = 20000
    save_interval = 250
    experiment_name = "k1_lower_body"
    run_name = "k1_lower_body"



_PRESET = int(os.environ.get("K1_TEACHER_PRESET", "0"))  # lower_body_env_cfg.TEACHER_PRESETS


@configclass
class K1LowerBodyV4PPORunnerCfg(K1LowerBodyPPORunnerCfg):
    """Teacher v4 PBT member (PLAN_HRL_CONTROLLER.md 6a). Member = env var K1_TEACHER_PRESET (0-3);
    seed 42 + N as in TEACHER_PRESETS; logs go to k1_lower_body_v4/<time>_pbt_p<N>. Fresh start only."""

    max_iterations = 10000
    experiment_name = "k1_lower_body_v4"
    run_name = f"pbt_p{_PRESET}"
    seed = 42 + _PRESET
    # v3 diverged (reward -2e11): action_jerk_l2 is a 3rd difference of the RAW actions, which
    # grew unbounded; the joint target is clipped at +-1 rad anyway, so +-2 changes no motion.
    clip_actions = 2.0


@configclass
class K1LowerBodyWalkPPORunnerCfg(K1LowerBodyV4PPORunnerCfg):
    """GEAR-style direct RL walk policy (lower_body_gear_env_cfg): asymmetric actor-critic.

    Actor = deployable ``policy`` group (proprio, upper joints, (vx, vy, wz, H*, f), clock;
    10-step history); critic = privileged ``teacher`` group. No teacher/student distillation.
    """

    experiment_name = "k1_lower_body_walk"
    run_name = "walk"
    seed = 42
    obs_groups = {"actor": ["policy"], "critic": ["teacher"]}
    # noisy 10-step history mixes joint angles and velocities: normalize both nets' inputs.
    # 2026-10-16 walk/stand: with clip_actions 2.0 noise past the clip is free and entropy 0.01
    # grew std 1 -> 103 (walk) / 34 (stand). Start at 0.5 rad, cap at 1.0.
    # 2026-10-16 8k walk with entropy 0: std collapsed 0.5 -> 0.02 and reward slid 212 -> -3 over 38k
    # its, so keep a small bonus plus a 0.15 floor to hold exploration.
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=CappedGaussianDistributionCfg(init_std=0.5, std_type="log", std_min=0.15, std_max=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
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
class K1LowerBodyStandPPORunnerCfg(K1LowerBodyWalkPPORunnerCfg):
    """GEAR-style direct RL stand policy (zero velocity, H* tracking, feet planted)."""

    max_iterations = 5000
    experiment_name = "k1_lower_body_stand"
    run_name = "stand"
