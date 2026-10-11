"""PPO runner for the Phase-1 squat base teacher (experiment: k1_squat_teacher)."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .rsl_rl_ppo_cfg import K1VelocityPPOTeacherRunnerCfg


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
