"""PPO runner for the Phase-1 squat base teacher (experiment: k1_squat_teacher)."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import K1VelocityPPOTeacherRunnerCfg


@configclass
class K1SquatPPOTeacherRunnerCfg(K1VelocityPPOTeacherRunnerCfg):
    """Same recipe as the validated zz-bw teacher, renamed so logs cannot mix.

    Teacher obs group (noise-free proprio + height scan), 512-256-128 nets and
    all PPO hparams are inherited unchanged so the Phase-1 velocity-parity
    gate compares like with like. The obs dim grows 235 -> 236 by itself: the
    H* column rides into the ``velocity_commands`` term of both obs groups.
    Log dir becomes ``logs/rsl_rl/k1_squat_teacher/{time-stamp}_{run_name}``
    and the wandb run is named accordingly.
    """

    experiment_name = "k1_squat_teacher"
    run_name = "k1_squat_teacher"
