"""RSL-RL distillation runner config for the K1 velocity student.

Teacher-student scheme:
  TEACHER (frozen): PPO policy trained on privileged "teacher" obs
    (48 noise-free proprio + 187 height scan + 2 phase clock = 237-dim) — see
    K1VelocityPPOTeacherRunnerCfg.
  STUDENT: MLP distilled to proprioceptive "policy" obs with a 10-step
    history stack (50 x 10 = 500-dim, term-major oldest->newest).

The student checkpoint is the deployment artifact: it runs on the robot with
only joint encoders + IMU, matching k1_locomotion's node-side history buffer
(input_mode=stacked).
"""
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class K1VelocityDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner for the K1 velocity student."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "p2_move_student"
    logger = "wandb"
    wandb_project = "booster_k1_soccer_hrl"
    wandb_entity = "thakk100-dhyan-home"
    # student <- blind proprioceptive group (history-stacked by the env cfg),
    # teacher <- privileged group (height scan + noise-free proprioception)
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}

    student = RslRlMLPModelCfg(
        # input: 50-dim obs x 10-step history = 500
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        # BOTH models need a distribution, and omitting it is what made distillation
        # look impossible in this rsl_rl build. MLPModel only creates `self.distribution`
        # when distribution_cfg is given (mlp_model.py: "if distribution_cfg is not
        # None: ... else: self.distribution = None"). With it absent the models are
        # deterministic, and BOTH symptoms follow from that single fact:
        #   * loading a PPO teacher checkpoint fails with
        #       Unexpected key(s) in state_dict: "distribution.log_std_param"
        #     because the deterministic model has nowhere to put the head;
        #   * the distillation loss reads the teacher's action std and raises
        #       AttributeError: 'MLPModel' object has no attribute 'output_std'
        #     because output_std is `self.distribution.std`.
        # So the two errors are NOT a version incompatibility -- they are one missing
        # cfg field, and the PPO runner's own actor already sets it (init_std=1.0).
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    teacher = RslRlMLPModelCfg(
        # input: 48 proprio + 187 height scan + 2 phase clock = 237 (must equal teacher env)
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        # MUST mirror the PPO actor exactly, including init_std: the checkpoint being
        # distilled was trained with this head, and a differing init_std changes the
        # initial log_std the weights are loaded into.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=5.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
    )


@configclass
class K1VelocityDistillForceRunnerCfg(K1VelocityDistillRunnerCfg):
    """Distillation runner for the **P2f** student.

    Teacher input = P2f teacher env obs: 50 clean (48 proprio + 2 clock) +
    187 scan + 6 shove wrench = 243 (net input sizes are inferred from the
    env, so the MLP fields stay inherited). Student stays blind 50x10 = 500 —
    no shove knowledge.
    Launch: ``--task Isaac-Velocity-Distill-K1-F-v0 --checkpoint <p2f teacher model.pt>``.
    """

    experiment_name = "p2f_move_student"
    run_name = "p2f_move_student"
