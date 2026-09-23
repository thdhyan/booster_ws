"""RSL-RL distillation runner config for the K1 velocity student.

Teacher-student scheme:
  TEACHER (frozen): PPO policy trained on privileged "teacher" obs
    (48 noise-free proprio + 187 height scan = 235-dim) — see
    K1VelocityPPOTeacherRunnerCfg.
  STUDENT: MLP distilled to proprioceptive "policy" obs with a 10-step
    history stack (48 x 10 = 480-dim, term-major oldest->newest).

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
        # input: 48-dim obs x 10-step history = 480
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    teacher = RslRlMLPModelCfg(
        # input: 48 proprio + 187 height scan = 235 (must equal teacher env)
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=5.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
    )
