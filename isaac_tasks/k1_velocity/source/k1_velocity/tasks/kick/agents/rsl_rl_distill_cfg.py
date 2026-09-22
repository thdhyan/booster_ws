"""RSL-RL distillation runner config for the K1 kick student.

Teacher-student scheme:
  TEACHER (frozen): PPO policy trained on privileged "teacher" obs
    (blind proprio + ball pose/vel = 48 + 3 + 4 + 3 = 58-dim).
  STUDENT: MLP distilled to proprioceptive "policy" obs with a history stack
    (48 x 10 = 480-dim, term-major oldest->newest).

The student checkpoint is the deployment artifact: it runs on the robot with
only joint encoders + IMU, matching k1_locomotion's node-side history buffer.
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class K1KickDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner for the K1 kick student."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "k1_kick_student"
    logger = "wandb"
    wandb_project = "booster_k1_soccer_hrl"
    wandb_entity = "thakk100-dhyan-home"
    # student <- blind proprioceptive group (history-stacked by the env cfg),
    # teacher <- privileged group (ball state + noise-free proprioception)
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}

    student = RslRlMLPModelCfg(
        # input: 48-dim obs x 10-step history = 480
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    teacher = RslRlMLPModelCfg(
        # input: 48 proprio + 3 ball_pos + 3 ball_vel = 54 (note: no ball quat in base teacher)
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
