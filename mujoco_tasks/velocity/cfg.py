"""RL configuration for Booster K1 velocity task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def booster_k1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "k1_velocity"
  return cfg
