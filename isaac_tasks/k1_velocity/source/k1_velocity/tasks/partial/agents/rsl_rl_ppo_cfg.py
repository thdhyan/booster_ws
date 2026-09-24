"""RSL-RL PPO runner config for the K1 partial-control (legs+head) velocity task.

Inherits the blind-policy PPO hyperparameters from the plain velocity task so
the two runs stay comparable; only the experiment/run naming differs.
"""
from isaaclab.utils.configclass import configclass

from k1_velocity.tasks.velocity.agents.rsl_rl_ppo_cfg import K1VelocityPPORunnerCfg


@configclass
class K1PartialPPORunnerCfg(K1VelocityPPORunnerCfg):
    """PPO training config for the hierarchical partial-control base policy."""

    max_iterations = 3000
    experiment_name = "k1_partialctrl_base"
    run_name = "k1_partialctrl_base"
    # Blind proprioceptive policy: single "policy" obs group feeds both actor and
    # critic (inherited: actor/critic = ["policy"]).
