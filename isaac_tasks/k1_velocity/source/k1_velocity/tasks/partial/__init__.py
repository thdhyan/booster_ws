"""K1 hierarchical partial-control velocity environments (14-dim action: 12 leg + 2 head).

The 8 arm joints are outside the action space: randomized per episode (with a
small-perturbation → full-random curriculum) and held by their PD controllers.
"""
import gymnasium as gym

# Register envs
gym.register(
    id="Isaac-Velocity-PartialCtrl-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.partial.partial_env_cfg:K1PartialCtrlEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.partial.agents.rsl_rl_ppo_cfg:K1PartialPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-PartialCtrl-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.partial.partial_play_cfg:K1PartialCtrlPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.partial.agents.rsl_rl_ppo_cfg:K1PartialPPORunnerCfg",
    },
)
