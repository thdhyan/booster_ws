"""K1 kick task environments."""
import gymnasium as gym

# Register environments

# Blind policy trained on proprioceptive observations
gym.register(
    id="Isaac-Kick-Ball-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.kick.kick_env_cfg:K1KickEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.kick.agents.rsl_rl_ppo_cfg:K1KickPPORunnerCfg",
    },
)

# Privileged teacher (ball state + noise-free observations)
gym.register(
    id="Isaac-Kick-Ball-K1-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.kick.kick_env_cfg:K1KickEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.kick.agents.rsl_rl_ppo_cfg:K1KickPPOTeacherRunnerCfg",
    },
)

# Distillation student (blind policy trained via teacher)
gym.register(
    id="Isaac-Kick-Ball-K1-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.kick.kick_env_cfg:K1KickEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.kick.agents.rsl_rl_distill_cfg:K1KickDistillRunnerCfg",
    },
)
