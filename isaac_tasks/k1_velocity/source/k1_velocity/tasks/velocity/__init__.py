"""K1 velocity locomotion environments."""
import gymnasium as gym

# Register envs
gym.register(
    id="Isaac-Velocity-Rough-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_env_cfg:K1VelocityRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_ppo_cfg:K1VelocityPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_play_cfg:K1VelocityRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_ppo_cfg:K1VelocityPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-K1-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_env_cfg:K1VelocityRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_ppo_cfg:K1VelocityPPOTeacherRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Distill-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_env_distill:K1VelocityDistillEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_distill_cfg:K1VelocityDistillRunnerCfg",
    },
)
