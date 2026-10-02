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
    id="Isaac-Velocity-Rough-K1-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Teacher actor = 237-dim privileged obs, so the play env must promote
        # the teacher group into `policy`. Loading a teacher checkpoint into
        # Isaac-Velocity-Rough-K1-Play-v0 fails on a 512x237 vs 512x50 mismatch.
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_play_cfg:K1VelocityRoughTeacherPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_ppo_cfg:K1VelocityPPOTeacherRunnerCfg",
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

gym.register(
    id="Isaac-Velocity-Distill-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_play_distill:K1VelocityDistillPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_distill_cfg:K1VelocityDistillRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Contact-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_play_contact:K1VelocityContactPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_distill_cfg:K1VelocityDistillRunnerCfg",
    },
)

# P2f teacher: rough velocity + Isaac Lab built-in force/torque shoves; teacher obs knows the shove
gym.register(
    id="Isaac-Velocity-Rough-K1-Teacher-F-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_force_cfg:K1VelocityRoughForceEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_ppo_cfg:K1VelocityPPOTeacherForceRunnerCfg",
    },
)

# P2f student: distillation from the shove-aware teacher (blind 500-dim policy obs)
gym.register(
    id="Isaac-Velocity-Distill-K1-F-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_force_cfg:K1VelocityDistillForceEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_distill_cfg:K1VelocityDistillForceRunnerCfg",
    },
)

# Phase 1 (TRACK B next pipeline): squat-capable base teacher — velocity + commanded
# trunk height H* at VR teleop BASE_LIMITS (squat_env_cfg.py / squat_command.py).
gym.register(
    id="Isaac-Velocity-Squat-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.squat_env_cfg:K1VelocitySquatEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.squat_ppo_cfg:K1SquatPPOTeacherRunnerCfg",
    },
)

# Phase-1 debug video: flat, stationary squat->rise demo (trained policy).
gym.register(
    id="Isaac-Velocity-Squat-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.squat_env_cfg:K1VelocitySquatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.squat_ppo_cfg:K1SquatPPOTeacherRunnerCfg",
    },
)
