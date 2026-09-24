"""K1 P1 BASIC stand/balance environments (PLAN §4 P1)."""
import gymnasium as gym

# P1 teacher: privileged run (obs group "teacher": 42 proprio + height scan + foot contact/slip)
gym.register(
    id="Isaac-Basic-Teacher-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.basic.basic_env_cfg:K1BasicTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.basic.agents.rsl_rl_ppo_cfg:K1BasicPPOTeacherRunnerCfg",
    },
)

# P1 student: distillation run (blind 42-dim x K10 history <- frozen teacher)
gym.register(
    id="Isaac-Basic-Student-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.basic.basic_env_student:K1BasicDistillEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.basic.agents.rsl_rl_distill_cfg:K1BasicDistillRunnerCfg",
    },
)

# P1f teacher: P1 + Isaac Lab built-in force/torque shoves; teacher obs knows the shove
gym.register(
    id="Isaac-Basic-Teacher-K1-F-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.basic.basic_env_force:K1BasicTeacherForceEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.basic.agents.rsl_rl_ppo_cfg:K1BasicPPOTeacherForceRunnerCfg",
    },
)

# P1f student: distillation from the shove-aware teacher (blind 420-dim policy obs)
gym.register(
    id="Isaac-Basic-Student-K1-F-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.basic.basic_env_force:K1BasicDistillForceEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.basic.agents.rsl_rl_distill_cfg:K1BasicDistillForceRunnerCfg",
    },
)
