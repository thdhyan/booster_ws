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


# Stage-0 lower-body base (PLAN_HRL_CONTROLLER.md WP2): (vx, vy, wz, H*) under random
# arm/head poses and stub-tip forces, time-based squat ramp (lower_body_env_cfg.py).
for _id, _cfg in (("Isaac-LowerBody-K1-v0", "K1LowerBodyEnvCfg"),
                  ("Isaac-LowerBody-K1-Play-v0", "K1LowerBodyPlayEnvCfg")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"k1_velocity.tasks.velocity.lower_body_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.squat_ppo_cfg:K1LowerBodyPPORunnerCfg",
        },
    )

# Teacher v4 PBT member (PLAN_HRL_CONTROLLER.md 6a): survival-gated ramps, stairs/slopes,
# member chosen by env var K1_TEACHER_PRESET=0..3.
for _id, _cfg in (("Isaac-LowerBody-V4-K1-v0", "K1LowerBodyV4EnvCfg"),
                  ("Isaac-LowerBody-V4-K1-Play-v0", "K1LowerBodyV4PlayEnvCfg")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"k1_velocity.tasks.velocity.lower_body_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.squat_ppo_cfg:K1LowerBodyV4PPORunnerCfg",
        },
    )

# Stage-0 lower-body STUDENT (distilled from Isaac-LowerBody-K1-v0; 710-dim stacked proprio).
for _id, _cfg in (("Isaac-LowerBody-K1-Distill-v0", "K1LowerBodyDistillEnvCfg"),
                  ("Isaac-LowerBody-K1-Distill-Play-v0", "K1LowerBodyDistillPlayEnvCfg")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"k1_velocity.tasks.velocity.lower_body_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_distill_cfg:K1LowerBodyDistillRunnerCfg",
        },
    )

# Squat teacher as a residual over the frozen velocity student (feet_flat included).
for _id, _cfg in (("Isaac-Velocity-Squat-Residual-K1-v0", "K1VelocitySquatResidualEnvCfg"),
                  ("Isaac-Velocity-Squat-Residual-K1-Play-v0", "K1VelocitySquatResidualPlayEnvCfg")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"k1_velocity.tasks.velocity.squat_residual_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.squat_ppo_cfg:K1SquatResidualPPORunnerCfg",
        },
    )

# Squat teacher -> deployable student. Separate from Isaac-Velocity-Distill-K1-v0 because
# the squat teacher's obs is 238-dim (H* appended to the command) and the velocity one is
# 237; sharing an env cfg fails with a size mismatch at the first policy call.
gym.register(
    id="Isaac-Velocity-Squat-K1-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_env_distill:K1VelocitySquatDistillEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_distill_cfg:K1VelocityDistillRunnerCfg",
    },
)


# Render the squat STUDENT. Without this the only squat distill task is the training
# one, so there is no way to produce a clip of the student at all.
gym.register(
    id="Isaac-Velocity-Squat-K1-Distill-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.velocity.velocity_env_distill:K1VelocitySquatDistillPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.velocity.agents.rsl_rl_distill_cfg:K1VelocityDistillRunnerCfg",
    },
)

# GEAR-style decoupled lower body (PLAN_HRL_CONTROLLER.md 6g): separate walk / stand policies,
# direct RL with an asymmetric actor-critic, gait-frequency command on the walk policy.
for _id, _cfg, _runner in (
    ("Isaac-LowerBody-Walk-K1-v0", "K1LowerBodyWalkEnvCfg", "K1LowerBodyWalkPPORunnerCfg"),
    ("Isaac-LowerBody-Walk-K1-Play-v0", "K1LowerBodyWalkPlayEnvCfg", "K1LowerBodyWalkPPORunnerCfg"),
    ("Isaac-LowerBody-Stand-K1-v0", "K1LowerBodyStandEnvCfg", "K1LowerBodyStandPPORunnerCfg"),
    ("Isaac-LowerBody-Stand-K1-Play-v0", "K1LowerBodyStandPlayEnvCfg", "K1LowerBodyStandPPORunnerCfg"),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"k1_velocity.tasks.velocity.lower_body_gear_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"k1_velocity.tasks.velocity.agents.squat_ppo_cfg:{_runner}",
        },
    )
