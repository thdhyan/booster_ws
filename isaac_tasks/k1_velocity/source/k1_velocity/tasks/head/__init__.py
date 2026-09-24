"""K1 head-tracking (P3) environments.

Single-stage, deployable-by-design: the policy only sees the *detection vector*
(visible flag, du, dv) produced by the camera+YOLO pipeline (or its geometric
fallback), never true ball state.  Head motors only; legs are held at the
standing default.  Ball speed is a curriculum (0 -> v_max): first centre a
static ball, then track a slowly rolling one.
"""
import gymnasium as gym

gym.register(
    id="Isaac-HeadTrack-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.head.head_env_cfg:K1HeadTrackEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.head.agents.rsl_rl_ppo_cfg:K1HeadTrackPPORunnerCfg",
    },
)
