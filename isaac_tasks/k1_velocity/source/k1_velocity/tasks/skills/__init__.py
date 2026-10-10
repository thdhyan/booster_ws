"""Residual skills on the frozen velocity student (P8 ball track, P9 kick)."""
import gymnasium as gym

_CFG = "k1_velocity.tasks.skills.skills_env_cfg"
_AGENT = "k1_velocity.tasks.skills.agents.rsl_rl_ppo_cfg"

for _id, _env, _agent in (
    ("Isaac-BallTrack-Residual-K1-v0", "K1BallTrackResidualEnvCfg", "K1BallTrackResidualPPORunnerCfg"),
    ("Isaac-BallTrack-Residual-K1-Play-v0", "K1BallTrackResidualPlayEnvCfg", "K1BallTrackResidualPPORunnerCfg"),
    ("Isaac-Kick-Residual-K1-v0", "K1KickResidualEnvCfg", "K1KickResidualPPORunnerCfg"),
    ("Isaac-Kick-Residual-K1-Play-v0", "K1KickResidualPlayEnvCfg", "K1KickResidualPPORunnerCfg"),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{_CFG}:{_env}", "rsl_rl_cfg_entry_point": f"{_AGENT}:{_agent}"},
    )
