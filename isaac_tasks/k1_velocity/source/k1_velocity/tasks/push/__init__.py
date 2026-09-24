"""K1 box-push (P6) environments — hierarchical: frozen locomotion base + wrist IK.

- Action (9): velocity override (3) | left wrist EE position delta (3) | right (3).
  The velocity slice drives the FROZEN Run-11 partial-control TorchScript policy
  (its 68-dim obs is assembled inside FrozenBaseVelocityAction); each wrist
  slice drives a DifferentialInverseKinematicsAction term (EE = the hand link).
- Box: 1 m prototype cube, per-env size 0.7-1.5x, mass 3-25 kg, friction
  0.3-1.2 (USD-time DR, fixed per env), semi-transparent green.
- Goals: rigid pose walking away from the box (cumulative integrator); corner
  and corner-centroid tracking are the primary rewards.
- Obs group is named TeacherCfg (fully observable by design: mass, size,
  corners, goal pose/corners) - matches the deployability test's privileged
  convention.
"""
import gymnasium as gym

gym.register(
    id="Isaac-Push-Reach-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.push.push_env_cfg:K1PushReachEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.push.agents.rsl_rl_ppo_cfg:K1PushReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Push-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.push.push_env_cfg:K1PushEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.push.agents.rsl_rl_ppo_cfg:K1PushPPORunnerCfg",
    },
)
