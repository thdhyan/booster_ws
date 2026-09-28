"""K1 box-push (P6) environments — hierarchical: frozen squat base + wrist IK.

v3 (Phase-0 degenerate-audit fixes A1-A15):

- Action (10): [vx, vy, wz, H*] command slice | left wrist EE position delta
  (3) | right (3). The 4-dim slice drives the FROZEN squat-capable base
  teacher (TorchScript export; its 236-dim obs is assembled inside
  FrozenBaseVelocityAction in the exact squat TeacherCfg order, H* maps to
  the command-relative height terms); legacy partial mode keeps a 3-dim
  slice (68-dim obs, legs+head). Each wrist slice drives a
  DifferentialInverseKinematicsAction term (EE = the hand link).
- Box: 1 m prototype cube, per-env size 0.7-1.5x, mass 3-25 kg, friction
  0.3-1.2 (USD-time DR, fixed per env), semi-transparent green; spawns in an
  annulus r in [0.9, 1.4] m with full yaw.
- Goal: FIXED randomized pose per episode (distance 0.3 -> 1.5 m + yaw
  +-30 -> +-180 deg on phased curricula) - no env-owned integrator; corner
  and corner-centroid tracking against the goal pose are the primary
  rewards; success = mean corner error < 0.08 m held 1 s (sparse +50),
  failures = tip > 45 deg / box OOB / height < H* - 0.05 (-200).
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
