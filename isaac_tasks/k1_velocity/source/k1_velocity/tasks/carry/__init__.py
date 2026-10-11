"""K1 box-carry (P7): walk to a ground goal holding a box welded between both forearms.

Box width 0.25-0.40 m is randomised per env at USD time; two FixedJoints weld it to
the forearm stub tips (carry_mdp). Mass is resampled per reset, U(1, hi) with hi
2 -> 8 kg following the goal-radius curriculum. The actor is decoupled:
one MLP drives the 12 leg joints, another the 8 arm joints (decoupled_policy).
Goal: up to 1 -> 10 m away (curriculum) in any direction; success = trunk within
0.25 m for 1 s, shown as a ghost of the carried box plus a 0.25 m disc, red, green
while the robot is inside (carry_command).
"""
import gymnasium as gym

gym.register(
    id="Isaac-Carry-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.carry.carry_env_cfg:K1CarryEnvCfg",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.carry.agents.rsl_rl_ppo_cfg:K1CarryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Carry-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "k1_velocity.tasks.carry.carry_env_cfg:K1CarryEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "k1_velocity.tasks.carry.agents.rsl_rl_ppo_cfg:K1CarryPPORunnerCfg",
    },
)
