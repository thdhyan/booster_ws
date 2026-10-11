"""Manager-based multi-robot push env -> DirectMARLEnv-style interface for skrl MAPPO.

skrl's ``IsaacLabMultiAgentWrapper`` expects Isaac Lab's DirectMARLEnv API: ``possible_agents``,
per-agent ``observation_spaces`` / ``action_spaces``, a global ``state_space`` + ``state()``, and
dict-valued ``step`` / ``reset``. This adapter provides that on top of the manager-based
``Isaac-Push-Multi-K1-*`` env, so every robot gets its OWN policy (no parameter sharing) and
a centralized critic:

* agent ``robot_k`` observes ``teacher_k`` (109 dims, single-robot layout)
* acts with its slice of the action vector (robot-major: base 4 | wrist L 3 | wrist R 3)
* reward = team terms + its own ``_k`` terms (``RewardSplit``)
* ``state()`` = concat of every robot's ``teacher_k`` (the critic's global view)
* terminated / truncated are shared: the env resets as a whole
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from .agents_in_batch import RewardSplit


class MultiRobotMARLAdapter:
    def __init__(self, env, num_agents: int, group: str = "teacher"):
        self.env = env
        base = env.unwrapped
        self.base = base
        self.R = num_agents
        self.group = group
        self.possible_agents = [f"robot_{k}" for k in range(num_agents)]
        self.agents = list(self.possible_agents)
        self.num_envs = base.num_envs
        self.device = base.device
        self.cfg = base.cfg
        obs_dim = base.observation_manager.group_obs_dim[f"{group}_0"][0]
        act_dim = base.action_manager.total_action_dim // num_agents
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), dtype=np.float32)  # noqa: E731
        self.observation_spaces = {a: box(obs_dim) for a in self.possible_agents}
        self.action_spaces = {a: box(act_dim) for a in self.possible_agents}
        self.state_space = box(obs_dim * num_agents)
        self._split = RewardSplit(base, num_agents)
        self._obs = None
        self.log_sink = None          # callable(dict) fed with extras["log"] after resets

    @property
    def unwrapped(self):
        # skrl reads spaces/agents from `.unwrapped`: that must be this adapter
        return self

    def _per_agent(self, obs):
        self._obs = obs
        return {a: obs[f"{self.group}_{k}"] for k, a in enumerate(self.possible_agents)}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._per_agent(obs), info

    def step(self, actions):
        a = torch.cat([actions[ag] for ag in self.possible_agents], dim=1)
        obs, rew, terminated, truncated, extras = self.env.step(a)
        rews = self._split(rew)
        if self.log_sink is not None and extras.get("log"):
            self.log_sink(extras["log"])
        return (self._per_agent(obs),
                {ag: rews[k] for k, ag in enumerate(self.possible_agents)},
                {ag: terminated for ag in self.possible_agents},
                {ag: truncated for ag in self.possible_agents},
                extras)

    def state(self):
        return torch.cat([self._obs[f"{self.group}_{k}"] for k in range(self.R)], dim=1)

    def render(self, *args, **kwargs):
        return None

    def close(self):
        self.env.close()
