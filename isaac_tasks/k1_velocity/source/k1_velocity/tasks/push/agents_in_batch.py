"""Agents-in-batch VecEnv wrapper (M2, PLAN_MAPPO_TEAM_SOCCER.md): R robots per env -> R*N rows.

Wraps an ``RslRlVecEnvWrapper`` around an env with per-robot obs groups ``{group}_k``,
per-robot action terms (robot-major in the action manager) and reward terms whose
per-robot copies end in ``_k``. rsl_rl then sees one ordinary single-agent VecEnv of
R*N rows and trains ONE shared actor (parameter sharing):

* obs   : ``{group}: cat([obs[f"{group}_k"] for k], 0)`` -> (R*N, D), robot-major rows
* action: (R*N, A) -> (N, R*A) robot-major, the action manager's term order
* reward: row (k, n) = team terms + robot k's ``_k`` terms (other robots' terms removed)
* dones / time_outs: broadcast to every robot of the env (the env resets as a whole)

The critic sees the same per-robot group as the actor (IPPO with shared weights), so a
single-robot push checkpoint warm-starts actor and critic unchanged.
"""
from __future__ import annotations

import re

import torch
from tensordict import TensorDict


class RewardSplit:
    """Per-robot rewards: total minus every OTHER robot's ``_k``-suffixed reward terms."""

    def __init__(self, base_env, num_agents: int):
        rm = base_env.reward_manager
        self._rm = rm
        self._dt = base_env.step_dt
        own = [[i for i, n in enumerate(rm.active_terms) if re.fullmatch(rf".*_{k}", n)]
               for k in range(num_agents)]
        self._other_cols = [
            torch.tensor([i for j in range(num_agents) if j != k for i in own[j]],
                         dtype=torch.long, device=base_env.device)
            for k in range(num_agents)
        ]

    def __call__(self, rew: torch.Tensor) -> list[torch.Tensor]:
        step_rew = self._rm._step_reward          # (N, T), already divided by dt
        return [rew - step_rew[:, cols].sum(dim=1) * self._dt if cols.numel() else rew
                for cols in self._other_cols]


class AgentsInBatchWrapper:
    def __init__(self, env, num_agents: int, group: str = "teacher"):
        self.env = env
        self.R = num_agents
        self.group = group
        self.N = env.num_envs
        self.num_envs = self.N * self.R
        self.num_actions = env.num_actions // self.R
        self.device = env.device
        self.cfg = env.cfg
        self.max_episode_length = env.max_episode_length

        self._split = RewardSplit(env.unwrapped, self.R)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf.repeat(self.R)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        # rsl_rl's init_at_random_ep_len writes R*N values; robots of one env share one clock
        self.env.episode_length_buf = value[: self.N]

    def _stack(self, obs: TensorDict) -> TensorDict:
        x = torch.cat([obs[f"{self.group}_{k}"] for k in range(self.R)], dim=0)
        return TensorDict({self.group: x}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._stack(self.env.get_observations())

    def step(self, actions: torch.Tensor):
        a = actions.view(self.R, self.N, self.num_actions).permute(1, 0, 2).reshape(self.N, -1)
        obs, rew, dones, extras = self.env.step(a)
        rews = self._split(rew)
        if "time_outs" in extras:
            extras["time_outs"] = extras["time_outs"].repeat(self.R)
        return self._stack(obs), torch.cat(rews), dones.repeat(self.R), extras

    def close(self):
        return self.env.close()
