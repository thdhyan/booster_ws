"""Train multi-robot push with skrl MAPPO: one policy PER robot + centralized critics.

Each agent ``robot_k`` has its own Gaussian policy (109 -> 512-256-128 -> 10, ELU, the
rsl_rl push actor's shape) and its own value net over the global state (both robots'
teacher obs, 218 dims). ``--checkpoint`` warm-starts every agent from a single-robot
rsl_rl push checkpoint:

* policy: actor MLP weights, log_std = log(rsl std_param), obs normalizer -> skrl
  RunningStandardScaler (running_mean / running_variance / current_count)
* value : rsl critic weights in the agent's own 109-column block of the first layer,
  zeros on the other robot's block; shared-state scaler = per-robot stats concatenated

Usage (container, from the clone root):
    python isaac_tasks/k1_velocity/scripts/train_mappo.py --task Isaac-Push-Multi-K1-v0 \
        --num_envs 2048 --max_iterations 3000 --headless \
        --checkpoint logs/rsl_rl/p6_push/<run>/model_<N>.pt
Checkpoints: logs/skrl/p6_push_mappo/<timestamp>/checkpoints/agent_<step>.pt

skrl: the thdhyan/skrl fork (2.1 API). On zz-bw it is a clone at /tmp/skrl inside the
container ($S/tmp/skrl on the host), put on sys.path below; elsewhere the installed skrl.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="skrl MAPPO, multi-robot push (separate policies)")
parser.add_argument("--task", default="Isaac-Push-Multi-K1-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--rollouts", type=int, default=24)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_interval", type=int, default=100, help="iterations between checkpoints")
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--checkpoint", default=None, help="single-robot rsl_rl push .pt to warm-start every agent")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------- Everything below runs AFTER Isaac Sim is up --------
_SKRL_FORK = os.environ.get("K1_SKRL_PATH", "/tmp/skrl")
if os.path.isdir(os.path.join(_SKRL_FORK, "skrl")):
    sys.path.insert(0, _SKRL_FORK)
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabMultiAgentWrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import k1_velocity.tasks.push  # noqa: F401 — registers Isaac-Push-Multi-K1-*
from k1_velocity.tasks.push.marl_adapter import MultiRobotMARLAdapter


def _mlp(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                         nn.Linear(256, 128), nn.ELU(), nn.Linear(128, n_out))


class Policy(GaussianMixin, Model):
    """Own observations (teacher_k) -> action mean; state-independent log std."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                               min_log_std=-5.0, max_log_std=1.0)
        self.net = _mlp(self.num_observations, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    """Centralized critic: global state (every robot's teacher_k) -> value."""

    def __init__(self, state_space, action_space, device):
        Model.__init__(self, state_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _mlp(self.num_states, 1)

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}


def _load_scaler(scaler, mean, var, count):
    scaler.running_mean.copy_(mean.to(scaler.running_mean))
    scaler.running_variance.copy_(var.to(scaler.running_variance))
    scaler.current_count.fill_(float(count))


def warm_start(agent, path, agents, obs_dim):
    """Single-robot rsl_rl push checkpoint -> every MAPPO agent (see module docstring)."""
    ck = torch.load(path, map_location=agent.device, weights_only=False)
    actor, critic = ck["actor_state_dict"], ck["critic_state_dict"]
    lin = (0, 2, 4, 6)
    for k, uid in enumerate(agents):
        pol, val = agent.policies[uid], agent.values[uid]
        with torch.no_grad():
            for i in lin:
                pol.net[i].weight.copy_(actor[f"mlp.{i}.weight"])
                pol.net[i].bias.copy_(actor[f"mlp.{i}.bias"])
            pol.log_std_parameter.copy_(actor["distribution.std_param"].clamp_min(1e-3).log())
            w0 = torch.zeros_like(val.net[0].weight)
            w0[:, k * obs_dim:(k + 1) * obs_dim] = critic["mlp.0.weight"]
            val.net[0].weight.copy_(w0)
            val.net[0].bias.copy_(critic["mlp.0.bias"])
            for i in lin[1:]:
                val.net[i].weight.copy_(critic[f"mlp.{i}.weight"])
                val.net[i].bias.copy_(critic[f"mlp.{i}.bias"])
            a_n, c_n = "obs_normalizer.", "obs_normalizer."
            _load_scaler(agent._observation_preprocessor[uid], actor[a_n + "_mean"][0],
                         actor[a_n + "_var"][0], actor[a_n + "count"])
            n = len(agents)
            _load_scaler(agent._state_preprocessor[uid],
                         critic[c_n + "_mean"][0].repeat(n), critic[c_n + "_var"][0].repeat(n),
                         critic[c_n + "count"])
    print(f"[INFO] WARM_START {len(agents)} MAPPO agents from {path} "
          f"(std {actor['distribution.std_param'].mean():.2f})")


def main():
    set_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    R = getattr(env_cfg, "agents_in_batch", 0)
    assert R > 1, f"{args_cli.task} is not a multi-robot task (agents_in_batch={R})"
    genv = gym.make(args_cli.task, cfg=env_cfg)
    adapter = MultiRobotMARLAdapter(genv, R, group="teacher")
    env = IsaacLabMultiAgentWrapper(adapter)
    device = env.device

    agents = env.possible_agents
    models, memories = {}, {}
    for uid in agents:
        models[uid] = {
            "policy": Policy(env.observation_space(uid), env.action_space(uid), device),
            "value": Value(env.state_space(uid), env.action_space(uid), device),
        }
        memories[uid] = RandomMemory(memory_size=args_cli.rollouts, num_envs=env.num_envs, device=device)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_p6_push_mappo"
    cfg = {
        "rollouts": args_cli.rollouts,
        "learning_epochs": 5,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 1.0e-4,
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.01},
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "entropy_loss_scale": 0.0,     # actions clamped in the action terms; see K1PushPPORunnerCfg
        "value_loss_scale": 1.0,
        "time_limit_bootstrap": True,
        # rsl_rl's normalizer does not clip; skrl's default (5 sigma) would distort the warm start.
        # play_record.py --mappo assumes the same clip (MAPPO_OBS_CLIP).
        "observation_preprocessor": RunningStandardScaler,
        "observation_preprocessor_kwargs": {"size": env.observation_space(agents[0]), "device": device,
                                            "clip_threshold": 1.0e3},
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.state_space(agents[0]), "device": device,
                                      "clip_threshold": 1.0e3},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
    }
    cfg["experiment"] = {
        "directory": os.path.abspath(os.path.join("logs", "skrl", "p6_push_mappo")),
        "experiment_name": stamp,
        "write_interval": args_cli.rollouts * 10,
        "checkpoint_interval": args_cli.rollouts * args_cli.save_interval,
        "store_separately": False,
        "wandb": not args_cli.no_wandb,
        "wandb_kwargs": {"project": "booster_k1_soccer_hrl", "name": stamp},
    }
    agent = MAPPO(possible_agents=agents, models=models, memories=memories,
                  observation_spaces=env.observation_spaces, state_spaces=env.state_spaces,
                  action_spaces=env.action_spaces, device=device, cfg=MAPPO_CFG(**cfg))

    # Isaac Lab episode stats (Episode_Reward/*, Episode_Termination/*) -> skrl tracking
    def _log(log):
        for key, v in log.items():
            agent.track_data(f"Info / {key}", float(v) if not torch.is_tensor(v) else v.float().mean().item())
    adapter.log_sink = _log

    # environment_info: extras["log"] goes through log_sink above (it also takes floats)
    trainer = SequentialTrainer(cfg={"timesteps": args_cli.max_iterations * args_cli.rollouts,
                                     "headless": True, "environment_info": "none"},
                                env=env, agents=agent)
    if args_cli.checkpoint:
        warm_start(agent, os.path.abspath(args_cli.checkpoint), agents,
                   env.observation_space(agents[0]).shape[0])
    # decorrelate episode clocks (rsl_rl's init_at_random_ep_len)
    base = adapter.base
    base.episode_length_buf = torch.randint_like(base.episode_length_buf, high=int(base.max_episode_length))
    print(f"[INFO] MAPPO: {len(agents)} separate policies, {env.num_envs} envs, "
          f"obs {env.observation_space(agents[0]).shape}, state {env.state_space(agents[0]).shape}, "
          f"log {cfg['experiment']['directory']}/{stamp}")
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
