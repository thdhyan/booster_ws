"""Decoupled whole-body actor for the carry task: one MLP for the legs, one for the arms.

Both heads read the same observation vector and are trained together by one PPO
update, but share no weights, so the lower body (12 leg joints) and the upper
body (8 arm joints) are two separate policies. The action manager concatenates
terms in config order (``joint_pos`` legs, then ``arm_pos``), so the leg head
owns the first ``leg_dim`` outputs.

Only ``self.mlp`` is replaced. rsl_rl's MLPModel forward, the state-independent
Gaussian std and the JIT/ONNX exporters (which deepcopy ``model.mlp``) all work
unchanged on the split module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


class SplitMLP(nn.Module):
    def __init__(self, input_dim: int, leg_dim: int, arm_dim: int, hidden_dims, activation: str):
        super().__init__()
        self.legs = MLP(input_dim, leg_dim, hidden_dims, activation)
        self.arms = MLP(input_dim, arm_dim, hidden_dims, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.legs(x), self.arms(x)], dim=-1)


class DecoupledMLPModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 256, 256),
                 activation="elu", leg_dim: int = 12, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims=hidden_dims,
                         activation=activation, **kwargs)
        if not 0 < leg_dim < output_dim:
            raise ValueError(f"leg_dim={leg_dim} must split output_dim={output_dim}")
        self.mlp = SplitMLP(self._get_latent_dim(), leg_dim, output_dim - leg_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)


@configclass
class DecoupledMLPModelCfg(RslRlMLPModelCfg):
    class_name: str = "k1_velocity.tasks.carry.decoupled_policy:DecoupledMLPModel"
    leg_dim: int = 12
