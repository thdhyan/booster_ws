"""Carry actor: two independent MLPs (12 legs, 8 arms) behind rsl_rl's MLPModel API.

Checks the split is real (no shared weights, each head only moves its own action
slice), that the PPO-facing surface (stochastic forward, std, log-prob) and the
TorchScript export still work, and that the cfg's class_name resolves.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

tensordict = pytest.importorskip("tensordict")
pytest.importorskip("rsl_rl")
pytest.importorskip("isaaclab_rl")

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/carry/decoupled_policy.py"


def _load():
    spec = importlib.util.spec_from_file_location("decoupled_policy", MOD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # configclass/dataclass looks the module up
    spec.loader.exec_module(mod)
    return mod


dp = _load()
OBS_DIM, N = 70, 16


def _model():
    obs = tensordict.TensorDict({"teacher": torch.randn(N, OBS_DIM)}, batch_size=[N])
    m = dp.DecoupledMLPModel(
        obs, {"actor": ["teacher"]}, "actor", 20, hidden_dims=[64, 32], activation="elu",
        obs_normalization=True, leg_dim=12,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
    )
    return m, obs


def test_output_shape_and_heads():
    m, obs = _model()
    assert isinstance(m.mlp, dp.SplitMLP)
    assert m.mlp.legs[-1].out_features == 12 and m.mlp.arms[-1].out_features == 8
    assert m(obs).shape == (N, 20)
    assert m(obs, stochastic_output=True).shape == (N, 20)
    assert m.output_std.shape == (N, 20)


def test_heads_share_no_parameters_and_own_their_slice():
    m, obs = _model()
    leg_ids = {id(p) for p in m.mlp.legs.parameters()}
    assert leg_ids.isdisjoint({id(p) for p in m.mlp.arms.parameters()})
    m(obs)[:, :12].sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.mlp.legs.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.mlp.arms.parameters())


def test_jit_export_matches_eager():
    m, obs = _model()
    m.eval()
    jit = torch.jit.script(m.as_jit())
    x = obs["teacher"]
    assert torch.allclose(jit(x), m(obs), atol=1e-6)


def test_bad_split_rejected():
    obs = tensordict.TensorDict({"teacher": torch.randn(N, OBS_DIM)}, batch_size=[N])
    with pytest.raises(ValueError):
        dp.DecoupledMLPModel(obs, {"actor": ["teacher"]}, "actor", 20, hidden_dims=[8], leg_dim=20)


def test_cfg_class_name_points_at_this_module():
    cfg = dp.DecoupledMLPModelCfg(hidden_dims=[8], activation="elu", stochastic=True)
    mod, cls = cfg.class_name.split(":")
    assert mod == "k1_velocity.tasks.carry.decoupled_policy" and cls == "DecoupledMLPModel"
    assert cfg.leg_dim == 12
