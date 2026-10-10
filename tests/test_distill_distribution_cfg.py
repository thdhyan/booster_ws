"""The distillation student AND teacher must both be stochastic.

The failure this pins
---------------------
Distillation looked impossible in this ``rsl_rl`` build. It produced two errors that
read like a version incompatibility:

    RuntimeError: Unexpected key(s) in state_dict: "distribution.log_std_param"
    AttributeError: 'MLPModel' object has no attribute 'output_std'

They are one missing cfg field. ``MLPModel.__init__`` does::

    if distribution_cfg is not None:
        self.distribution = dist_class(output_dim, **distribution_cfg)
        mlp_output_dim = self.distribution.input_dim
    else:
        self.distribution = None

so with no ``distribution_cfg`` the model is deterministic, and then:

* a PPO teacher checkpoint carries ``distribution.log_std_param`` (the PPO actor sets
  ``GaussianDistributionCfg(init_std=1.0)``) and the deterministic model has nowhere to
  put it -> "Unexpected key(s)";
* the distillation loss reads the teacher's action std, and ``output_std`` is literally
  ``self.distribution.std`` -> AttributeError.

I spent several attempts concluding these were mutually exclusive. They were not: the
PPO runner's own actor already sets the field, and the distill cfg simply omitted it.

The teacher MUST match the PPO actor's distribution, ``init_std`` included, because the
checkpoint's learned ``log_std`` is loaded into whatever head the config builds.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VEL = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/agents"
DISTILL = VEL / "rsl_rl_distill_cfg.py"
PPO = VEL / "rsl_rl_ppo_cfg.py"

MODELS = ("student", "teacher")


def _model_cfgs(path: pathlib.Path, cls: str | None = None) -> dict[str, ast.Call]:
    """{'actor': <Call>, 'teacher': <Call>} for RslRlMLPModelCfg assignments.

    `cls` scopes the search to one class. Without it a file with several runner cfgs
    collapses to whichever assignment happens to come last, which is how the wrong
    actor's distribution_cfg gets pinned as the reference.
    """
    out: dict[str, ast.Call] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ClassDef):
            continue
        if cls is not None and node.name != cls:
            continue
        for sub in node.body:
            if not isinstance(sub, ast.Assign) or not isinstance(sub.value, ast.Call):
                continue
            fn = getattr(sub.value.func, "attr", getattr(sub.value.func, "id", ""))
            if "RslRlMLPModelCfg" not in str(fn):
                continue
            for t in sub.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = sub.value
    return out


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _dist_cfg_source() -> str:
    return ast.unparse(_kw(_model_cfgs(DISTILL)["teacher"], "distribution_cfg"))


#: The squat teacher (Isaac-Velocity-Squat-K1-v0) uses K1SquatPPOTeacherRunnerCfg, which
#: subclasses K1VelocityPPOTeacherRunnerCfg. THAT class is the reference for the
#: distill teacher -- not the blind K1VelocityPPORunnerCfg actor, which uses a different
#: distribution form (no std_type) and would silently change what the loaded log_std means.
SQUAT_TEACHER_CLS = "K1VelocityPPOTeacherRunnerCfg"


@pytest.mark.parametrize("model", MODELS)
def test_distill_models_are_stochastic(model):
    cfg = _model_cfgs(DISTILL)
    assert model in cfg, f"{model} model cfg not found in rsl_rl_distill_cfg.py"
    assert _kw(cfg[model], "distribution_cfg") is not None, (
        f"{model} has no distribution_cfg, so MLPModel sets self.distribution = None. "
        "That makes the teacher's checkpoint fail to load ('Unexpected key(s): "
        "distribution.log_std_param') AND makes the distillation loss raise "
        "('output_std'), because output_std is self.distribution.std."
    )


def test_teacher_distribution_matches_the_ppo_actor_it_is_distilled_from():
    """init_std included: the learned log_std is loaded into whatever head is built."""
    ppo_actor = _kw(_model_cfgs(PPO, SQUAT_TEACHER_CLS)["actor"], "distribution_cfg")
    assert ppo_actor is not None, "the PPO actor must stay stochastic; the teacher mirrors it"
    assert ast.unparse(ppo_actor) == _dist_cfg_source(), (
        "the distill teacher's distribution_cfg has drifted from the PPO actor's; the "
        "teacher checkpoint was trained with the PPO actor's head"
    )


def test_critic_is_left_deterministic():
    """The critic has no distribution head -- it regresses a scalar.

    Checked on K1VelocityPPORunnerCfg, where the critic is actually ASSIGNED.
    K1VelocityPPOTeacherRunnerCfg only overrides `actor` and inherits `critic`, so
    scoping the search to the teacher class finds no critic at all -- which is what
    the first version of this test tripped over.
    """
    base = _model_cfgs(PPO, "K1VelocityPPORunnerCfg")
    assert "critic" in base, "expected the base runner to define a critic"
    assert _kw(base["critic"], "distribution_cfg") is None, (
        "the critic should not have gained a distribution head"
    )


def test_the_two_ppo_actors_deliberately_differ():
    """Why scoping matters: the blind and privileged actors use different std forms.

    The teacher uses std_type="log" because the rsl_rl default "scalar"
    parameterisation can cross zero late in long PPO runs (see the runner's comment).
    Pinning the distill teacher to the WRONG actor would load the checkpoint's log_std
    into a head that means something else -- silently, with no shape error.
    """
    blind = _kw(_model_cfgs(PPO, "K1VelocityPPORunnerCfg")["actor"], "distribution_cfg")
    priv = _kw(_model_cfgs(PPO, SQUAT_TEACHER_CLS)["actor"], "distribution_cfg")
    assert ast.unparse(blind) != ast.unparse(priv), (
        "expected the blind and privileged actors to differ; if they have converged, "
        "re-check which one the distill teacher is pinned to"
    )
    # ast.unparse normalises quotes to single, so compare on the parsed value rather
    # than on the source text.
    kw = {k.arg: k.value for k in priv.keywords if k.arg}
    assert ast.literal_eval(kw["std_type"]) == "log", (
        f"the privileged actor's std_type changed to {ast.literal_eval(kw['std_type'])!r}; "
        "the distill teacher is pinned to it and would silently reinterpret log_std"
    )

# ------------------------------------------------- squat teacher needs its own task


def test_squat_distill_task_is_registered_separately():
    """238 vs 237: sharing the velocity distill env fails with a size mismatch.

    The squat command appends H* (commanded trunk height) to the observation, so the
    squat teacher is 238-dim where the velocity teacher is 237. The runner cfg infers
    the input width from the env, so distilling a squat checkpoint through
    Isaac-Velocity-Distill-K1-v0 raises
        size mismatch for mlp.0.weight: checkpoint [512, 238], model [512, 237]
    """
    reg = (_ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/__init__.py").read_text()
    assert 'id="Isaac-Velocity-Squat-K1-Distill-v0"' in reg
    block = reg.split('id="Isaac-Velocity-Squat-K1-Distill-v0"', 1)[1].split("gym.register", 1)[0]
    assert "K1VelocitySquatDistillEnvCfg" in block, (
        "the squat distill task must use the SQUAT env cfg, not the velocity one"
    )
    # and it must NOT point at the plain velocity distill cfg
    assert "K1VelocityDistillEnvCfg" not in block


def test_squat_distill_env_extends_the_squat_env_not_the_velocity_one():
    src = (_ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/"
           "velocity_env_distill.py").read_text()
    assert "class K1VelocitySquatDistillEnvCfg(K1VelocitySquatEnvCfg)" in src, (
        "the squat distill env must inherit the squat cfg so the student is distilled "
        "against the same terrain/commands/rewards its teacher trained on"
    )
    body = src.split("class K1VelocitySquatDistillEnvCfg", 1)[1]
    assert "history_length" in body, "the student still needs its 10-step history stack"
