"""Tests for the gait-structure rewards and the gait gate.

Two failure modes are covered, and both were real:

* **Wiring** (static, AST): the reward terms must exist, be penalties with
  negative weights, and the tracking reward must actually be sharpened. A reward
  term that is defined but never referenced is inert, which is the same silent
  failure as a ``CurrTerm`` that loads zero terms.

* **Gate logic** (numeric, numpy): ``gait_gate.py`` must FAIL a high-frequency
  shuffle and PASS a human-like gait. A gate that only ever fails is useless, and
  a gate that only ever passes is worse than none -- the displacement gate proved
  that, by passing an 8 Hz chatter.

The reward functions themselves need torch and Isaac Lab, so their numerical
behaviour is checked on the GPU box; what is verified here is that they are
correctly declared and that the gate around them discriminates.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
VEL = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity"
CFG = VEL / "velocity_env_cfg.py"
GAIT = VEL / "gait_rewards.py"
GATE = _ROOT / "isaac_tasks/k1_velocity/scripts/gait_gate.py"


# --- helpers -----------------------------------------------------------------
def _rewards_cfg(path: pathlib.Path) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg":
            return node
    raise AssertionError("no RewardsCfg found")


def _terms(cfg: ast.ClassDef) -> dict:
    """Map term name -> RewTerm call node inside RewardsCfg."""
    out = {}
    for node in ast.walk(cfg):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Call):
                out[tgt.id] = node.value
    return out


def _kw(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _const(node) -> float | None:
    """Literal float, handling the unary minus that ``-1.0`` parses into.

    ``-1.0`` is a UnaryOp(USub, Constant(1.0)) in the AST, not a negative
    Constant, so a naive literal read returns None for every penalty weight.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


# --- the new terms exist and are penalties -----------------------------------
NEW_TERMS = ["gait_cadence", "feet_clearance", "feet_alternation", "stride_length",
             "action_jerk_l2", "phase_swing"]


@pytest.mark.parametrize("term", NEW_TERMS)
def test_gait_term_is_declared_in_rewards_cfg(term: str):
    terms = _terms(_rewards_cfg(CFG))
    assert term in terms, f"{term} is not wired into RewardsCfg; a defined-but-unused reward is inert"


@pytest.mark.parametrize("term", NEW_TERMS)
def test_gait_term_has_a_negative_weight(term: str):
    """All six are penalties. A positive weight would reward chattering."""
    terms = _terms(_rewards_cfg(CFG))
    w = _const(_kw(terms[term], "weight"))
    assert w is not None, f"{term} has no literal weight"
    assert w < 0, f"{term} weight is {w}; penalties must be negative"


def test_tracking_reward_was_sharpened():
    """std 0.25 -> 0.15.

    At std=0.25 the exponential is nearly flat near the optimum, so a 0.14 m/s
    tracking error cost almost nothing and the policy settled at 69% of commanded
    speed while being paid for it.
    """
    terms = _terms(_rewards_cfg(CFG))
    params = _kw(terms["track_lin_vel_xy_exp"], "params")
    assert params is not None
    std = None
    for k in params.keys:
        if getattr(k, "value", None) == "std":
            std = _const(params.values[params.keys.index(k)])
    assert std is not None, "track_lin_vel_xy_exp has no std"
    assert std == pytest.approx(0.15), (
        f"tracking std is {std}; it must be 0.15 or the reward is too flat to "
        "pressure the policy to track accurately"
    )


def test_action_rate_was_strengthened():
    """-0.5 could not see 8 Hz chatter; action_rate_l2 only sees 1st differences."""
    terms = _terms(_rewards_cfg(CFG))
    w = _const(_kw(terms["action_rate_l2"], "weight"))
    assert w is not None and w <= -2.0, f"action_rate_l2 weight is {w}, expected <= -2.0"


def test_gait_functions_defined_in_module():
    src = GAIT.read_text()
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in ("gait_cadence_penalty", "feet_clearance", "feet_alternation_penalty",
               "stride_length_penalty", "action_jerk_l2", "phase_synced_swing"):
        assert fn in defined, f"{fn} is not defined in gait_rewards.py"


def test_no_dead_placeholder_code_left_in_jerk_term():
    """A placeholder var that is computed then deleted is a smell, not a feature."""
    src = GAIT.read_text()
    assert "placeholder, replaced below" not in src
    assert "del jerk" not in src


# --- gate logic --------------------------------------------------------------
def _synth(cadence_hz: float, speed: float, amp: float, steps: int = 750, envs: int = 8) -> str:
    """Write a synthetic trace: knee oscillation at cadence_hz/2, given speed."""
    t = np.arange(steps) * 0.02
    A = np.zeros((steps, envs, 12), np.float32)
    V = np.zeros((steps, envs, 3), np.float32)
    for e in range(envs):
        for j in range(12):
            phase = 0.0 if j < 6 else np.pi
            A[:, e, j] = amp * np.sin(2 * np.pi * cadence_hz * t + phase + j * 0.1)
        V[:, e, 0] = speed
    p = pathlib.Path("/tmp/opencode/_gt.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, actions=A, root_lin_vel=V)
    return str(p)


def _run_gate(path: str) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(GATE), path], capture_output=True, text=True)
    return r.returncode, r.stdout


def test_gate_fails_a_high_frequency_shuffle():
    """4 Hz knee oscillation = 8 steps/s. This is the failure we actually hit."""
    path = _synth(cadence_hz=4.0, speed=0.35, amp=0.1)
    rc, out = _run_gate(path)
    assert rc == 1, f"gate passed a shuffle:\n{out}"
    assert "GAIT_GATE=FAIL" in out
    assert "cadence" in out


def test_gate_passes_a_human_like_walk():
    """1 Hz knee oscillation = 2 steps/s, human walking cadence."""
    path = _synth(cadence_hz=1.0, speed=0.5, amp=0.35)
    rc, out = _run_gate(path)
    assert rc == 0, f"gate failed a plausible walk:\n{out}"
    assert "GAIT_GATE=PASS" in out


def test_gate_passes_a_slow_deliberate_walk():
    """0.7 Hz knee = 1.4 steps/s: slow but legitimately human."""
    path = _synth(cadence_hz=0.7, speed=0.45, amp=0.35)
    rc, out = _run_gate(path)
    assert rc == 0, f"gate failed a slow walk:\n{out}"


def test_gate_rejects_a_non_trace_cleanly():
    p = pathlib.Path("/tmp/opencode/_notrace.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, something_else=np.zeros(3))
    rc, out = _run_gate(str(p))
    assert rc == 2, f"expected exit 2 for a bad input, got {rc}:\n{out}"


def test_gate_rejects_a_missing_file():
    rc, out = _run_gate("/tmp/opencode/definitely_absent.npz")
    assert rc == 2


def test_gait_phase_buffer_shape_is_requested_exactly():
    """The jerk term needs (num_envs, num_actions), not a 1-D (num_envs,).

    Allocating 1-D produced "size of tensor a (12) must match the size of tensor
    b (8) at non-singleton dimension 1" and killed the preflight. The state helper
    must therefore honour the requested shape rather than assuming 1-D.
    """
    src = GAIT.read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "get"
    )
    # the signature must take a shape, not a num_envs int
    args = [a.arg for a in fn.args.args]
    assert "shape" in args, f"_GaitPhase.get signature is {args}, expected a 'shape' arg"
    assert "num_envs" not in args, "a num_envs int invites the 1-D bug back"
    # ast.dump renders torch.zeros as Name(torch) + Attribute(zeros), not as a
    # literal string, so look for the attribute node instead.
    calls_zeros = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "zeros"
        for n in ast.walk(fn)
    )
    assert calls_zeros, "_GaitPhase.get must allocate with torch.zeros"
    # the jerk term must request the full action shape
    assert "flat.shape" in src, "action_jerk_l2 must pass flat.shape, not an int"


def test_gait_rewards_use_the_data_accessor_not_the_raw_object():
    """Isaac Lab exposes joint/body state on the articulation's .data.

    env.scene[...].joint_pos and env.scene[...].body_pos do not exist and killed
    the preflight with 'Articulation object has no attribute joint_pos'. Body
    positions are body_pos_w, matching the working terms in this same file.
    """
    src = GAIT.read_text()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        # env.scene[cfg.name].<attr>  ->  attribute access straight off the prim
        v = node.value
        if (
            isinstance(v, ast.Attribute)
            and v.attr in ("joint_pos", "body_pos", "body_pos_w", "joint_vel")
            and isinstance(v.value, ast.Subscript)
        ):
            inner = v.value.value
            looks_like_scene = (
                isinstance(inner, ast.Attribute) and inner.attr == "scene"
            )
            if looks_like_scene and "data" not in ast.unparse(v):
                bad.append(ast.unparse(v))
    assert not bad, f"accessing state directly off the prim instead of .data: {bad}"


def test_foot_terms_receive_both_sensor_and_asset_cfgs():
    """A ContactSensor has no body_pos, so foot terms need the robot as well."""
    cfg_src = CFG.read_text()
    for term in ("feet_clearance", "stride_length", "phase_swing"):
        assert f"{term} = RewTerm(" in cfg_src
    # feet_clearance and stride_length need both; phase_swing the sensor only.
    assert cfg_src.count('"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])') >= 2


# --- cadence estimator: the maths, verified without torch -------------------
def _cadence_estimate(knee: np.ndarray, dt: float = 0.02, alpha: float = 0.15,
                      window_s: float = 2.5) -> float:
    """NumPy mirror of gait_cadence_penalty's estimator.

    Kept in the test rather than only in the module because torch is not
    installed on the laptop, and this is the term that has to actually
    discriminate a walk from the 8 steps/s shuffle.
    """
    window = max(int(window_s / dt), 8)
    h = 0.0
    hist = np.zeros(window)          # index 0 = newest
    vals = []
    for k in knee:
        h = (1.0 - alpha) * h + alpha * k
        hist = np.roll(hist, 1)
        hist[0] = h
        c = hist - hist.mean()
        s = np.sign(c)
        frac = (s[1:] * s[:-1] < 0).mean()
        vals.append(frac * (window - 1) / (window * dt))
    return float(np.mean(vals[-400:]))


@pytest.mark.parametrize("knee_hz,expected_steps", [(1.0, 2.0), (2.0, 4.0), (0.7, 1.4)])
def test_cadence_estimator_recovers_step_rate(knee_hz, expected_steps):
    t = np.arange(3000) * 0.02
    knee = 0.35 * np.sin(2 * np.pi * knee_hz * t)
    assert _cadence_estimate(knee) == pytest.approx(expected_steps, rel=0.15)


def test_cadence_window_long_enough_to_resolve_low_rates():
    """A 1.28 s window only fits ~2.5 crossings, so the count quantises to
    integers and a 2 steps/s gait reads as anything from 1.5 to 2.5. 2.5 s
    resolves it."""
    t = np.arange(3000) * 0.02
    knee = 0.35 * np.sin(2 * np.pi * 1.0 * t)
    short = _cadence_estimate(knee, window_s=1.28)
    long = _cadence_estimate(knee, window_s=2.5)
    assert abs(long - 2.0) < abs(short - 2.0), (
        f"2.5 s window ({long:.2f}) should be closer to 2.0 than 1.28 s ({short:.2f})"
    )


def test_shuffle_gets_a_strong_penalty_against_the_2hz_target():
    """The 7.9-8.9 steps/s shuffle must feel a real gradient to slow down."""
    t = np.arange(3000) * 0.02
    knee = 0.35 * np.sin(2 * np.pi * 4.0 * t)   # 8 steps/s
    est = _cadence_estimate(knee)
    assert est > 6.0, f"estimator read the shuffle as only {est:.2f} steps/s"
    assert abs(est - 2.0) > 3.0, "shuffle must be far from the 2.0 target to earn a penalty"


def test_default_window_matches_the_tested_value():
    """Guards the default from being shortened back to the quantised window."""
    src = GAIT.read_text()
    assert "window_s: float = 2.5" in src, (
        "the cadence window must default to 2.5 s; 1.28 s quantises too coarsely "
        "to resolve a 2 steps/s gait"
    )


def test_no_invented_isaac_apis_in_gait_rewards():
    """Guard against APIs I invented rather than read.

    compute_contact_sensor_data() does not exist anywhere in Isaac Lab; the
    preflight failed with "'ContactSensorData' object has no attribute
    'compute_contact_sensor_data'". net_forces_w is a property on the sensor's
    .data object, so it is read directly with no call.
    """
    src = GAIT.read_text()
    for invented in ("compute_contact_sensor_data", "get_contact_forces",
                     "contact_forces_data", "is_in_contact("):
        assert invented not in src, (
            f"{invented!r} is not an Isaac Lab API; read the property instead "
            "(e.g. sensor.data.net_forces_w)"
        )
    assert "data.net_forces_w" in src, "contact terms must read sensor.data.net_forces_w"


def test_state_reads_go_through_dot_data():
    """Joint/body/contact state all live under a .data accessor."""
    src = GAIT.read_text()
    for needed in ("data.joint_vel", "data.body_pos_w", "data.net_forces_w"):
        assert needed in src, f"expected {needed} in the gait reward terms"


def test_cadence_reads_velocity_not_position():
    """Regression: the cadence estimator must not read joint_pos.

    Measured failure of the joint_pos version: the knee position's slow
    postural drift (0.07-1.0 Hz) dominates its 4-6 Hz stepping oscillation, so
    the estimator read ~1-3 steps/s while gait_gate measured 6.68 -- the one
    term priced against the shuffle paid only -0.394/step and the retrain
    shuffled 8/8 again. Velocity oscillates about zero at the step frequency
    and has no drift to hide it behind.
    """
    src = GAIT.read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "gait_cadence_penalty"
    )
    # Drop the docstring: it names joint_pos when explaining why it is wrong.
    stmts = [
        s for s in fn.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str))
    ]
    body = "\n".join(ast.unparse(s) for s in stmts)
    assert "data.joint_vel" in body, "gait_cadence_penalty must measure joint velocity"
    assert "joint_pos" not in body, (
        "gait_cadence_penalty reads joint_pos, whose postural drift hides the "
        "chatter -- that estimator blindness is what let the shuffle survive a "
        "3000-iteration retrain"
    )


def test_phase_swing_consumes_the_same_clock_as_the_observation():
    """The reward must read the clock through gait_clock, not a private copy.

    Two phase sources would drift: the observation advances on
    episode_length_buf changing, and a reward that advanced on call order would
    be one group-read out of phase with what the policy sees.
    """
    src = GAIT.read_text()
    assert "gait_clock.get_phase(" in src, (
        "phase_synced_swing must read the shared phase from gait_clock"
    )
    clock_src = (CFG.parent / "gait_clock.py").read_text()
    tree = ast.parse(clock_src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "get_phase" in defined, "gait_clock.py must expose get_phase"
    # phase_clock must delegate to it rather than duplicating the advance logic.
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "phase_clock"
    )
    assert "get_phase(" in ast.unparse(fn), (
        "phase_clock must encode get_phase's output, not re-advance its own copy"
    )


def test_probe_treats_new_gait_terms_as_zero_action_inert():
    """The probe holds the action at zero, so these terms are structurally zero.

    feet_clearance needs a swinging foot, action_jerk_l2 needs action history and
    gait_cadence needs zero crossings. All are legitimately 0 for a constant
    zero action, so without listing them the probe reports them as dead terms and
    fails the run.
    """
    probe = _ROOT / "isaac_tasks/k1_velocity/scripts/probe_rewards.py"
    src = probe.read_text()
    tree = ast.parse(src)
    inert = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "ZERO_ACTION_INERT":
                    inert = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    assert inert is not None, "ZERO_ACTION_INERT set not found in probe_rewards.py"
    for term in ("feet_clearance", "action_jerk_l2", "gait_cadence"):
        assert term in inert, (
            f"{term} is structurally zero under a constant zero action and must be "
            "listed in ZERO_ACTION_INERT or the probe fails every run"
        )
    # and the pre-existing entries must not have been dropped
    for term in ("action_rate_l2", "termination_penalty", "undesired_contacts"):
        assert term in inert


# --- launcher hygiene --------------------------------------------------------
def test_no_launcher_deletes_a_cache_it_cannot_delete():
    """`rm -rf $CACHE` silently fails on root-owned files and leaves a stale lock.

    The container runs as --user 0, so its cache files are root-owned and cannot
    be removed from the host. The stale ov/_cache.lock then blocks the next Kit
    boot forever: 0% CPU, ~43 MiB, no output, no exception. Each launcher must
    use a fresh per-run cache directory instead of trying to clean the old one.
    """
    import glob
    bad = []
    for path in sorted(glob.glob(str(_ROOT / "scripts/spark_*_host.sh"))):
        src = pathlib.Path(path).read_text()
        if 'rm -rf "$CACHE"' in src:
            bad.append(pathlib.Path(path).name)
    assert not bad, (
        f"these launchers rm -rf a root-owned cache, which fails silently and "
        f"leaves a stale ov/_cache.lock: {bad}. Use a per-run timestamped dir."
    )


def test_p2_launchers_use_a_unique_cache_dir():
    import glob
    for path in sorted(glob.glob(str(_ROOT / "scripts/spark_p2_*_host.sh"))):
        src = pathlib.Path(path).read_text()
        assert "date +%Y%m%d_%H%M%S" in src, (
            f"{pathlib.Path(path).name} must build a unique cache dir per run"
        )


def test_campaign_uses_one_kit_launch_per_container():
    """AppLauncher deadlocks on a second Kit launch inside one container.

    train.py hung at 0% CPU and 43 MiB with zero output, right after the
    preflight probe had booted Kit successfully in that same container. Every
    import was verified fine, so being the second Kit launch was the only
    difference. The campaign must therefore be split into a preflight container
    and a train container.
    """
    d = _ROOT / "scripts"
    pre = (d / "spark_p2_gaitfix_preflight_container.sh")
    trn = (d / "spark_p2_gaitfix_train_container.sh")
    host = (d / "spark_p2_gaitfix_host.sh")
    for f in (pre, trn, host):
        assert f.is_file(), f"missing {f.name}"
    # Neither stage container may both preflight and train.
    assert "probe_rewards.py" in pre.read_text()
    assert "probe_rewards.py" not in trn.read_text(), (
        "the train container must not run the preflight probe: that would be a "
        "second Kit launch and it deadlocks"
    )
    assert "train.py" in trn.read_text()
    # The host must invoke the container script twice, in two stages.
    h = host.read_text()
    assert h.count("run_stage") >= 3, "host must run both stages via run_stage"
    assert "preflight" in h and "train" in h
    # And the preflight stage must exit before doing any training.
    assert "exit 0" in pre.read_text()
