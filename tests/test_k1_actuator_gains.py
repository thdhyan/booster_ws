"""The K1 actuator gains must be the manufacturer's, not hand-pasted over them.

Found 2026-10-02, and it is the most consequential defect found in this project.

Our fork of ``booster_train`` had dropped ``armature_ratio=(1.4, 0.4)`` from both
K1 ankle joints, silently falling back to the class default ``(2.0, 2.0)``. The
K1 ankle is a 4-bar parallel linkage, so the pitch and roll axes have genuinely
different effective inertia -- that per-axis ratio is the whole point of the
wrapper. On top of the wrong fallback, ``velocity_env_cfg`` was hand-overriding
every leg and ankle gain (hips/knee to 100.0, ankle pitch to 100.0) on the
reasoning that the published 4 Hz natural frequency was "too soft to stand".

Measured consequence, at 8192 envs: **mean episode length 6 steps** of a
1000-step horizon and **96%** of episodes terminating on ``base_orientation``. A
policy that falls in six steps cannot be taught to walk by any reward, which is
why four consecutive runs failed the gait gate 8/8. The gains were 4x and 5x
stiffer than the real robot's.

These tests pin the published values and forbid the overrides coming back.
"""
from __future__ import annotations

import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOSTER = (
    _ROOT
    / "isaac_tasks/booster_train_ref/source/booster_train/booster_train/assets/robots/booster.py"
)
ACTUATOR = BOOSTER.with_name("actuator.py")
_SUBMODULE = "isaac_tasks/booster_train_ref"
CFG = (
    _ROOT
    / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_env_cfg.py"
)


def test_upstream_ankle_armature_ratio_is_present():
    """Both ankle joints must pass armature_ratio=(1.4, 0.4) explicitly.

    Dropping it is silent: the wrapper's class default is (2.0, 2.0), so the
    config loads fine, runs fine, and is simply wrong.
    """
    src = BOOSTER.read_text()
    k1 = src.split("BOOSTER_K1_CFG", 1)[1].split("BOOSTER_T1_CFG", 1)[0]
    assert k1.count("armature_ratio=(1.4, 0.4)") == 2, (
        "both K1 ankle joints must set armature_ratio=(1.4, 0.4); "
        f"found {k1.count('armature_ratio=(1.4, 0.4)')} (expected 2)"
    )


def test_ankle_wrapper_still_wraps_the_right_motor_and_index():
    """Guard the rest of the ankle block, so a fix cannot come at their expense."""
    src = BOOSTER.read_text()
    k1 = src.split("BOOSTER_K1_CFG", 1)[1].split("BOOSTER_T1_CFG", 1)[0]
    assert k1.count("BoosterK1AnkleParaWrapperCfg") == 2
    assert k1.count("BoosterJointE4310()") == 2, "the ankle base motor is the E4310"
    assert 'serial_index=0' in k1 and 'serial_index=1' in k1, (
        "serial_index 0 is pitch and 1 is roll; they are not interchangeable"
    )
    assert '".*_Ankle_Pitch"' in k1 and '".*_Ankle_Roll"' in k1


def test_velocity_cfg_overrides_no_actuator_gains():
    """No hand-tuned gains. The articulation is used exactly as published.

    This is the assertion that would have caught the original defect. The old
    override block set hips/knee to 100.0 and ankle pitch to 100.0 -- 4x and 5x
    the real robot -- on the belief that published gains were "too soft".
    """
    src = CFG.read_text()
    for pattern in (
        r'actuators\["legs"\]\.stiffness',
        r'actuators\["legs"\]\.damping',
        r'actuators\["feet"\]\.stiffness',
        r'actuators\["feet"\]\.damping',
    ):
        assert not re.search(pattern, src), (
            f"{pattern} reintroduces a hand-tuned gain over the manufacturer's "
            "actuator model; use the published values and the Gain Tuner instead"
        )


def test_the_old_override_constants_are_gone():
    """The K1_P2_* dicts must not survive even unused -- they invite re-use."""
    src = CFG.read_text()
    for name in (
        "K1_P2_LEG_STIFFNESS",
        "K1_P2_LEG_DAMPING",
        "K1_P2_FOOT_STIFFNESS",
        "K1_P2_FOOT_DAMPING",
    ):
        assert name not in src, (
            f"{name} still present; a stale copy of the wrong gains is how this "
            "bug gets reintroduced"
        )


def test_articulation_is_a_plain_deepcopy_of_the_published_cfg():
    """Nothing may be mutated on the way in."""
    src = CFG.read_text()
    assert "K1_ARTICULATION_CFG = copy.deepcopy(BOOSTER_K1_CFG)" in src
    # The assignment must be the last thing done to the cfg.
    tail = src.split("K1_ARTICULATION_CFG = copy.deepcopy(BOOSTER_K1_CFG)", 1)[1]
    assert "K1_ARTICULATION_CFG." not in tail.split("\n\n\n", 1)[0], (
        "the articulation cfg is being mutated immediately after the copy"
    )


def test_published_gains_are_the_documented_ones():
    """Pin the arithmetic so a change to the motor models is a deliberate act.

    BoosterDelayedPDActuator derives ``stiffness = armature * (2*pi*f)^2`` with
    f = 4 Hz. E4310 (the ankle base motor) has armature 0.0282528, giving 17.85
    Nm/rad before the armature ratio; the published (1.4, 0.4) then yields
    24.98 pitch / 7.14 roll.
    """
    act = ACTUATOR.read_text()
    assert "armature: float = 0.0282528" in act, "E4310 armature changed"
    assert "armature_ratio: tuple[float, float] = (2.0, 2.0)" in act, (
        "the wrapper's class default changed; re-verify the derived gains"
    )
    assert "class BoosterJointE4310" in act


def test_the_pinned_submodule_commit_contains_the_fix():
    """The PINNED submodule commit must carry the fix, not just the working tree.

    This is the guard for the actual failure. ``isaac_tasks/booster_train_ref`` is
    a git submodule, and the armature_ratio fix was originally made only in its
    working tree. Every test in this file passed locally while the training box
    ran ``git reset --hard origin/main``, which restored the submodule to a commit
    WITHOUT the fix -- so the box silently trained on the wrong ankle gains
    (35.69/35.69 instead of 24.98/7.14) and nobody noticed until the two
    environments were compared.

    Checking the file on disk cannot catch that: locally the file was correct.
    This reads what the parent repo actually pins, which is what a fresh clone and
    every training host will check out.
    """
    import subprocess

    pinned = subprocess.run(
        ["git", "ls-tree", "HEAD", _SUBMODULE],
        cwd=_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert len(pinned) >= 3 and pinned[1] == "commit", (
        f"could not read the pinned submodule commit for {_SUBMODULE}: {pinned!r}"
    )
    sha = pinned[2]

    # The fix must be present IN that commit, not merely in the working tree.
    blob = subprocess.run(
        ["git", "-C", str(_ROOT / _SUBMODULE), "show", f"{sha}:source/booster_train/"
         "booster_train/assets/robots/booster.py"],
        capture_output=True, text=True,
    )
    if blob.returncode != 0:
        pytest.skip(f"submodule commit {sha[:8]} is not fetched locally")
    assert blob.stdout.count("armature_ratio=(1.4, 0.4)") == 2, (
        f"submodule commit {sha[:8]} does NOT contain the ankle armature_ratio fix. "
        "The training hosts check out this commit, so they would silently run the "
        "wrong gains. Commit and push the fix inside the submodule, then bump the "
        "pointer here."
    )


def test_working_tree_matches_the_pinned_submodule():
    """No uncommitted submodule edits -- they will vanish on the next clone/reset."""
    import subprocess

    dirty = subprocess.run(
        ["git", "-C", str(_ROOT / _SUBMODULE), "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert not dirty, (
        "the booster_train submodule has uncommitted changes; they are invisible to "
        f"the parent repo and are lost on any clone or reset:\n{dirty}"
    )


def test_the_cause_is_documented_where_the_fix_lives():
    """The regression must be explained next to the corrected values.

    A silent restoration of a magic number is how the next person re-breaks it.
    """
    src = BOOSTER.read_text()
    assert "RESTORED from upstream" in src, (
        "the armature_ratio restoration must say where the value came from"
    )
    cfg = CFG.read_text()
    assert "manufacturer" in cfg.lower(), (
        "velocity_env_cfg must record that gains are now the manufacturer's"
    )