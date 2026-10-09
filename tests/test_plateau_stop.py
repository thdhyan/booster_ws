"""Tests for the reward-plateau early stopper.

The whole value of this script is NOT firing on a run that is still improving, and
firing on one that is not. Both failure modes are cheap to simulate and expensive to
discover on a live 40-hour run, so they are pinned here.

The tempting wrong implementation is ``reward[-1] <= reward[-2]``. RSL-RL's
per-iteration reward is dominated by which envs happened to fall early, so that
version fires within tens of iterations on a perfectly healthy run -- which would
make this tool actively dangerous.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = _ROOT / "isaac_tasks/k1_velocity/scripts/plateau_stop.py"

_spec = importlib.util.spec_from_file_location("plateau_stop", SCRIPT)
M = importlib.util.module_from_spec(_spec)
sys.modules["plateau_stop"] = M
_spec.loader.exec_module(M)


def _ramp(n, start=0.0, slope=1.0, noise=0.0):
    """Still-improving reward with optional jitter."""
    return [start + slope * i + noise * ((i % 2) * 2 - 1) for i in range(n)]


# ------------------------------------------------------------------ never fires


def test_a_steadily_improving_run_is_never_called_plateau():
    """The critical safety property: a healthy run must run to its cap."""
    done, why = M.is_plateau(_ramp(4000, slope=1.0), window=200, min_delta=0.5, patience=3)
    assert done is False, why


def test_flat_but_noisy_data_does_count_as_a_plateau():
    """A flat window MEAN is a plateau even when the individual iterations are noisy.

    Worth stating explicitly because it looks like the wrong answer: RSL-RL's
    per-iteration reward is dominated by which envs fell early, so a real flat run
    looks nothing like a constant series. What is being tested here is that the
    detector reads the window MEAN -- so jitter that cancels out does not hide a
    plateau. The safety property is the opposite test: real drift clears min_delta.
    """
    import random
    rng = random.Random(7)
    noisy = [rng.gauss(0.0, 50.0) for _ in range(4000)]
    done, why = M.is_plateau(noisy, window=200, min_delta=0.5, patience=3)
    assert done is True, f"a flat mean hidden by noise should still be caught: {why}"


def test_drift_larger_than_min_delta_clears_each_window():
    """The complementary safety test, with realistic jitter rather than a clean ramp."""
    import random
    rng = random.Random(11)
    drift = 10.0  # per iteration: far exceeds min_delta/window
    s = [drift * i + rng.gauss(0.0, 20.0) for i in range(4000)]
    done, why = M.is_plateau(s, window=200, min_delta=0.5, patience=3)
    assert done is False, f"real progress was called a plateau: {why}"


def test_a_short_run_is_not_called_plateau():
    """Too few windows to judge -- warmup must never be mistaken for convergence."""
    done, why = M.is_plateau(_ramp(300), window=200, patience=3)
    assert done is False
    assert "need 4" in why


def test_empty_and_tiny_inputs_are_safe():
    assert M.is_plateau([], window=200, patience=3)[0] is False
    assert M.is_plateau([1.0], window=200, patience=3)[0] is False


# -------------------------------------------------------------------- fires


def test_a_genuinely_flat_tail_is_caught():
    flat = _ramp(1200, slope=1.0) + [1200.0] * 1200
    done, why = M.is_plateau(flat, window=200, min_delta=0.5, patience=3)
    assert done is True, why


def test_a_single_bad_window_is_not_enough():
    """One dip then recovery must NOT stop the run -- that is what patience buys."""
    s = _ramp(2000, slope=1.0)
    s[600:800] = [-100.0] * 200  # one collapsed window
    s[800:] = [900.0 + 2.0 * i for i in range(1200)]  # then real progress
    done, why = M.is_plateau(s, window=200, min_delta=0.5, patience=3)
    assert done is False, f"a single dip triggered a stop: {why}"


def test_plateau_after_progress_is_detected():
    s = _ramp(1200, slope=1.0) + [1200.0] * 800
    assert M.is_plateau(s, window=200, min_delta=0.5, patience=3)[0] is True


def test_higher_patience_is_more_conservative():
    """Fewer tolerated stale windows must fire no later, never later-still."""
    # ramp for 6 windows (each improves), then a flat block. The flat block's FIRST
    # window still counts as an improvement (1200 > the ramp's last mean of 1099.5),
    # so 800 flat values give exactly 3 stale windows: patience 2 and 3 fire, 4 does not.
    s = _ramp(1200, slope=1.0) + [1200.0] * 800
    assert M.is_plateau(s, window=200, min_delta=0.5, patience=2)[0] is True
    assert M.is_plateau(s, window=200, min_delta=0.5, patience=3)[0] is True
    assert M.is_plateau(s, window=200, min_delta=0.5, patience=4)[0] is False


def test_an_unreachable_min_delta_treats_everything_as_stale():
    """Documented consequence, not a bug: if nothing can clear the bar, nothing can.

    min_delta has to be scaled to the reward's own units. Set it above the
    per-window improvement of a healthy run and every window counts as stale, so the
    detector stops the run at patience+1 windows regardless of progress.
    """
    s = _ramp(4000, slope=1.0)
    assert M.is_plateau(s, window=200, min_delta=1e9, patience=3)[0] is True


# ------------------------------------------------------------------ arg guards


def test_degenerate_parameters_are_rejected():
    with pytest.raises(ValueError):
        M.is_plateau(_ramp(1000), window=1, patience=3)
    with pytest.raises(ValueError):
        M.is_plateau(_ramp(1000), window=200, patience=0)


# ------------------------------------------------------------------- parsing


def test_parses_the_metric_key_and_survives_other_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "Learning iteration 1/100\n"
        "Reward/total: -12.5\n"
        "Reward/track_lin_vel_xy_exp: 1.18\n"
        "Reward/total: -3.25e+01\n"   # exponent form must parse
        "  total time: 1.2\n"          # must NOT be mistaken for the key
    )
    got = M.parse_series(str(log), "Reward/total")
    assert got == [-12.5, -3.25e01], got


def test_window_means_drops_the_trailing_partial_window():
    # 250 values with window 100 -> two full windows, the 50 left over is dropped
    assert M.window_means(list(range(250)), 100) == [49.5, 149.5]