#!/usr/bin/env python3
# Copyright (c) 2026, Booster K1 soccer project. SPDX-License-Identifier: Apache-2.0
"""Static deployability test (PLAN_PHASE6_SOCCER_HRL / TASKS T6.3.5).

Deployability invariant (user requirement): **deployable observation groups never
read ground-truth ball/goal state.**  GT is allowed only in (a) reward functions,
(b) termination/event terms, and (c) obs groups explicitly marked privileged
("teacher").  Everything the real robot runs on ("policy" / student groups) must
see proprioception and *estimator* outputs only (YOLO vector, ball/goal estimator
buffers) — never ``scene["ball"]`` / ``scene["goal_bar_*"]``.

Mechanism (no simulator needed — pure config introspection):

1. Iterate every registered K1 task in the gym registry.
2. Load its env cfg class and inspect every observation group.
3. For each term, resolve the observation function and:
   - fail if its name is in the known GT-reader blacklist, or
   - fail if its source directly touches a GT scene entity
     (``scene["ball"]``, ``scene["goal_bar..."]``).
   Groups named ``teacher`` are exempt (privileged by definition).

Run:  ``python tests/test_deployability.py``   (exit 0 = invariant holds)
Also runs under pytest (``pytest tests/test_deployability.py``).
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys

# --- blacklist: known ground-truth observation functions (kick/mdp.py) --------
GT_FUNC_NAMES = {
    "ball_pos_in_robot_frame",
    "ball_lin_vel_in_robot_frame",
    "goal_pos_in_robot_frame",
    "goal_distance",
    "goal_scored",
    "ball_to_goal_progress",
}

# --- blacklist: source patterns that reach straight into GT scene entities ----
GT_SCENE_PATTERNS = [
    re.compile(r'scene\[\s*["\']ball["\']\s*\]'),
    re.compile(r'scene\[\s*["\']goal_bar(_pos|_neg)?["\']\s*\]'),
]

PRIVILEGED_GROUPS = {"teacher"}  # obs groups allowed to see GT / height scans


def _iter_groups(env_cfg):
    """Yield (group_name, group_cfg) for each observation group on the cfg."""
    obs = getattr(env_cfg, "observations", None)
    if obs is None:
        return
    for name, val in vars(obs).items():
        if name.startswith("_"):
            continue
        if hasattr(val, "__class__") and val.__class__.__name__.endswith("Cfg"):
            # a group cfg holds ObsTerm fields
            terms = {k: t for k, t in vars(val).items() if not k.startswith("_") and hasattr(t, "func")}
            if terms:
                yield name, terms


def _violations(group_name: str, terms: dict) -> list[str]:
    out = []
    if group_name in PRIVILEGED_GROUPS:
        return out
    for term_name, term in terms.items():
        func = term.func
        fname = getattr(func, "__name__", str(func))
        why = None
        if fname in GT_FUNC_NAMES:
            why = f"GT blacklist function {fname!r}"
        else:
            try:
                src = inspect.getsource(func)
            except (TypeError, OSError):
                src = ""
            for pat in GT_SCENE_PATTERNS:
                if pat.search(src):
                    why = f"{fname}() reads GT scene entity ({pat.pattern})"
                    break
        if why:
            out.append(f"obs group {group_name!r} term {term_name!r}: {why}")
    return out


def main() -> int:
    # register all task families (velocity, basic, kick, ...)
    from k1_velocity.register_tasks import register_tasks

    register_tasks()

    import gymnasium as gym

    task_ids = sorted(k for k in gym.registry if "K1" in k and "Play" not in k)
    if not task_ids:
        print("FATAL: no K1 tasks in gym registry")
        return 1

    all_violations: list[str] = []
    print(f"{'task':38s} {'group':10s} {'terms':>5s}  status")
    print("-" * 72)
    for task_id in task_ids:
        spec = gym.spec(task_id)
        entry = spec.kwargs["env_cfg_entry_point"]  # "module:Class"
        mod_name, cls_name = entry.split(":")
        mod = importlib.import_module(mod_name)
        env_cfg = getattr(mod, cls_name)()
        groups = list(_iter_groups(env_cfg))
        if not groups:
            all_violations.append(f"{task_id}: no observation groups found")
            print(f"{task_id:38s} {'-':10s} {'-':>5s}  FAIL (no obs groups)")
            continue
        for gname, terms in groups:
            viols = _violations(gname, terms)
            all_violations.extend(f"{task_id}: {v}" for v in viols)
            status = "OK" if not viols else "FAIL"
            priv = " (privileged)" if gname in PRIVILEGED_GROUPS else ""
            print(f"{task_id:38s} {gname:10s} {len(terms):5d}  {status}{priv}")
            for v in viols:
                print(f"    !! {v}")

    print("-" * 72)
    if all_violations:
        print(f"DEPLOYABILITY INVARIANT VIOLATED ({len(all_violations)} finding(s)):")
        for v in all_violations:
            print(f"  - {v}")
        return 1
    print(f"DEPLOYABILITY INVARIANT HOLDS for {len(task_ids)} task(s).")
    return 0


# pytest entry points ----------------------------------------------------------
def test_deployability_invariant():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
