"""Generate per-task REWARDS.md + ARCHITECTURE.png in each task folder.

Reward/termination/curriculum tables come from a training run's dumped
``params/env.yaml`` (what the run actually trained with, not the current code),
with each term's description taken from its function docstring. Architecture
figures are drawn with graphviz from the SPECS below (layer sizes verified
against the checkpoints' state-dict shapes, 2026-10-11).

Usage (laptop, needs graphviz `dot`):
    rsync <run>/params/{env,agent}.yaml k1m_out/params/<run_name>/
    uv run --no-project --with pyyaml python isaac_tasks/k1_velocity/scripts/task_docs.py
"""
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks"
SRC = ROOT / "isaac_tasks/k1_velocity/source"
PARAMS = ROOT / "k1m_out/params"


class _L(yaml.SafeLoader):
    pass


_L.add_multi_constructor("", lambda l, s, n: l.construct_sequence(n) if isinstance(n, yaml.SequenceNode)
                         else l.construct_mapping(n) if isinstance(n, yaml.MappingNode) else l.construct_scalar(n))

# Isaac Lab built-ins (no local source to read a docstring from)
BUILTIN = {
    "flat_orientation_l2": "Squared xy of projected gravity (trunk tilt). Penalty.",
    "action_rate_l2": "Squared change of the action between steps. Penalty.",
    "joint_torques_l2": "Squared joint torques on the selected joints. Penalty.",
    "joint_pos_limits": "Joint positions beyond soft limits. Penalty.",
    "joint_vel_limits": "Joint velocities beyond soft limits. Penalty.",
    "joint_vel_l2": "Squared joint velocities. Penalty.",
    "joint_acc_l2": "Squared joint accelerations. Penalty.",
    "applied_torque_limits": "Gap between computed and applied (clipped) torque. Penalty.",
    "body_lin_acc_l2": "Squared linear acceleration of the selected bodies. Penalty.",
    "lin_vel_z_l2": "Squared vertical trunk velocity. Penalty.",
    "ang_vel_xy_l2": "Squared roll/pitch rate of the trunk. Penalty.",
    "is_terminated": "1 on ANY termination except time-out. Penalty.",
    "is_alive": "1 every step the episode is running.",
    "feet_slide": "Foot xy speed while the foot is in contact. Penalty.",
    "undesired_contacts": "Count of bodies (not feet) touching the ground over a force threshold. Penalty.",
    "time_out": "Episode length reached (not a failure).",
    "bad_orientation": "Trunk tilt beyond limit_angle (rad).",
    "root_height_below_minimum": "Trunk below a fixed height.",
    "illegal_contact": "Selected bodies touch the ground.",
    "terrain_levels_vel": "Move envs up/down the terrain grid by distance walked.",
    "track_lin_vel_xy_exp": "exp(-|v_xy - cmd_xy|^2 / std^2).",
    "track_ang_vel_z_exp": "exp(-(w_z - cmd_wz)^2 / std^2).",
}

# Architecture specs. nodes: id -> (label, kind); edges: (src, dst, label). "obs" is the input box.
_SQUAT_OBS = [("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3), ("cmd vx,vy,wz,H*", 4),
              ("joint_pos 12 legs", 12), ("joint_vel 12 legs", 12), ("last action", 12),
              ("height_scan (privileged)", 187), ("phase_clock", 2)]
_SKILL_OBS = [("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3), ("leg_pos", 12), ("leg_vel", 12),
              ("head_pos", 2), ("last action", 17), ("ball_pos", 3), ("ball_vel", 3), ("head_err", 2)]
_PUSH_OBS = [("box_state (box + goal corners, ...)", 68), ("base_lin_vel", 3), ("base_ang_vel", 3),
             ("projected_gravity", 3), ("wrist_targets", 6), ("arm_joint_pos", 8), ("arm_joint_vel", 8), ("last action", 10)]
_STUDENT = ("frozen velocity student\nk1_velocity_student_gaitfix.pt\nMLP 500(50x10 hist)-512-256-128-12", "frozen")


def _residual(scale):
    return {"actor": None, "base": _STUDENT,
            "cmd": ("vx, vy, wz", "base"), "res": (f"12 leg residual x {scale}", "residual"),
            "sum": ("legs = student + residual\n12 leg targets -> PD", "PD"), "head": ("head yaw/pitch (2) -> PD", "PD")}, [
        ("obs", "actor", ""), ("actor", "cmd", ""), ("actor", "res", ""), ("actor", "head", ""),
        ("cmd", "base", "command"), ("base", "sum", ""), ("res", "sum", "+")]


SPECS = {
    "velocity/squat_teacher": dict(
        title="Squat teacher (Isaac-Velocity-Squat-K1-v0): PPO from scratch", run="k1_squat_teacher",
        inputs=_SQUAT_OBS,
        nodes={"actor": ("actor MLP 238-512-256-128-12 ELU", "trained"), "legs": ("12 leg joint targets -> PD", "PD")},
        edges=[("obs", "actor", ""), ("actor", "legs", "")],
        critic="critic MLP 238-512-256-128-1 (teacher obs)",
        note="Not residual, not decoupled. One actor for the 12 leg joints; head and arms held at default.",
    ),
    "velocity/squat_residual": dict(
        title="Squat residual (Isaac-Velocity-Squat-Residual-K1-v0): PPO, RESIDUAL on frozen velocity student",
        run="k1_squat_residual", inputs=_SQUAT_OBS,
        nodes={"actor": ("actor MLP 238-512-256-128-12 ELU", "trained"), "base": _STUDENT,
               "cmd": ("env command vx, vy, wz\n(not from the actor)", "base"),
               "res": ("12 leg residual x 0.5", "residual"),
               "sum": ("legs = student + residual\n12 leg targets -> PD", "PD")},
        edges=[("obs", "actor", ""), ("actor", "res", ""), ("cmd", "base", "command"), ("base", "sum", ""),
               ("res", "sum", "+")],
        critic="critic MLP 238-512-256-128-1 (teacher obs)",
        note="Residual: the frozen student walks the env velocity command; the residual adds squat depth (H*) and "
             "feet_flat. Actor init std 0.3.",
    ),
    "velocity/squat_student": dict(
        title="Squat student (Isaac-Velocity-Squat-K1-Distill-v0): distillation", run=None,
        inputs=[("policy obs x 10-step history", 510), ("(lin/ang vel, gravity, cmd 4, leg pos/vel,", 0),
                (" last action, phase = 51 per step; no height scan)", 0)],
        nodes={"actor": ("student MLP 510-512-256-128-12 ELU", "trained"),
               "teacher": ("squat teacher MLP 238-512-256-128-12", "frozen"),
               "legs": ("12 leg joint targets -> PD", "PD")},
        edges=[("obs", "actor", ""), ("actor", "legs", ""), ("teacher", "actor", "action targets (MSE)")],
        critic=None,
        note="Distillation (DAgger-style behaviour cloning) of the frozen squat teacher; rewards are only logged.",
    ),
    "velocity/velocity_student": dict(
        title="Velocity student gaitfix (models/k1_velocity_student_gaitfix.pt): frozen base of track/kick",
        run=None,
        inputs=[("policy obs x 10-step history", 500), ("(as squat student but cmd vx,vy,wz = 50 per step)", 0)],
        nodes={"actor": ("student MLP 500-512-256-128-12 ELU", "frozen in skills"),
               "teacher": ("P2 velocity teacher MLP 237-512-256-128-12", "frozen"),
               "legs": ("12 leg joint targets -> PD", "PD")},
        edges=[("obs", "actor", ""), ("actor", "legs", ""), ("teacher", "actor", "distilled from")],
        critic=None, note="Used FROZEN by the residual skills (ball track, kick).",
    ),
    "push/push_single": dict(
        title="Push (Isaac-Push-SG-K1-v0): PPO, command policy on frozen squat base", run="p6_push",
        inputs=_PUSH_OBS,
        nodes={"actor": ("actor MLP 109-512-256-128-10 ELU", "trained"),
               "cmd": ("vx, vy, wz, H* (4)", "base"),
               "base": ("frozen base k1_push_base.pt\n(= squat teacher JIT 238-512-256-128-12)", "frozen"),
               "legs": ("12 leg joint targets -> PD", "PD"),
               "ik": ("wrist L/R xyz deltas (3+3)\n-> differential IK -> 8 arm joints", "IK")},
        edges=[("obs", "actor", ""), ("actor", "cmd", ""), ("actor", "ik", ""), ("cmd", "base", "command"),
               ("base", "legs", "")],
        critic="critic MLP 109-512-256-128-1",
        note="Hierarchical, NOT residual: the legs come only from the frozen base. Arms by IK.",
    ),
    "push/push_simple": dict(
        title="Push simple rewards (Isaac-Push-Simple-K1-v0): PPO, command policy on frozen squat base",
        run="p6_push_simple", inputs=_PUSH_OBS,
        nodes={"actor": ("actor MLP 109-512-256-128-10 ELU", "trained"),
               "cmd": ("vx, vy, wz, H* (4)", "base"),
               "base": ("frozen base k1_push_base.pt\n(= squat teacher JIT 238-512-256-128-12)", "frozen"),
               "legs": ("12 leg joint targets -> PD", "PD"),
               "ik": ("wrist L/R xyz deltas (3+3)\n-> differential IK -> 8 arm joints", "IK")},
        edges=[("obs", "actor", ""), ("actor", "cmd", ""), ("actor", "ik", ""), ("cmd", "base", "command"),
               ("base", "legs", "")],
        critic="critic MLP 109-512-256-128-1",
        note="Same network as push_single; only the rewards changed: hands to box, box progress (m/s), box near "
             "goal, success bonus, fall/box-fail penalty. Box spawn radius and goal distance on curricula.",
    ),
    "push/push_mappo": dict(
        title="Push multi-robot (Isaac-Push-Multi-K1-v0): skrl MAPPO, one policy per robot", run="p6_push_mappo",
        inputs=[("robot_k's own obs (same 109 as single push)", 109)],
        nodes={"actor": ("policy_0 MLP 109-512-256-128-10", "trained"),
               "actor1": ("policy_1 MLP 109-512-256-128-10", "trained"),
               "b0": ("robot_0: cmd -> frozen squat base; wrist IK", "base"),
               "b1": ("robot_1: cmd -> frozen squat base; wrist IK", "base"),
               "v": ("value_k MLP 218 (both robots' obs)-512-256-128-1\n(centralized critic, one per robot)", "critic")},
        edges=[("obs", "actor", "robot_0 obs"), ("obs", "actor1", "robot_1 obs"), ("actor", "b0", ""),
               ("actor1", "b1", ""), ("obs", "v", "both")],
        critic=None, note="skrl MAPPO (thdhyan/skrl fork), not rsl_rl. Decoupled per robot (separate weights), centralized critics. Rewards: team box terms + per-robot hands/fall terms (RewardSplit, _k suffix).",
    ),
    "skills/track": dict(
        title="Ball track (p8_balltrack_residual): PPO, RESIDUAL on frozen velocity student",
        run="p8_balltrack_residual", inputs=_SKILL_OBS,
        critic="critic MLP 60-512-256-128-1",
        note="Residual: legs = student(cmd) + 0.1 * delta. Head joints are direct.",
    ),
    "kick/kick": dict(
        title="Kick (p9_kick_residual): PPO, RESIDUAL on frozen velocity student", run="p9_kick_residual",
        inputs=_SKILL_OBS + [("goal_pos", 2)],
        critic="critic MLP 62-512-256-128-1",
        note="Residual: legs = student(cmd) + 0.3 * delta. Goal fixed at (4, 0) (FIELD_L/2); no curriculum.",
    ),
    "carry/carry": dict(
        title="Carry (Isaac-Carry-K1-v0): PPO from scratch, DECOUPLED legs/arms actor", run="p7_carry",
        inputs=[("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3), ("derived cmd", 3),
                ("joint_pos legs", 12), ("joint_vel legs", 12), ("last action", 20), ("phase_clock", 2),
                ("goal x, y, dist", 3), ("arm_joint_pos", 8), ("arm_joint_vel", 8), ("box_params (teacher)", 2)],
        nodes={"actor": ("leg head MLP 79-512-256-128-12", "trained"),
               "arms": ("arm head MLP 79-512-256-128-8", "trained"),
               "legs": ("12 leg joint targets -> PD", "PD"), "armj": ("8 arm joint targets (x0.25) -> PD", "PD")},
        edges=[("obs", "actor", ""), ("obs", "arms", ""), ("actor", "legs", ""), ("arms", "armj", "")],
        critic="critic MLP 79-512-256-128-1",
        note="Decoupled: two heads, no shared weights, one PPO update. Not residual (no frozen base).",
    ),
}
for _k, _s in (("skills/track", 0.1), ("kick/kick", 0.3)):
    _n, _e = _residual(_s)
    _n["actor"] = (f"actor MLP {sum(d for _, d in SPECS[_k]['inputs'])}-512-256-128-17 ELU", "trained")
    SPECS[_k].update(nodes=_n, edges=_e)

COLORS = {"critic": "#eeeeee", "frozen in skills": "#cfd8dc", "trained": "#c8e6c9", "frozen": "#cfd8dc", "base": "#ffe0b2", "residual": "#f8bbd0", "PD": "#bbdefb", "IK": "#d1c4e9"}


def _doc(func: str) -> str:
    mod, name = func.split(":")
    if not mod.startswith("k1_velocity"):
        return BUILTIN.get(name, "Isaac Lab built-in.")
    path = SRC / (mod.replace(".", "/") + ".py")
    if not path.exists():
        return ""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            d = ast.get_docstring(node) or ""
            return " ".join(d.split("\n\n")[0].split())
    return ""


def _fmt_params(p) -> str:
    if not p:
        return ""
    if "inner" in p:  # carry's gated wrapper
        return f"gated({p['inner'].split(':')[1]}, {_fmt_params(p.get('inner_params'))})"
    keep = {k: v for k, v in p.items() if not isinstance(v, dict)}
    for k, v in p.items():
        if isinstance(v, dict) and "name" in v:
            keep[k] = v["name"] + (f"[{','.join(map(str, v.get('body_names') or v.get('joint_names') or []))}]"
                                   if (v.get("body_names") or v.get("joint_names")) else "")
    return ", ".join(f"{k}={v}" for k, v in keep.items())


def _func_name(term) -> str:
    f = term["func"]
    p = term.get("params") or {}
    return p["inner"] if "inner" in p else f


def reward_md(run: str) -> str:
    e = yaml.load(open(PARAMS / run / "env.yaml"), _L)
    out = [f"Source: run `{run}` params/env.yaml (what the run trained with). Episode {e['episode_length_s']} s, "
           f"policy dt {e['sim']['dt'] * e['decimation']:.3f} s. Isaac Lab multiplies every weight by dt; "
           "`Episode_Reward/<term>` in TensorBoard is the episode sum divided by the episode length in seconds.\n",
           "| term | weight | function | params | what it computes (docstring) |", "|---|---|---|---|---|"]
    for k, v in (e.get("rewards") or {}).items():
        if not v:
            continue
        f = _func_name(v)
        out.append(f"| `{k}` | {v['weight']} | `{f.split(':')[1]}` ({f.split(':')[0].split('.')[-1]}) | "
                   f"{_fmt_params(v.get('params'))} | {_doc(f)} |")
    out += ["", "### Terminations", "", "| term | function | params | meaning |", "|---|---|---|---|"]
    for k, v in (e.get("terminations") or {}).items():
        if v:
            out.append(f"| `{k}` | `{v['func'].split(':')[1]}` | {_fmt_params(v.get('params'))} | {_doc(v['func'])} |")
    out += ["", "### Curriculum", ""]
    cur = {k: v for k, v in (e.get("curriculum") or {}).items() if v}
    if not cur:
        out.append("None.")
    for k, v in cur.items():
        out.append(f"- `{k}`: `{v['func'].split(':')[1]}` {_fmt_params(v.get('params'))}. {_doc(v['func'])}")
    return "\n".join(out)


def figure(spec: dict, png: Path):
    q = lambda t: t.replace('"', "'")
    lines = ["digraph G {", 'rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
             f'labelloc=t; label="{q(spec["title"])}"; fontsize=15; fontname="Helvetica-Bold";']
    obs = "\\l".join(f"{n}  [{d}]" if d else n for n, d in spec["inputs"]) + "\\l"
    total = sum(d for _, d in spec["inputs"])
    lines.append(f'obs [label="Inputs ({total})\\n{obs}", fillcolor="#fff9c4"];')
    for nid, (label, kind) in spec["nodes"].items():
        lines.append(f'{nid} [label="{q(label)}\\n[{kind}]", fillcolor="{COLORS[kind]}"];')
    for a, b, lab in spec["edges"]:
        lines.append(f'{a} -> {b} [label="{lab}"];')
    if spec.get("critic"):
        lines.append(f'critic [label="{spec["critic"]}", fillcolor="#eeeeee", style="rounded,dashed,filled"];')
        lines.append('obs -> critic [style=dashed];')
    lines.append(f'note [shape=note, label="{q(spec["note"])}", fillcolor="#ffffff"];')
    lines.append("}")
    png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dot", "-Tpng", "-Gdpi=110", "-o", str(png)], input="\n".join(lines).encode(), check=True)


def main():
    by_folder: dict[str, list[str]] = {}
    for key in SPECS:
        by_folder.setdefault(key.split("/")[0], []).append(key)
    for folder, keys in by_folder.items():
        md = [f"# {folder}: models, inputs and reward terms", "",
              "Generated by `isaac_tasks/k1_velocity/scripts/task_docs.py`. Re-run after changing rewards.", "",
              "Legend: green = trained, grey = frozen, orange = command to a frozen base, pink = residual added "
              "to the frozen base's legs, blue = joint targets to PD, purple = arm IK.", ""]
        for key in keys:
            spec = SPECS[key]
            png = TASKS / folder / "figures" / f"{key.split('/')[1]}.png"
            figure(spec, png)
            md += [f"## {spec['title']}", "", f"![{key}](figures/{png.name})", "", spec["note"], ""]
            if spec["run"] and (PARAMS / spec["run"] / "env.yaml").exists():
                md += ["### Reward terms", "", reward_md(spec["run"]), ""]
        notes = TASKS / folder / "REWARDS_NOTES.md"
        if notes.exists():
            md += ["---", "", notes.read_text()]
        (TASKS / folder / "REWARDS.md").write_text("\n".join(md) + "\n")
        print("wrote", TASKS / folder / "REWARDS.md")


if __name__ == "__main__":
    main()
