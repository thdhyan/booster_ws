"""Scene-level contract for the box-push env: cell pitch and per-env ground.

Both of these were wrong before this file existed.

ENV PITCH
---------
``env_spacing`` was 2.5 m while ``reset_box`` spawns the box in an annulus
r in [0.9, 1.4] m with FULL direction and ``randomize_box_geometry`` scales the
1.0 m prototype by 0.7-1.5x. Two boxes on the line between adjacent env origins
therefore sat 2.5 - 2.8 = **-0.3 m apart at their centres** -- interpenetrating by
0.3 m plus 1.5 m of combined extent. The pushed box was merged with a neighbour's and
every corner/goal reward was measured against geometry another env was also pushing.

PER-ENV GROUND
--------------
There was one world ground shared by every env, so "each cell has its own friction"
was not expressible. Adding a per-env patch is only half the fix, and the half that
is easy to get wrong: the frozen base's 187-ray privileged height scan raycasts
``/World/ground``. If the patch is not in ``mesh_prim_paths``, the scan keeps
measuring the surface the robot is no longer standing on -- no crash, just a
systematically wrong terrain height handed to a policy that cannot compensate for it
because it is frozen.
"""
from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/push/push_env_cfg.py"
MDP = _ROOT / "isaac_tasks/k1_velocity/source/k1_velocity/tasks/push/push_mdp.py"

#: reset_box annulus outer radius, and the 1.0 m prototype scaled by 1.5.
BOX_REACH_M = 1.4
BOX_HALF_EXTENT_M = 0.75
#: spacing that keeps the two worst-case boxes from touching, with 50% margin.
MIN_SAFE_SPACING_M = 2 * (BOX_REACH_M + BOX_HALF_EXTENT_M)  # 4.3 m


def _cfg_src() -> str:
    return CFG.read_text()


def _scene_class() -> ast.ClassDef:
    for node in ast.walk(ast.parse(_cfg_src())):
        if isinstance(node, ast.ClassDef) and node.name == "K1PushSceneCfg":
            return node
    raise AssertionError("K1PushSceneCfg not found")


def _post_init_src() -> str:
    tree = ast.parse(_cfg_src())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "K1PushEnvCfg":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__post_init__":
                    return ast.unparse(sub)
    raise AssertionError("K1PushEnvCfg.__post_init__ not found")


# --------------------------------------------------------------------- spacing


def test_the_spacing_arithmetic_is_what_we_think():
    assert MIN_SAFE_SPACING_M == 4.3


def test_env_spacing_clears_the_worst_case_boxes():
    """2.5 m put neighbouring boxes -0.3 m apart at their centres."""
    src = _post_init_src()
    assert "env_spacing = 8.0" in src, (
        f"env_spacing must be at least {MIN_SAFE_SPACING_M} m; 2.5 m let boxes "
        "interpenetrate with the 1.4 m annulus spawn and 0.75 m half-extent"
    )
    # and it must actually be applied to the scene, not just mentioned
    assert "self.scene.env_spacing" in src


def test_the_patch_is_sized_to_the_cell():
    src = _post_init_src()
    assert "self.scene.ground_patch.spawn.size" in src, (
        "the patch must be resized to the new cell, or it no longer covers the floor"
    )
    assert "env_spacing - 0.1" in src


# ---------------------------------------------------------------- ground patch


def test_scene_has_a_per_env_ground_patch():
    cls = _scene_class()
    names = {t.id for s in cls.body if isinstance(s, ast.Assign)
             for t in s.targets if isinstance(t, ast.Name)}
    assert "ground_patch" in names, "no per-env ground patch in the scene"
    assert "{ENV_REGEX_NS}" in _cfg_src().split("ground_patch = ", 1)[1][:200], (
        "the patch must be per-env, not one shared prim"
    )


def test_patch_top_face_is_the_trained_standing_height():
    """The frozen base was trained standing at z=0; the patch top must be z=0."""
    body = _cfg_src().split("ground_patch = RigidObjectCfg", 1)[1].split("\n    )", 1)[0]
    assert "GROUND_PATCH_THICKNESS * 0.5" in body, (
        "a cuboid centred at -t/2 has its top face at z=0; anything else moves the contact "
        "surface away from the height the frozen base expects"
    )


def test_the_world_ground_is_dropped_so_the_patch_is_the_contact_surface():
    """Two colliders sharing a plane make the contact normal solver-dependent."""
    g = _cfg_src().split("ground = AssetBaseCfg", 1)[1].split("\n    )", 1)[0]
    assert "GROUND_PATCH_THICKNESS" in g, (
        "the world ground must sit one patch-thickness below z=0, otherwise the patch "
        "and the world plane are coincident and z-fight"
    )


def _scene_assign(cls_name, target):
    """The AST of the single assignment to `target` inside `cls_name`."""
    for node in ast.walk(ast.parse(_cfg_src())):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if isinstance(sub, ast.Assign) and any(
                    getattr(t, "id", "") == target for t in sub.targets
                ):
                    return sub.value
    raise AssertionError(f"{target} not assigned in {cls_name}")


def _kwargs(call):
    return {k.arg: k.value for k in call.keywords if k.arg}


def _dict_get(node, key):
    """Value for `key` in an ast.Dict, without literal_eval (params hold Calls)."""
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    raise AssertionError(f"{key} not found in dict literal")


def _call_str_arg(node):
    """`SceneEntityCfg("ground_patch")` -> "ground_patch"."""
    if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
        return node.args[0].value
    return None


def test_height_scanner_sees_the_patch_not_just_the_backstop():
    """The silent-corruption case: a frozen base reading the wrong surface."""
    call = _scene_assign("K1PushSceneCfg", "height_scanner")
    paths = ast.literal_eval(_kwargs(call)["mesh_prim_paths"])
    assert any("ground_patch" in p for p in paths), (
        "the patch must be in the scanner's mesh_prim_paths; otherwise the frozen base's "
        "187-ray obs measures the backstop 5 cm below its feet"
    )
    idx_patch = next(i for i, p in enumerate(paths) if "ground_patch" in p)
    idx_world = next(i for i, p in enumerate(paths) if p == "/World/ground")
    assert idx_patch < idx_world, (
        "the patch is the real contact surface and should be listed first"
    )


def _event_term(term_name):
    """The EventTerm call assigned to `term_name` in K1PushEventCfg."""
    for node in ast.walk(ast.parse(_cfg_src())):
        if isinstance(node, ast.ClassDef) and node.name == "K1PushEventCfg":
            for sub in node.body:
                if isinstance(sub, ast.Assign) and any(
                    getattr(t, "id", "") == term_name for t in sub.targets
                ):
                    return sub.value
    raise AssertionError(f"event term {term_name} not found")


def test_ground_friction_is_randomised_per_env():
    """The actual ask: each cell gets its own friction."""
    call = _event_term("randomize_ground_friction")
    kw = _kwargs(call)
    assert ast.literal_eval(kw["mode"]) == "startup", (
        "mode='startup' fires just after sim play, which is when the material impls can "
        "see asset.root_view; prestartup cannot write materials"
    )
    name = _call_str_arg(_dict_get(kw["params"], "asset_cfg"))
    assert name == "ground_patch", (
        "the ground friction must be written to the patch, not to the shared world plane"
    )


def test_ground_friction_lower_bound_respects_the_frozen_base():
    """0.5 was measured to drop the frozen base; do not quietly widen below it.

    push_env_cfg's ground comment records the failure (home@cmd=0 stands, this scene
    falls in ~17 steps at the GroundPlaneCfg default of 0.5, chain2
    DIAG_FAILED_BASE_STILL_FALLS done_rate=0.0425). The base is frozen, so a slick
    floor is an unrecoverable early termination rather than a robustness gain.
    """
    lo = 0.7
    params = _kwargs(_event_term("randomize_ground_friction"))["params"]
    for key in ("static_friction_range", "dynamic_friction_range"):
        rng = ast.literal_eval(_dict_get(params, key))
        assert rng[0] >= lo, (
            f"{key} lower bound {rng[0]} is below the {lo} floor; 0.5 was measured to drop "
            "the FROZEN base (DIAG_FAILED_BASE_STILL_FALLS). Widen deliberately."
        )


def test_box_annulus_matches_what_the_spacing_assumes():
    """If reset_box's spawn radius grows, MIN_SAFE_SPACING_M must grow with it."""
    body = MDP.read_text().split("def reset_box", 1)[1].split("\ndef ", 1)[0]
    assert "1.4" in body, (
        "reset_box no longer uses the 1.4 m annulus the spacing arithmetic assumes; "
        "recompute MIN_SAFE_SPACING_M or boxes will interpenetrate again"
    )