"""Generate a native booster_k1 SOMA Retargeter target from the bundled t1 one.

K1 is the 22-DoF sibling of T1: same head (2), same 4-DoF arms, same 6-DoF legs.
T1-29dof adds 3-DoF wrists/hands per arm that K1 does not have, so those are
dropped, plus the Waist (T1-23dof only; the 29dof variant has none).
"""
import json, os, pathlib, shutil, sys

ROOT = pathlib.Path("soma_retargeter/assets/robotics/booster")
SRC, DST = ROOT / "t1", ROOT / "k1"

# --- joints K1 does not have -------------------------------------------------
DROP_JOINTS = {
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll", "Waist",
}
# --- links/segments keyed by the dropped joints ------------------------------
DROP_SEGMENTS = {"AL7", "AR7", "L7", "R7", "Waist"}

def strip(node):
    """Recursively drop Waist/*7 entries from any dict/list tree."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in DROP_JOINTS or k in DROP_SEGMENTS:
                continue
            if "Wrist" in str(k) or "Hand_Roll" in str(k):
                continue
            out[k] = strip(v)
        return out
    if isinstance(node, list):
        return [strip(v) for v in node]
    return node

DST.mkdir(parents=True, exist_ok=True)
(DST / "configs").mkdir(exist_ok=True)
(DST / "desc").mkdir(exist_ok=True)

# manifest
mf = json.loads((SRC / "manifest.json").read_text())
mf["name"] = "booster_k1"
mf["desc"]["urdf_path"] = "desc/K1_22dof.urdf"
mf["retarget_configs"]["soma"] = "configs/soma_to_booster_k1_retargeter_config.json"
(DST / "manifest.json").write_text(json.dumps(mf, indent=2) + "\n")

# retargeter config
rc = json.loads((SRC / "configs/soma_to_booster_t1_retargeter_config.json").read_text())
rc = strip(rc)
rc["human_robot_scaler_config"] = "configs/soma_to_booster_k1_scaler_config.json"
rc["post_processing"]["robot_config"] = "configs/booster_k1_post_processing_config.json"
(DST / "configs/soma_to_booster_k1_retargeter_config.json").write_text(
    json.dumps(rc, indent=2) + "\n")

# scaler config
sc = strip(json.loads((SRC / "configs/soma_to_booster_t1_scaler_config.json").read_text()))
(DST / "configs/soma_to_booster_k1_scaler_config.json").write_text(
    json.dumps(sc, indent=2) + "\n")

# post-processing config
pp = strip(json.loads((SRC / "configs/booster_t1_post_processing_config.json").read_text()))
cc = pp.get("contact_correction", {})
# K1 foot box sits at local (0.026, 0, -0.02) in the foot_link frame, so the
# sole points along the link's -Z. SOMA's default is +Z, which silently no-ops
# foot flattening if left wrong.
cc["sole_normal_local"] = [0.0, 0.0, -1.0]
pp["contact_correction"] = cc
(DST / "configs/booster_k1_post_processing_config.json").write_text(
    json.dumps(pp, indent=2) + "\n")

print("wrote", DST)
for p in sorted(DST.rglob("*.json")):
    print("  ", p.relative_to(DST))
