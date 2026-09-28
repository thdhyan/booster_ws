# SOMA Retargeter → Booster K1 (route B)

Branch: `feat/retarget-soma-k1` · worktree: `~/Projects/wt-soma`

## Why SOMA Retargeter runs on `dl`, not the sparks

`usd-core==26.3` is a hard pin in SOMA's `pyproject.toml` and NVIDIA publishes
wheels only for `manylinux_2_27_x86_64`, `manylinux_2_28_x86_64`,
`macosx_10_15_universal2` and `win_amd64`. There is **no aarch64 wheel**, so
`soma-retargeter` cannot be installed on any GB10 spark:

```
× Because usd-core==26.3 has no wheels with a matching platform tag
    (e.g. `manylinux_2_39_aarch64`) and soma-retargeter==0.2.0 depends
    on usd-core==26.3, we can conclude that soma-retargeter==0.2.0 cannot
    be used.
```

`dl` (128.101.125.152, x86_64, 4× RTX 6000 Ada) installs it cleanly.
Newton/Warp also want a real GPU, which `dl` has.

## Pipeline

```
Kimodo-SOMA-RP-v1.1   (text -> SOMA skeleton BVH, spark03)
        │
        ▼
SOMA Retargeter       (SOMA BVH -> robot CSV, dl, Newton/Warp on GPU)
        │  booster_t1 (29 DoF)  <-- the only Booster target bundled
        ▼
soma_csv_to_k1_csv.py (29 DoF T1 -> 22 DoF K1 contract, laptop)
        │
        ▼
motion_feasibility_gate.py  (10 kinematic safety checks)
        │
        ▼
record_k1_motion.py   (tiled 3-panel debug video)
        │
        ▼
BeyondMimic RL tracker (Phase 3 — the physically feasible product)
```

## The schema gap is the whole difficulty

SOMA's bundled `booster_t1` CSV and the Booster K1 contract disagree on five
things. All five are silent-corruption risks, which is why the conversion is a
checked script rather than a `pandas` one-liner.

| | SOMA `booster_t1` CSV | Booster K1 contract |
|---|---|---|
| index column | leading integer `Frame` | none |
| root translation | **centimetres** | metres |
| root rotation | **Euler XYZ, degrees** | **quaternion, xyzw** |
| joint units | **degrees** | radians |
| joints | 29 (7-DoF arms, `Waist`) | 22 (4-DoF arms, no waist) |

### Joints dropped

`Left_Wrist_Pitch`, `Left_Wrist_Yaw`, `Left_Hand_Roll`,
`Right_Wrist_Pitch`, `Right_Wrist_Yaw`, `Right_Hand_Roll`, `Waist`

### Joints renamed

SOMA's T1 model omits the `A` prefix the K1 URDF uses on the shoulders:

| SOMA | K1 contract |
|---|---|
| `Left_Shoulder_Pitch` | `ALeft_Shoulder_Pitch` |
| `Right_Shoulder_Pitch` | `ARight_Shoulder_Pitch` |

(Ankle joints map 1:1 by name and carry straight over; see
`docs/policy_io_reference.md` for the K1 4-bar `CrankUp`/`CrankDown` caveat,
which SOMA's T1 model does not have either.)

## Two things to verify before trusting a converted clip

1. **Floor height.** SOMA's root Z lands wherever its own ground plane was.
   Check it against K1's 0.57 m standing trunk height; use `--root-z-offset`
   if the soles float or sink.
2. **Feet.** The converter does not do foot planting. Enable SOMA's own
   `enable_post_processing` / `enable_contact_processing` with
   `sole_normal_local` set for the K1 foot (the K1 foot box sits at local
   `[0.026, 0, -0.02]`, so sole-up is **-Z**, not the +Z default — get this
   wrong and flattening silently no-ops).

Then run the gate. Expect foot-slip and double-support failures on a first
pass; that is the retargeter telling you the IK weights need the optimiser
(`app/tools/ik-weight-optimizer`), not that the pipeline is broken.

## Commands

```bash
# on dl
cd ~/Projects/soma-retargeter && source .venv/bin/activate
uv run python app/bvh_to_csv_converter.py \
    --config assets/default_bvh_to_csv_converter_config.json

# on the laptop
python scripts/soma_csv_to_k1_csv.py <t1>.csv out_k1.csv
python scripts/motion_feasibility_gate.py out_k1.csv
python scripts/record_k1_motion.py out_k1.csv --out docs/videos/<name>.mp4
```

## Related

- `docs/video_to_motion_plan.md` §5b–5c — the K1 configurator process and the
  foot-contact knobs
- `docs/sdk_ros2_audit.md` — the 22-DoF hardware export path
- route A (text → SMPL-X → GMR) lives on `feat/t2m-smplx-gmr`
