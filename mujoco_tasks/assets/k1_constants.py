"""Booster K1 constants (22-DoF, serial ankles)."""

import math
import os
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_REPO_ROOT = Path(__file__).resolve().parents[2]
_K1_REL = Path("robots") / "K1" / "K1_22dof.xml"
_SERIAL_ANKLE_LINK = "left_ankle_roll_link"


def _find_k1_xml() -> Path:
  """Find a K1 MJCF with the serial-ankle link names these tasks are written for.

  Search order: $BOOSTER_ASSETS, <repo>/booster_assets, the old submodule path,
  ~/booster_assets. Copies of K1_22dof.xml without `left_ankle_roll_link` (older
  K1 models with a parallel Ankle_Cross ankle) are skipped.
  """
  bases = []
  if os.environ.get("BOOSTER_ASSETS"):
    bases.append(Path(os.environ["BOOSTER_ASSETS"]))
  bases += [
    _REPO_ROOT / "booster_assets",
    _REPO_ROOT / "src" / "k1_description" / "assets",
    Path.home() / "booster_assets",
  ]
  found = [b / _K1_REL for b in bases if (b / _K1_REL).exists()]
  for xml in found:
    if _SERIAL_ANKLE_LINK in xml.read_text():
      return xml
  if found:
    raise RuntimeError(
      f"Found K1_22dof.xml but none has the link {_SERIAL_ANKLE_LINK}, which these "
      f"tasks need (serial-ankle K1 model): {', '.join(map(str, found))}. "
      "Clone BoosterRobotics/booster_assets and set BOOSTER_ASSETS to it."
    )
  raise FileNotFoundError(
    "K1_22dof.xml not found. Run `git submodule update --init` or set BOOSTER_ASSETS."
  )


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_find_k1_xml()))
  # mjlab supplies the terrain and the actuators, so drop the file's own.
  for act in list(spec.actuators):
    spec.delete(act)
  for geom in list(spec.worldbody.geoms):
    if geom.name == "ground":
      spec.delete(geom)

  # Booster's collision geoms are unnamed; name them so mjlab's collision config can match them.
  for body in spec.bodies:
    for i, g in enumerate(body.geoms):
      if g.contype == 0 and g.conaffinity == 0:
        continue  # visual meshes
      if g.type == mujoco.mjtGeom.mjGEOM_BOX and body.name.endswith("_ankle_roll_link"):
        g.name = body.name.replace("_ankle_roll_link", "_foot_collision")
      else:
        g.name = f"{body.name}_{i}_collision"
  # Foot sites at the sole center, used by the foot clearance / slip rewards and height scan.
  for side in ("left", "right"):
    body = spec.body(f"{side}_ankle_roll_link")
    body.add_site(name=f"{side}_foot", pos=(0.026, 0.0, -0.038), size=(0.005, 0.005, 0.005))
  # Sensors the velocity task reads (the G1/Go1 XMLs define these themselves).
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="trunk",
  )
  return spec


# Gains: stiffness = armature * w^2, damping = 2 * zeta * armature * w (Booster's convention).
def _grp(names, armature, effort, freq=4.0, zeta=1.5):
  w = 2 * math.pi * freq
  return BuiltinPositionActuatorCfg(
    target_names_expr=names,
    stiffness=armature * w**2,
    damping=2 * zeta * armature * w,
    effort_limit=effort,
    armature=armature,
  )


K1_ACTUATORS = (
  _grp((".*_hip_pitch_joint",), 0.0478125, 68.0, freq=8.0),
  _grp((".*_hip_roll_joint",), 0.0339552, 43.0, freq=8.0),
  _grp((".*_hip_yaw_joint",), 0.0282528, 38.3, freq=8.0),
  _grp((".*_knee_pitch_joint",), 0.095625, 112.0, freq=8.0, zeta=1.0),
  _grp((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), 0.0565, 38.3, freq=8.0),
  _grp((".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
        ".*_elbow_pitch_joint", ".*_elbow_yaw_joint"), 0.001, 14.0, freq=10.0, zeta=2.0),
  _grp((".*head.*",), 0.002, 6.0, freq=10.0, zeta=2.0),  # placeholder values
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.55),  # calibrate with the check script
  joint_pos={
    ".*_hip_pitch_joint": -0.2,
    ".*_knee_pitch_joint": 0.4,
    ".*_ankle_pitch_joint": -0.2,
    "left_shoulder_roll_joint": -1.3,
    "right_shoulder_roll_joint": 1.3,
  },
  joint_vel={".*": 0.0},
)

FOOT = r"^(left|right)_foot_collision$"
K1_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={FOOT: 3, ".*_collision": 1},
  priority={FOOT: 1, ".*": 0},
  friction={FOOT: (0.6,)},
)

K1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=K1_ACTUATORS, soft_joint_pos_limit_factor=0.9
)


def get_k1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(K1_COLLISION,),
    spec_fn=get_spec,
    articulation=K1_ARTICULATION,
  )


K1_ACTION_SCALE: dict[str, float] = {}
for a in K1_ACTUATORS:
  for n in a.target_names_expr:
    K1_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness
