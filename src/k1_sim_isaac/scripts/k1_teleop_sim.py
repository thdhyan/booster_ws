#!/usr/bin/env python3
"""K1 VR-teleop sim (Isaac Lab, headless): K1 hanging in the air on a fixed
base, ZED 2i stereo pair on ``Head_2``, chase camera, ROS 2 in/out. Walking
comes later; for now the thumbsticks move the suspended trunk (``/{ns}/cmd_vel``).

Suspension = a gantry: the K1 URDF is extended at start-up with a world-fixed
``gantry_world`` link and four virtual joints ``gantry_x -> gantry_y -> gantry_z
-> gantry_yaw -> Trunk`` (fixed-base articulation, stiff position + velocity
drives). Two simpler routes do not move the robot: PhysX ignores root-pose
writes on a fixed-base articulation, and a FixedJoint to a kinematic body is not
dragged when the body's pose is written (GPU pipeline).

ROS (bundled Jazzy rclpy, ``k1_teleop.isaac_ros_env``):
  in   /{ns}/cmd_vel                  geometry_msgs/Twist: linear.x/y/z, angular.z in the
                                      base heading frame; zero after 0.5 s without one
       /{ns}/teleop/upper_body_cmd    sensor_msgs/JointState: head + arm position targets
                                      (URDF names, e.g. AAHead_yaw); hold-last per joint
       /{ns}/teleop/ik_goals          geometry_msgs/PoseArray (Trunk frame): drawn as spheres
       /{ns}/yolo/image_raw           sensor_msgs/Image: tiled into the debug video
  out  /{ns}/joint_states             sensor_msgs/JointState, 50 Hz
       /{ns}/base_pose                geometry_msgs/PoseStamped (world)
       /{ns}/zed/{left,right}/image_raw   rgb8, same stamp for both eyes
       /{ns}/chase/image_raw          rgb8

Debug video (always on, compulsory): overview / top-down / chase / ZED L / ZED R
/ YOLO tiles with a status header -> ``<out>/videos/k1_teleop_<stamp>.mp4``.
Markers: arena boundary (red), IK goals (green), hand tips (yellow).

    source scripts/phase6_env.sh
    phase6-python src/k1_sim_isaac/scripts/k1_teleop_sim.py --demo --duration 20   # smoke, no ROS input
    phase6-python src/k1_sim_isaac/scripts/k1_teleop_sim.py                         # live, ROS_DOMAIN_ID from env
"""
import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot-ns", default="k1_0")
parser.add_argument("--demo", nargs="?", const="move", choices=("move", "lift"),
                    help="scripted input instead of ROS: move (base/head/arm circles, default) or lift "
                         "(two-arm squeeze + lift of the small box on the counter)")
parser.add_argument("--duration", type=float, default=0.0, help="stop after N s of sim time (0 = run forever)")
parser.add_argument("--no-ros", action="store_true")
parser.add_argument("--base-z", type=float, default=0.9, help="initial trunk height, m")
parser.add_argument("--eye-size", default="960x540", help="per-eye stream size (ZED HD720 is 1280x720)")
parser.add_argument("--chase-size", default="640x360")
parser.add_argument("--cam-hz", type=float, default=30.0, help="camera render + publish rate")
parser.add_argument("--zed-offset", default="0.065,0.085", help="x,z of the ZED lenses in Head_2, m")
parser.add_argument("--baseline", type=float, default=0.120, help="ZED 2i baseline, m")
parser.add_argument("--people-dir", default="", help="local Isaac People mirror (has Characters/); default: asset root")
parser.add_argument("--out", default="logs/k1_teleop_sim", help="run directory (videos/ inside)")
parser.add_argument("--video-sec", type=float, default=60.0, help="length of the debug clip (sim s)")
parser.add_argument("--video-stride", type=int, default=3, help="capture every N control steps")
parser.add_argument("--no-realtime", action="store_true", help="do not pace the loop to wall clock")
args0, _ = parser.parse_known_args()

from k1_teleop import isaac_ros_env  # noqa: E402

USE_ROS = not args0.no_ros and isaac_ros_env.ensure()     # re-execs once, before Kit starts

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

from booster_assets import BOOSTER_ASSETS_DIR  # noqa: E402

PHYS_DT, DECIM = 0.005, 4                     # 200 Hz physics, 50 Hz control
CTRL_DT = PHYS_DT * DECIM
ARENA = 4.0                                   # m, half size of the square the base may move in
Z_RANGE = (0.6, 1.5)                          # trunk height; feet are ~0.55 m below the trunk
EYE_HFOV = 101.0                              # ZED 2i 2.1 mm lens, rectified HD720
HAND_TIP = 0.20                               # m along the forearm, as k1_teleop.k1_arm_ik
TILE = (480, 270)
# name, Isaac People character, (x, y), heading deg: in front of the K1 (it starts facing +x)
# Graspable props. K1 has no hands: "lifting" = squeezing between the forearms, so the small
# boxes sit on a counter at hand height right in front of the start pose (the arms reach
# 0.37 m from the shoulder, ~1.07 m up with the trunk at 0.9 m).
GRIP = sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, friction_combine_mode="max")
COUNTER = ((0.35, 0.0), (0.3, 0.6, 0.93))                     # centre xy, size xyz (top at 0.93 m)
# name, centre xyz, size xyz, mass kg, colour
PROPS = [("small_box", (0.28, 0.0, 0.99), (0.12, 0.12, 0.12), 0.2, (0.95, 0.45, 0.1)),
         ("small_box_2", (0.40, 0.18, 0.98), (0.10, 0.16, 0.10), 0.3, (0.3, 0.8, 0.4))]
# the big boxes: dynamic too (they were static colliders = immovable), pushable, too big to lift
BIG_BOXES = [((1.8, -0.6), (0.5, 0.5, 0.5), 3.0, (0.8, 0.2, 0.2)),
             ((2.6, 0.3), (0.8, 0.5, 0.75), 6.0, (0.2, 0.4, 0.85)),
             ((1.4, 1.2), (0.3, 0.3, 1.0), 2.0, (0.9, 0.7, 0.1))]
PEOPLE = [("M_Medical_01", (3.0, 0.8), 180.0), ("F_Business_02", (3.6, -1.3), 150.0),
          ("male_adult_police_04", (2.2, 2.2), -120.0), ("male_adult_construction_05_new", (-2.5, 0.6), 0.0)]

ROBOT_CFG = ArticulationCfg(
    prim_path="/World/K1",
    spawn=sim_utils.UrdfFileCfg(
        asset_path="",                         # gantry_urdf() fills it in
        fix_base=True,
        replace_cylinders_with_capsules=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={"Left_Shoulder_Roll": -1.3, "Right_Shoulder_Roll": 1.3,   # arms down
                   "gantry_z": args.base_z},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(joint_names_expr=[".*Hip.*", ".*Knee.*", ".*Ankle.*"], stiffness=60.0, damping=3.0),
        "arms": ImplicitActuatorCfg(joint_names_expr=[".*Shoulder.*", ".*Elbow.*"], stiffness=40.0, damping=2.0),
        "head": ImplicitActuatorCfg(joint_names_expr=[".*Head.*"], stiffness=20.0, damping=1.0),
        # ~20 kg carried: wn 32 rad/s, over-damped; yaw carries ~0.3 kg m^2. 1e6/1e4 on
        # 0.05 kg virtual links oscillated (yaw effort 3.7e5 Nm at 0.4 deg error)
        "gantry_xyz": ImplicitActuatorCfg(joint_names_expr=["gantry_[xyz]"], stiffness=2e4, damping=2e3,
                                          effort_limit_sim=1e5),
        "gantry_yaw": ImplicitActuatorCfg(joint_names_expr=["gantry_yaw"], stiffness=1e3, damping=100.0,
                                          effort_limit_sim=1e4),
    },
)


GANTRY = ("gantry_x", "gantry_y", "gantry_z", "gantry_yaw")


def gantry_urdf(out_dir: Path) -> str:
    """K1_22dof-ZED.urdf + world gantry (see module doc), meshes made absolute."""
    src = Path(BOOSTER_ASSETS_DIR) / "robots/K1/K1_22dof-ZED.urdf"
    text = src.read_text().replace('filename="meshes/', f'filename="{src.parent}/meshes/')
    tiny = ('<inertial><origin xyz="0 0 0"/><mass value="1.0"/>'
            '<inertia ixx="1e-2" ixy="0" ixz="0" iyy="1e-2" iyz="0" izz="1e-2"/></inertial>')
    links = "".join(f'<link name="{n}">{tiny}</link>' for n in ("gantry_world", "gantry_lx", "gantry_ly", "gantry_lz"))
    chain = [("gantry_x", "prismatic", "gantry_world", "gantry_lx", "1 0 0", -ARENA, ARENA),
             ("gantry_y", "prismatic", "gantry_lx", "gantry_ly", "0 1 0", -ARENA, ARENA),
             ("gantry_z", "prismatic", "gantry_ly", "gantry_lz", "0 0 1", Z_RANGE[0], Z_RANGE[1]),
             ("gantry_yaw", "revolute", "gantry_lz", "Trunk", "0 0 1", -1000.0, 1000.0)]
    joints = "".join(f'<joint name="{n}" type="{t}"><parent link="{a}"/><child link="{b}"/>'
                     f'<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="{ax}"/>'
                     f'<limit lower="{lo}" upper="{hi}" effort="1e7" velocity="100"/></joint>'
                     for n, t, a, b, ax, lo, hi in chain)
    i = text.index("<link")
    out = out_dir / "K1_22dof-ZED_gantry.urdf"
    out.write_text(text[:i] + links + joints + text[i:])
    return str(out)


def size(s: str) -> tuple[int, int]:
    w, h = (int(v) for v in s.split("x"))
    return w, h


def camera(path: str, wh: tuple[int, int], hfov_deg: float, offset=None) -> Camera:
    w, h = wh
    ap = 20.955
    cfg = CameraCfg(
        prim_path=path, update_period=0.0, width=w, height=h, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=ap / (2 * math.tan(math.radians(hfov_deg) / 2)),
                                         horizontal_aperture=ap, clipping_range=(0.05, 60.0)),
        offset=offset or CameraCfg.OffsetCfg(),
    )
    return Camera(cfg)


def build_scene() -> dict[str, RigidObject]:
    """Static world + people; returns the graspable props (tracked, for the header/log)."""
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=900.0).func("/World/dome", sim_utils.DomeLightCfg(intensity=900.0))
    light = sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0)
    light.func("/World/sun", light, orientation=(0.3, 0.2, 0.0, 0.93))
    for i, ((x, y), size_, mass, c) in enumerate(BIG_BOXES):
        box = sim_utils.CuboidCfg(size=size_, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c),
                                  collision_props=sim_utils.CollisionPropertiesCfg(),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=mass), physics_material=GRIP)
        box.func(f"/World/box_{i}", box, translation=(x, y, size_[2] / 2))
    (cx, cy), (sx, sy, sz) = COUNTER
    counter = sim_utils.CuboidCfg(size=(sx, sy, sz), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
                                  physics_material=GRIP)
    counter.func("/World/counter", counter, translation=(cx, cy, sz / 2))
    props = {}
    for name, pos, size_, mass, c in PROPS:
        props[name] = RigidObject(RigidObjectCfg(
            prim_path=f"/World/{name}",
            spawn=sim_utils.CuboidCfg(size=size_, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c),
                                      collision_props=sim_utils.CollisionPropertiesCfg(),
                                      rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                      mass_props=sim_utils.MassPropertiesCfg(mass=mass), physics_material=GRIP),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos)))
    red = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1), opacity=0.35)
    for i, (x, y, sx, sy) in enumerate([(ARENA, 0, 0.03, 2 * ARENA), (-ARENA, 0, 0.03, 2 * ARENA),
                                        (0, ARENA, 2 * ARENA, 0.03), (0, -ARENA, 2 * ARENA, 0.03)]):
        wall = sim_utils.CuboidCfg(size=(sx, sy, 0.3), visual_material=red)
        wall.func(f"/Visuals/arena_{i}", wall, translation=(x, y, 0.15))
    people = Path(args.people_dir) if args.people_dir else None
    for i, (char, (x, y), yaw) in enumerate(PEOPLE):
        local = people / "Characters" / char / f"{char}.usd" if people else None
        usd = str(local) if local and local.exists() else f"{ISAAC_NUCLEUS_DIR}/People/Characters/{char}/{char}.usd"
        h = math.radians(yaw) / 2
        cfg = sim_utils.UsdFileCfg(usd_path=usd)
        try:
            cfg.func(f"/World/person_{i}", cfg, translation=(x, y, 0.0), orientation=(0.0, 0.0, math.sin(h), math.cos(h)))
        except Exception as e:   # no asset root offline: boxes still give the cameras structure
            print(f"[k1_teleop_sim] person {char} skipped: {e}")
    return props


def find_prim(name: str, under: str) -> str:
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    hits = [(p.HasAPI(UsdPhysics.RigidBodyAPI), str(p.GetPath())) for p in stage.Traverse()
            if p.GetName() == name and str(p.GetPath()).startswith(under)]
    assert hits, f"{name} not found under {under}"
    return sorted(hits, reverse=True)[0][1]


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))            # xyzw


class Demo:
    """Scripted operator for smoke tests: base moves, head looks around, hands trace circles (IK)."""

    # lift: hand-tip waypoints in the trunk frame (trunk at 0.89 m), per side y = sign * value.
    # Up in front of the counter edge, forward beside small_box, squeeze 3 cm into it, lift 15 cm.
    LIFT = [(0.0, None), (2.5, (0.12, 0.16, 0.15)), (4.5, (0.28, 0.16, 0.10)),
            (6.5, (0.28, 0.03, 0.10)), (9.0, (0.28, 0.03, 0.25)), (99.0, (0.28, 0.03, 0.25))]

    def __init__(self, mode: str = "move") -> None:
        from k1_teleop.k1_arm_ik import ArmIK

        self.ik, self.mode = ArmIK(), mode
        self.q = {"Left_Shoulder_Roll": -1.3, "Right_Shoulder_Roll": 1.3}
        self.start = {s: self.ik.fk(s, self.q)[:3, 3] for s in ("left", "right")}
        self.goals = {}

    def cmd(self, t: float) -> tuple[float, float, float, float]:
        if self.mode == "lift":
            return (0.0, 0.0, 0.0, 0.0)
        if t < 4:
            return (0.25, 0.0, 0.0, 0.0)
        if t < 7:
            return (0.0, 0.0, 0.0, 0.5)
        if t < 9:
            return (0.0, 0.0, 0.12, 0.0)
        if t < 11:
            return (0.0, 0.0, -0.12, 0.0)
        return (0.0, 0.2 * math.sin(0.8 * (t - 11)), 0.0, 0.0)

    def upper(self, t: float) -> dict[str, float]:
        if self.mode == "lift":
            return self._lift(t)
        out = {"AAHead_yaw": 0.6 * math.sin(0.8 * t), "Head_pitch": 0.25 + 0.25 * math.sin(0.5 * t)}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            sh = self.ik.shoulder_pos(side, {})
            a = 1.5 * t + (0.0 if sign > 0 else math.pi)
            g = np.eye(4)
            g[:3, 3] = sh + np.array([0.22, sign * 0.03 + 0.06 * math.cos(a), -0.05 + 0.08 * math.sin(a)])
            q, _ = self.ik.solve(side, g, self.q)
            self.q.update(zip(self.ik.names[side], q))
            self.goals[side] = g[:3, 3]
        out.update(self.q)
        return out


    def _lift(self, t: float) -> dict[str, float]:
        out = {"AAHead_yaw": 0.0, "Head_pitch": 0.6}                      # look down at the box
        k = next(i for i in range(1, len(self.LIFT)) if t < self.LIFT[i][0])
        (t0, a), (t1, b) = self.LIFT[k - 1], self.LIFT[k]
        u = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        for side, sign in (("left", 1.0), ("right", -1.0)):
            pa = self.start[side] if a is None else np.array([a[0], sign * a[1], a[2]])
            pb = np.array([b[0], sign * b[1], b[2]])
            g = np.eye(4)
            g[:3, 3] = pa + u * (pb - pa)
            q, _ = self.ik.solve(side, g, self.q)
            self.q.update(zip(self.ik.names[side], q))
            self.goals[side] = g[:3, 3]
        out.update(self.q)
        return out


class RosIO:
    def __init__(self, ns: str) -> None:
        import rclpy
        from geometry_msgs.msg import PoseArray, PoseStamped, Twist
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import Image, JointState

        rclpy.init()
        self.rclpy, self.Image, self.JointState, self.PoseStamped = rclpy, Image, JointState, PoseStamped
        self.node = rclpy.create_node("k1_teleop_sim")
        big = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE)   # images >> rmem_max
        self.pub_js = self.node.create_publisher(JointState, f"/{ns}/joint_states", 10)
        self.pub_base = self.node.create_publisher(PoseStamped, f"/{ns}/base_pose", 10)
        self.pub_img = {k: self.node.create_publisher(Image, f"/{ns}/{k}/image_raw", big)
                        for k in ("zed/left", "zed/right", "chase")}
        self.cmd, self.cmd_t = (0.0, 0.0, 0.0, 0.0), -1.0
        self.upper: dict[str, float] = {}
        self.goals: dict[str, np.ndarray] = {}
        self.goals_t = -1.0
        self.yolo, self.yolo_t = None, -1.0
        self.node.create_subscription(Twist, f"/{ns}/cmd_vel", self._on_cmd, 10)
        self.node.create_subscription(JointState, f"/{ns}/teleop/upper_body_cmd", self._on_upper, 10)
        self.node.create_subscription(PoseArray, f"/{ns}/teleop/ik_goals", self._on_goals, 10)
        self.node.create_subscription(Image, f"/{ns}/yolo/image_raw", self._on_yolo, big)

    def _on_cmd(self, m) -> None:
        self.cmd, self.cmd_t = (m.linear.x, m.linear.y, m.linear.z, m.angular.z), time.monotonic()

    def _on_upper(self, m) -> None:
        self.upper.update(zip(m.name, (float(p) for p in m.position)))

    def _on_goals(self, m) -> None:
        self.goals = {s: np.array([p.position.x, p.position.y, p.position.z])
                      for s, p in zip(("left", "right"), m.poses)}
        self.goals_t = time.monotonic()

    def _on_yolo(self, m) -> None:
        if m.encoding == "rgb8":
            self.yolo, self.yolo_t = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3), time.monotonic()

    def spin(self) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def current_cmd(self):
        return self.cmd if time.monotonic() - self.cmd_t < 0.5 else (0.0, 0.0, 0.0, 0.0)

    def image(self, key: str, rgb: np.ndarray, stamp) -> None:
        m = self.Image(height=rgb.shape[0], width=rgb.shape[1], encoding="rgb8", step=rgb.shape[1] * 3)
        m.header.stamp, m.header.frame_id = stamp, key.replace("/", "_")
        m.data = rgb.tobytes()
        self.pub_img[key].publish(m)


def prop_text(props: dict[str, RigidObject]) -> str:
    return " ".join(f"{n} z {float(o.data.root_link_pose_w.torch[0, 2]):.2f}" for n, o in props.items())


def tile(img: np.ndarray | None, label: str) -> np.ndarray:
    import cv2

    out = np.zeros((TILE[1], TILE[0], 3), np.uint8)
    if img is not None:
        out = cv2.resize(np.ascontiguousarray(img[..., :3]), TILE, interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (len(label) * 11 + 12, 24), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    import cv2
    import imageio.v2 as imageio

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYS_DT, render_interval=DECIM))
    props = build_scene()
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    ROBOT_CFG.spawn.asset_path = gantry_urdf(out)
    robot = Articulation(ROBOT_CFG)
    head = find_prim("Head_2", "/World/K1")
    zx, zz = (float(v) for v in args.zed_offset.split(","))
    eye_wh, chase_wh = size(args.eye_size), size(args.chase_size)
    cams = {
        "zed/left": camera(f"{head}/zed_left", eye_wh, EYE_HFOV,
                           CameraCfg.OffsetCfg(pos=(zx, args.baseline / 2, zz), convention="world")),
        "zed/right": camera(f"{head}/zed_right", eye_wh, EYE_HFOV,
                            CameraCfg.OffsetCfg(pos=(zx, -args.baseline / 2, zz), convention="world")),
        "chase": camera("/Visuals/chase_cam", chase_wh, 70.0),
        "overview": camera("/Visuals/overview_cam", TILE, 60.0),
        "top": camera("/Visuals/top_cam", TILE, 60.0),
    }
    goal_mk = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/ik_goals", markers={
        "goal": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 1.0, 0.2), opacity=0.6))}))
    tip_mk = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/hand_tips", markers={
        "tip": sim_utils.SphereCfg(radius=0.02, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.85, 0.1), opacity=0.8))}))
    sim.reset()
    dev = robot.device

    jidx = {n: i for i, n in enumerate(robot.joint_names)}
    names = [n for n in robot.joint_names if n not in GANTRY]          # the K1's own 22 joints
    k1_ids = [jidx[n] for n in names]
    g_ids = [jidx[n] for n in GANTRY]
    default_q = robot.data.default_joint_pos[0].clone()
    target = default_q.clone()
    hand_ids = [robot.body_names.index(f"{s}_hand_link") for s in ("left", "right")]
    print(f"[k1_teleop_sim] {len(names)} joints, head prim {head}, ROS {'on' if USE_ROS else 'off'}")

    ros = RosIO(args.robot_ns) if USE_ROS else None
    demo = Demo(args.demo) if args.demo else None

    clip = out / "videos" / f"k1_teleop_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = imageio.get_writer(clip, fps=1.0 / (CTRL_DT * args.video_stride), macro_block_size=1)
    frames_written = 0

    base = np.array([0.0, 0.0, args.base_z])
    yaw = 0.0
    cam_every = max(1, round(1.0 / (args.cam_hz * CTRL_DT)))
    step, t = 0, 0.0
    last_rgb: dict[str, np.ndarray] = {}
    wall0 = wall_log = time.perf_counter()
    while simulation_app.is_running():
        if args.duration and t >= args.duration:
            break
        if ros:
            ros.spin()
        # ── inputs ──
        if demo:
            cmd, upper, goals = demo.cmd(t), demo.upper(t), demo.goals
        else:
            cmd = ros.current_cmd() if ros else (0.0, 0.0, 0.0, 0.0)
            upper = ros.upper if ros else {}
            goals = ros.goals if ros and time.monotonic() - ros.goals_t < 0.5 else {}
        vx, vy, vz, wz = cmd
        yaw += wz * CTRL_DT                           # unwrapped: gantry_yaw is a revolute joint
        c, s = math.cos(yaw), math.sin(yaw)
        base += CTRL_DT * np.array([c * vx - s * vy, s * vx + c * vy, vz])
        base[:2] = np.clip(base[:2], -ARENA + 0.3, ARENA - 0.3)
        base[2] = min(max(base[2], Z_RANGE[0]), Z_RANGE[1])
        target.copy_(default_q)
        for n, v in upper.items():
            if n in jidx and n not in GANTRY:
                target[jidx[n]] = v
        target[g_ids] = torch.tensor([*base, yaw], dtype=target.dtype, device=dev)
        vel = torch.zeros_like(target)
        vel[g_ids] = torch.tensor([c * vx - s * vy, s * vx + c * vy, vz, wz], dtype=target.dtype, device=dev)
        robot.set_joint_position_target_index(target=target[None])
        robot.set_joint_velocity_target_index(target=vel[None])
        robot.write_data_to_sim()
        for _ in range(DECIM):
            sim.step(render=False)
        robot.update(CTRL_DT)
        for prop in props.values():
            prop.update(CTRL_DT)
        step += 1
        t = step * CTRL_DT

        # ── markers (world frame) ──
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        gw = [base + R @ g for g in goals.values()] or [np.array([0.0, 0.0, -10.0])]
        goal_mk.visualize(translations=torch.tensor(np.array(gw), dtype=torch.float32, device=dev))
        body = robot.data.body_link_pose_w[0].cpu().numpy()                 # (B, 7) pos + xyzw
        tips = []
        for i, sign in zip(hand_ids, (1.0, -1.0)):
            x, y, z, w = body[i, 3:7]
            Rb = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                           [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            tips.append(body[i, :3] + Rb @ np.array([0.0, sign * HAND_TIP, 0.0]))
        tip_mk.visualize(translations=torch.tensor(np.array(tips), dtype=torch.float32, device=dev))

        if ros:
            stamp = ros.node.get_clock().now().to_msg()
            q = robot.data.joint_pos[0].cpu().numpy()[k1_ids]
            js = ros.JointState(name=names, position=[float(v) for v in q])
            js.header.stamp = stamp
            ros.pub_js.publish(js)
            bp = ros.PoseStamped()
            bp.header.stamp, bp.header.frame_id = stamp, "world"
            bp.pose.position.x, bp.pose.position.y, bp.pose.position.z = (float(v) for v in base)
            (bp.pose.orientation.x, bp.pose.orientation.y, bp.pose.orientation.z,
             bp.pose.orientation.w) = yaw_quat(yaw)
            ros.pub_base.publish(bp)

        # ── cameras: render only when a stream frame or a video frame is due ──
        recording = t <= args.video_sec
        want_stream = step % cam_every == 0
        want_video = recording and step % args.video_stride == 0
        if want_stream or want_video:
            trunk = base + np.array([0.0, 0.0, 0.1])
            eye = trunk + R @ np.array([-1.4, 0.0, 0.5])
            cams["chase"].set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32, device=dev),
                                                    torch.tensor([trunk], dtype=torch.float32, device=dev))
            if want_video:
                cams["overview"].set_world_poses_from_view(
                    torch.tensor([[base[0] + 2.2, base[1] - 2.2, 1.8]], dtype=torch.float32, device=dev),
                    torch.tensor([trunk], dtype=torch.float32, device=dev))
                cams["top"].set_world_poses_from_view(
                    torch.tensor([[base[0] - 0.01, base[1], 5.0]], dtype=torch.float32, device=dev),
                    torch.tensor([[base[0], base[1], 0.0]], dtype=torch.float32, device=dev))
            sim.render()
            for k, cam in cams.items():
                if k in ("overview", "top") and not want_video:
                    continue
                cam.update(CTRL_DT, force_recompute=True)
                last_rgb[k] = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
            if ros and want_stream:
                for k in ("zed/left", "zed/right", "chase"):
                    ros.image(k, last_rgb[k], stamp)
            if want_video:
                yolo = ros.yolo if ros and time.monotonic() - ros.yolo_t < 2.0 else None
                row1 = np.hstack([tile(last_rgb["overview"], "overview"), tile(last_rgb["top"], "top down"),
                                  tile(last_rgb["chase"], "chase (follow)")])
                row2 = np.hstack([tile(last_rgb["zed/left"], "ZED left"), tile(last_rgb["zed/right"], "ZED right"),
                                  tile(yolo, "YOLO (ZED left)" if yolo is not None else "YOLO: no /yolo/image_raw")])
                hdr = np.zeros((56, row1.shape[1], 3), np.uint8)
                src = "demo" if demo else ("ROS" if ros else "idle")
                hy = upper.get("AAHead_yaw", 0.0)
                hp = upper.get("Head_pitch", 0.0)
                cv2.putText(hdr, f"K1 teleop sim  t {t:6.2f}s  step {step}  input {src}  fixed base (suspended)",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(hdr, f"base x {base[0]:+.2f} y {base[1]:+.2f} z {base[2]:.2f} yaw {math.degrees(yaw):+5.0f}  "
                                 f"cmd vx {vx:+.2f} vy {vy:+.2f} vz {vz:+.2f} wz {wz:+.2f}  head y {hy:+.2f} p {hp:+.2f}  "
                                 f"goals {len(goals)} (green) tips (yellow) arena (red)  {prop_text(props)}",
                            (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 1, cv2.LINE_AA)
                writer.append_data(np.vstack([hdr, row1, row2]))
                frames_written += 1
        if frames_written and not recording and writer is not None:
            writer.close()
            writer = None
            print(f"[k1_teleop_sim] debug video: {clip} ({frames_written} frames)")
        if not args.no_realtime:
            lag = step * CTRL_DT - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)
        if step % 250 == 0:
            now = time.perf_counter()
            rtf = t / (now - wall0)
            rtf_5s = 250 * CTRL_DT / (now - wall_log) if step > 250 else rtf      # last 5 s of sim
            wall_log = now
            trunk = robot.data.body_link_pose_w[0, robot.body_names.index("Trunk")].cpu().numpy()
            print(f"[k1_teleop_sim] t {t:.1f}s RTF {rtf:.2f} (last 5 s {rtf_5s:.2f}, "
                  f"{'recording' if recording else 'streaming only'}) {prop_text(props)}  cmd base {base.round(2)} yaw {math.degrees(yaw):.0f}  "
                  f"sim trunk {trunk[:3].round(2)} yaw {math.degrees(2 * math.atan2(trunk[5], trunk[6])):.0f}")
    if writer is not None:
        writer.close()
        print(f"[k1_teleop_sim] debug video: {clip} ({frames_written} frames)")
    if ros:
        ros.node.destroy_node()
        ros.rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
