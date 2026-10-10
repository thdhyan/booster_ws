"""Small tiled debug-video helper for K1 play/eval runs.

Provides three/four fixed USD cameras (overview, top-down, follow, side),
labelled panels, a status header, and transparent task markers.  It follows the
same cheap pattern as the PE debug recorder: render only while a clip is being
captured, and pump Kit with physics disabled.
"""
from __future__ import annotations

import os

import numpy as np


def pump_app() -> None:
    """Update Kit once without advancing physics."""
    import carb
    import omni.kit.app

    settings = carb.settings.get_settings()
    previous = settings.get("/app/player/playSimulations")
    settings.set_bool("/app/player/playSimulations", False)
    omni.kit.app.get_app().update()
    settings.set_bool("/app/player/playSimulations", bool(previous))


class PanelCameras:
    """Replicator RGB cameras posed from world-space eye/target pairs."""

    def __init__(self, names=("overview", "top_down", "follow", "side"), size=(480, 270), focal_length=16.0):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        self._Gf = Gf
        self._rep = rep
        self._stage = omni.usd.get_context().get_stage()
        self._ops = {}
        self._annotators = {}
        self._render_products = []
        self.names = tuple(names)
        for index, name in enumerate(self.names):
            path = f"/Visuals/K1DebugCams/{name}"
            camera = UsdGeom.Camera.Define(self._stage, path)
            camera.CreateFocalLengthAttr(focal_length)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
            xform = UsdGeom.Xformable(camera)
            xform.ClearXformOpOrder()
            self._ops[name] = xform.AddTransformOp()
            product = rep.create.render_product(path, size)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annotator.attach([product])
            self._render_products.append(product)
            self._annotators[name] = annotator

    def set_views(self, views) -> None:
        """Pose cameras from ``[(name, eye, target), ...]``."""
        Gf = self._Gf
        for name, eye, target in views:
            if name not in self._ops:
                continue
            matrix = Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*[float(v) for v in eye]),
                Gf.Vec3d(*[float(v) for v in target]),
                Gf.Vec3d(0.0, 0.0, 1.0),
            )
            self._ops[name].Set(matrix.GetInverse())

    def read(self, name):
        raw = self._annotators[name].get_data()
        if isinstance(raw, dict):
            raw = raw.get("data")
        if raw is None:
            return None
        raw = np.asarray(raw, dtype=np.uint8)
        return raw if raw.ndim == 3 else None

    def close(self) -> None:
        for product in self._render_products:
            try:
                self._rep.remove_render_product(product)
            except Exception:
                pass


class TiledPanelRecorder:
    """Write labelled panels and a status header to one MP4."""

    # quality=5 maps to roughly crf 25 in imageio's ffmpeg writer. The previous 8
    # (crf ~10) produced a 55 Mbit/s stream -- RTX path-tracing noise is close to
    # incompressible -- and writing that onto the bind-mounted logs filesystem
    # corrupted the mdat: the moov index still advertised 700 frames while only 39
    # decoded, which ffmpeg reports as "Invalid NAL unit size".
    QUALITY = 5

    def __init__(self, path, names, size=(480, 270), cols=2, fps=25):
        import imageio.v2 as imageio
        from PIL import ImageFont

        self.path = path
        self.names = tuple(names)
        self.size = tuple(size)
        self.cols = cols
        self.frames_added = 0
        # Encode to a container-local path, then move into place on close(). The logs
        # dir is a bind mount from the host; streaming tens of Mbit/s at it was the
        # suspected source of the bitstream damage.
        self.tmp_path = f"/tmp/_panel_{os.getpid()}.mp4"
        self.writer = imageio.get_writer(
            self.tmp_path, fps=fps, codec="libx264", quality=self.QUALITY, macro_block_size=1
        )
        try:
            self.font = ImageFont.load_default(size=16)
        except TypeError:
            self.font = ImageFont.load_default()

    def add(self, images, status: str) -> None:
        from PIL import Image, ImageDraw

        cells = []
        for name in self.names:
            image = images.get(name)
            if image is None:
                image = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
            else:
                image = np.asarray(image[:, :, :3], dtype=np.uint8)
            image = Image.fromarray(image).resize(self.size)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 12 * len(name) + 16, 24), fill=(0, 0, 0))
            draw.text((6, 4), name, fill=(255, 255, 255), font=self.font)
            cells.append(np.asarray(image))

        while len(cells) % self.cols:
            cells.append(np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8))
        rows = [np.concatenate(cells[i:i + self.cols], axis=1) for i in range(0, len(cells), self.cols)]
        grid = np.concatenate(rows, axis=0)
        header = Image.new("RGB", (grid.shape[1], 34), (18, 18, 18))
        ImageDraw.Draw(header).text((8, 8), status[:220], fill=(240, 240, 240), font=self.font)
        self.writer.append_data(np.concatenate([np.asarray(header), grid], axis=0))
        self.frames_added += 1

    def close(self) -> None:
        """Close, verify, publish, verify again.

        The pre-move check is a bisect, not ceremony. Two recordings came out corrupt
        ("Invalid NAL unit size (117004 > 116125)", 505 decode errors) with the moov
        atom intact and the correct duration, so file existence and ffprobe duration are
        both worthless as evidence. Checking at the container-local tmp path and again
        after the move to the bind mount isolates which stage does the damage:
        the encoder, or the copy onto /workspace/mounts.
        """
        import shutil

        self.writer.close()

        pre = self._count_decodable(self.tmp_path)
        print(f"[PANEL_VERIFY] pre-move  frames_added={self.frames_added} "
              f"decodable={pre} bytes={os.path.getsize(self.tmp_path)}")

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        shutil.move(self.tmp_path, self.path)
        post = self._count_decodable(self.path)

        # -1 means the check itself could not run (no ffmpeg, timeout, probe failure).
        # That is NOT the same as zero decodable frames and must not be reported as
        # corruption, which is how the first fixed run printed CORRUPT for a file that
        # had merely gone unchecked.
        if pre < 0 or post < 0:
            verdict = "UNCHECKED"
        elif post == self.frames_added:
            verdict = "OK"
        else:
            verdict = "CORRUPT"
        blame = ""
        if verdict == "CORRUPT" and pre == self.frames_added:
            blame = " (damaged by the move onto the bind mount)"
        elif verdict == "CORRUPT":
            blame = " (already corrupt in the encoder's own output)"
        print(
            f"[PANEL_VERIFY] post-move frames_added={self.frames_added} "
            f"decodable={post} {verdict}{blame} bytes={os.path.getsize(self.path)} {self.path}"
        )

    def _count_decodable(self, path) -> int:
        """Count frames a decoder can actually produce. -1 = could not check."""
        import shutil as _shutil
        import subprocess

        ffmpeg = _shutil.which("ffmpeg")
        ffprobe = _shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            # imageio drives ffmpeg through the imageio-ffmpeg wheel, which ships its
            # own binary; the system one is often absent inside the container.
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:  # noqa: BLE001
                print(f"[PANEL_VERIFY] no ffmpeg available: {type(exc).__name__}: {exc}")
                return -1
            ffprobe = ffmpeg  # ffprobe is not shipped; fall back to ffmpeg's own count

        try:
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-i", path, "-f", "null", "-"],
                capture_output=True, text=True, timeout=900,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[PANEL_VERIFY] decode check failed: {type(exc).__name__}: {exc}")
            return -1

        if proc.stderr.strip():
            errs = [l for l in proc.stderr.splitlines() if l.strip()]
            print(f"[PANEL_VERIFY] ffmpeg emitted {len(errs)} error line(s), first: {errs[0][:90]}")
            return 0

        try:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=900,
            )
            return int(probe.stdout.strip().splitlines()[0])
        except Exception as exc:  # noqa: BLE001
            print(f"[PANEL_VERIFY] frame count failed: {type(exc).__name__}: {exc}")
            return -1


class K1DebugMarkers:
    """Transparent robot/target/boundary markers for the debug panels."""

    def __init__(self):
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        self._Gf = Gf
        self._Vt = Vt
        self._stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(self._stage, "/Visuals/K1DebugMarkers")
        self.robot = self._sphere("robot", Gf.Vec3f(0.1, 0.8, 1.0))
        self.target = self._sphere("target", Gf.Vec3f(1.0, 0.8, 0.1))
        self.estimate = self._sphere("estimate", Gf.Vec3f(1.0, 0.1, 0.8))
        # A solid translucent cube would sit between the cameras and the robot,
        # so the arena boundary is drawn as a flat square outline instead.
        boundary = UsdGeom.BasisCurves.Define(self._stage, "/Visuals/K1DebugMarkers/boundary")
        half = 5.0
        points = [Gf.Vec3f(x, y, 0.0) for x, y in
                  ((-half, -half), (half, -half), (half, half), (-half, half))]
        boundary.CreatePointsAttr(Vt.Vec3fArray(points))
        boundary.CreateWidthsAttr([0.03])
        boundary.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.5, 1.0)])
        boundary.CreateDisplayOpacityAttr([0.7])
        self.boundary = boundary
        self._xforms = {
            "robot": UsdGeom.Xformable(self.robot),
            "target": UsdGeom.Xformable(self.target),
            "estimate": UsdGeom.Xformable(self.estimate),
            "boundary": UsdGeom.Xformable(boundary),
        }
        del root

    def _sphere(self, name, color):
        from pxr import Gf, UsdGeom

        sphere = UsdGeom.Sphere.Define(self._stage, f"/Visuals/K1DebugMarkers/{name}")
        sphere.CreateRadiusAttr(0.11)
        sphere.CreateDisplayColorAttr([color])
        sphere.CreateDisplayOpacityAttr([0.45])
        return sphere

    def _translate(self, name, position) -> None:
        xform = self._xforms[name]
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(self._Gf.Vec3d(*[float(v) for v in position]))

    def update(self, base_env, robot, task: str) -> dict:
        """Update markers and return status/knowledge fields for the header."""
        scene = base_env.scene
        root = robot.data.root_pos_w[0].detach().cpu().numpy()
        origin = scene.env_origins[0].detach().cpu().numpy()
        self._translate("robot", root + np.array([0.0, 0.0, 0.18]))
        self._translate("boundary", origin + np.array([0.0, 0.0, -0.03]))

        target = root + np.array([0.8, 0.0, 0.15])
        knowledge = "command target"
        ball = None
        try:
            ball = scene["ball"]
        except Exception:
            pass
        if ball is not None:
            target = ball.data.root_pos_w[0].detach().cpu().numpy()
            knowledge = "ball GT (reward only)"
        else:
            try:
                cmd = base_env.command_manager.get_command("base_velocity")[0].detach().cpu().numpy()
                norm = float(np.linalg.norm(cmd[:2]))
                if norm > 0.05:
                    target = root + np.array([cmd[0], cmd[1], 0.0]) / norm * min(1.5, max(0.5, norm))
                    knowledge = f"command ({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})"
            except Exception:
                pass
        self._translate("target", target)
        self._translate("estimate", target + np.array([0.0, 0.0, 0.12]))
        return {"target": target, "knowledge": knowledge, "task": task}
