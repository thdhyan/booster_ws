"""Gradio front end for k1m.

    ~/.venvs/k1m/bin/python app.py            # http://127.0.0.1:7860

Upload a video or type a prompt, get back the K1 joint angles (CSV + an
on-screen table) and the 3-panel MuJoCo replay, plus the feasibility gate's
verdict.

The gate verdict is shown prominently and unavoidably, and the joint-angle table
marks any joint that is at or beyond its URDF limit. A generated dance often
looks fine and still saturates elbows at exactly their limits, so "it rendered
nicely" is not evidence that it is trackable.

Heavy stages run on a DGX Spark over ssh, which means a run takes minutes and
the two modes differ in cost:

  * csv / npz   ~10-60 s, no GPU needed
  * video       GVHMR + GMR, minutes
  * text        Kimodo + GMR, minutes
"""
from __future__ import annotations

import os
import re
import sys
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k1m import local_stages, pipeline, remote_stages, schema  # noqa: E402

import gradio as gr  # noqa: E402

OUT_ROOT = os.environ.get("K1M_OUT", "k1m_out")
_lock = threading.Lock()


def _joint_table(csv_path: str, frame: int = 0) -> str:
    """Markdown table of joint angles, flagging ones at their URDF limits."""
    try:
        motion, fps = schema.read_csv(csv_path)
    except Exception as e:                             # noqa: BLE001
        return f"could not read motion: {e}"
    frame = max(0, min(frame, len(motion) - 1))
    try:
        lim = schema.joint_limits_from_urdf()
    except Exception:                                  # noqa: BLE001
        lim = {}
    row = motion[frame]
    out = ["| joint | q (rad) | deg | URDF range | at limit |",
           "|---|---:|---:|---|:--:|"]
    n_at = 0
    for i, name in enumerate(schema.K1_JOINT_NAMES):
        q = float(row[7 + i])
        rng = lim.get(name)
        at = ""
        if rng:
            lo, hi = rng
            slack = min(abs(q - lo), abs(hi - q))
            if slack < 1e-3:
                at = "**YES**"
                n_at += 1
            elif slack < 0.05:
                at = "near"
        rs = f"[{rng[0]:+.2f}, {rng[1]:+.2f}]" if rng else "-"
        out.append(f"| `{name}` | {q:+.4f} | {q * 57.2958:+.1f} | {rs} | {at} |")
    hdr = (f"**frame {frame}/{len(motion)} @ {fps:g} Hz** — "
           f"root ({row[0]:+.3f}, {row[1]:+.3f}, {row[2]:+.3f}) m")
    if n_at:
        hdr += f" — **{n_at} joint(s) pinned at their limit**"
    return hdr + "\n\n" + "\n".join(out)


def _gate_md(csv_path: str) -> str:
    try:
        rep = local_stages.gate(csv_path,
                                report_json=os.path.splitext(csv_path)[0] + "_gate.json")
    except Exception as e:                             # noqa: BLE001
        return f"gate failed to run: {e}"
    return local_stages.summarize_gate(rep)


def _safe_stem(name: str, maxlen: int = 40) -> str:
    """A filesystem-safe output basename.

    Uploaded files are named by whoever recorded them, e.g.
    "Screencast from 2026-09-28 06-30-17", which then becomes a directory name.
    """
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-")
    return (s or "clip")[:maxlen]


def _run(fn, *a, **kw):
    """Run a pipeline call under a lock (the sparks are shared) and report errors.

    Every non-file return must be None on failure, never "": Gradio resolves an
    empty string as a path and then tries to open the current working
    directory, which crashes postprocessing and hides the real error.
    """
    with _lock:
        try:
            r = fn(*a, **kw)
        except Exception as e:                          # noqa: BLE001
            tb = traceback.format_exc(limit=4)
            msg = (f"**FAILED: {type(e).__name__}**\n\n```\n{e}\n```\n\n"
                   f"<details><summary>traceback</summary>\n\n```\n{tb}\n```"
                   f"</details>")
            return None, msg, None, None
    verdict = "**PASS** - physics checks clear" if r.gate_passed else \
        "**FAIL** - not safe for hardware"
    warn = ("\n".join(f"> {w}" for w in r.warnings))
    md = f"{verdict}\n\n```\n{r.gate_summary}\n```"
    if warn:
        md += f"\n\n{warn}"
    if not r.gate_passed and r.gate_json:
        md += ("\n\nThis is a kinematic retarget with no foot locking, no dynamics "
               "and no actuator model. A clean render is not evidence that a "
               "real K1 can track it.")
    video = r.replay_mp4 if r.replay_mp4 and os.path.exists(r.replay_mp4) else None
    table = _joint_table(r.motion_csv) if r.motion_csv else None
    csv = r.motion_csv if r.motion_csv and os.path.exists(r.motion_csv) else None
    return csv, md, video, table


def ui_video(video, fps, seconds, do_gate, do_render):
    if video is None or (isinstance(video, str) and not video.strip()):
        return None, "upload a video first", None, None
    path = getattr(video, "name", video)
    if isinstance(path, dict):                     # gr.Video may hand back a dict
        path = path.get("path") or path.get("video") or path.get("name") or ""
    if not path or not os.path.isfile(path):
        return None, (f"could not read the uploaded video (got {type(video).__name__}: "
                      f"{str(video)[:80]!r})"), None, None
    stem = _safe_stem(path)
    out = os.path.join(OUT_ROOT, stem)
    return _run(pipeline.from_video, path, outdir=out, stem=stem, fps=float(fps),
                do_gate=do_gate, do_render=do_render,
                max_seconds=float(seconds or 0))


def ui_text(prompt, seconds, seed, fps, do_gate, do_render):
    if not prompt or not prompt.strip():
        return None, "type a prompt first", None, None
    stem = "text_" + str(abs(hash(prompt.strip())) % 10**6)
    out = os.path.join(OUT_ROOT, stem)
    return _run(pipeline.from_text, prompt.strip(), outdir=out, stem=stem,
                seconds=float(seconds), seed=int(seed), fps=float(fps),
                do_gate=do_gate, do_render=do_render)


def ui_csv(csv_file, fps, do_gate, do_render):
    if csv_file is None:
        return None, "upload a K1 motion CSV first", None, None
    path = getattr(csv_file, "name", csv_file)
    if isinstance(path, dict):
        path = path.get("path") or path.get("name") or ""
    if not path or not os.path.isfile(path):
        return None, f"could not read the uploaded file: {str(csv_file)[:80]!r}", None, None
    out = os.path.join(OUT_ROOT, _safe_stem(path))
    return _run(pipeline.from_csv, path, outdir=out, do_gate=do_gate,
                do_render=do_render)


def ui_doctor():
    lines = ["### remote stages", "",
             "The laptop is a client; the models run on a DGX Spark over ssh.", "",
             "```"]
    lines += remote_stages.preflight()
    lines += ["```", "", "### joint order", ""]
    try:
        schema.verify_joint_order()
        lines.append("K1 MJCF hinge order matches `K1_JOINT_NAMES` exactly.")
    except Exception as e:                              # noqa: BLE001
        lines.append(f"**mismatch: {e}**")
    return "\n".join(lines)


def build() -> gr.Blocks:
    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    with gr.Blocks(title="K1 motion retargeting") as b:
        gr.Markdown("# Booster K1 motion retargeting\n"
                    "video or text in &rarr; 22-DoF joint angles + MuJoCo replay.\n"
                    "The gate verdict is the safety signal; the render is not.")

        with gr.Tab("Video"):
            with gr.Row():
                vin = gr.Video(label="input video", sources=["upload"])
                with gr.Column():
                    vfps = gr.Slider(10, 60, 30, step=1, label="motion fps")
                    vsec = gr.Slider(0, 60, 0, step=1,
                                     label="max seconds (0 = whole clip)")
                    vgate = gr.Checkbox(True, label="run feasibility gate")
                    vrender = gr.Checkbox(True, label="render replay")
                    vgo = gr.Button("Retarget", variant="primary")
            vcsv = gr.File(label="joint angles CSV")
            vrep = gr.Video(label="MuJoCo replay (3 panels)")
            vverdict = gr.Markdown()
            vtab = gr.Markdown()
            vgo.click(ui_video, [vin, vfps, vsec, vgate, vrender],
                      [vcsv, vverdict, vrep, vtab])
            vin.upload(ui_video, [vin, vfps, vsec, vgate, vrender],
                       [vcsv, vverdict, vrep, vtab])

        with gr.Tab("Text"):
            tin = gr.Textbox(label="prompt",
                             value="A person dances the Macarena, stepping side "
                                   "to side and swinging both arms in wide arcs.",
                             lines=2)
            with gr.Row():
                tsec = gr.Slider(2, 30, 8, step=1, label="seconds")
                tseed = gr.Number(7, label="seed", precision=0)
                tfps = gr.Slider(10, 60, 30, step=1, label="motion fps")
            with gr.Row():
                tgate = gr.Checkbox(True, label="run feasibility gate")
                trender = gr.Checkbox(True, label="render replay")
            tgo = gr.Button("Generate", variant="primary")
            tcsv = gr.File(label="joint angles CSV")
            trep = gr.Video(label="MuJoCo replay (3 panels)")
            tverdict = gr.Markdown()
            ttab = gr.Markdown()
            tgo.click(ui_text, [tin, tsec, tseed, tfps, tgate, trender],
                      [tcsv, tverdict, trep, ttab])

        with gr.Tab("Existing CSV"):
            cin = gr.File(label="K1 motion CSV (7 root + 22 joints)", file_types=[".csv"])
            with gr.Row():
                cfps = gr.Slider(10, 60, 50, step=1, label="motion fps")
                cgate = gr.Checkbox(True, label="run feasibility gate")
                crender = gr.Checkbox(True, label="render replay")
            cgo = gr.Button("Gate + render", variant="primary")
            ccsv = gr.File(label="joint angles CSV")
            crep = gr.Video(label="MuJoCo replay (3 panels)")
            cverdict = gr.Markdown()
            ctab = gr.Markdown()
            cgo.click(ui_csv, [cin, cfps, cgate, crender],
                      [ccsv, cverdict, crep, ctab])

        with gr.Tab("Environment"):
            gr.Markdown("Run `k1m doctor` in a terminal for the same report.")
            dbtn = gr.Button("Check stages")
            dout = gr.Markdown()
            dbtn.click(ui_doctor, None, dout)

    return b


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    build().launch(
        theme=gr.themes.Monochrome(),
        server_name=os.environ.get("K1M_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("K1M_PORT", "7860")),
        share=False, show_error=True,
    )
