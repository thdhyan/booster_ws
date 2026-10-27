"""On-robot audio: arecord -> Segmenter -> faster-whisper, Kokoro (GPU) -> aplay. Run via robot/run.sh (needs the mic free).

    python -m k1_chat.robot_audio smoke "Hello, I am a K1."   # speak, hear one utterance, say it back
    python -m k1_chat.robot_audio listen                      # print every utterance until killed
"""
import argparse
import io
import os
import select
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

from k1_chat.audio import Audio, Segmenter

RATE, FRAME_MS = 16000, 30
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2
MIC = "plughw:CARD=Speaker,DEV=0"
SPEAKER = "plughw:CARD=Device,DEV=0"
HOME = Path(os.environ.get("K1_CHAT_HOME", Path.home() / "k1_chat"))


LIVE_CH, NCH = 4, 6  # the K1 mic array exposes 6 channels; 0-3 are live mics, 4-5 are silent (rms ~3)


def set_mic_gain(level: int | None = None) -> None:
    """Capture volume is 10/25 by default (peaks ~1500/32768); ALSA mixer setting, lost on reboot."""
    level = int(os.environ.get("K1_CHAT_MIC_GAIN", 16)) if level is None else level
    subprocess.run(["amixer", "-q", "-c", "Speaker", "cset", "name=Mic Capture Volume", ",".join([str(level)] * NCH)], check=False)


class ArecordCapture:
    """6-channel capture mixed down to mono (mean of the live channels); ALSA's own 6->1 mix includes the dead ones and loses ~40%."""

    def __init__(self, device: str = MIC):
        set_mic_gain()
        self._p = subprocess.Popen(["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(RATE), "-c", str(NCH), "-t", "raw"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def __iter__(self):
        import numpy as np
        size = FRAME_BYTES * NCH
        while chunk := self._p.stdout.read(size):
            if len(chunk) == size:
                x = np.frombuffer(chunk, np.int16).reshape(-1, NCH)[:, :LIVE_CH]
                yield x.astype(np.int32).mean(axis=1).astype(np.int16).tobytes()

    def drain(self) -> None:
        fd = self._p.stdout.fileno()
        while select.select([fd], [], [], 0)[0]:
            os.read(fd, 65536)

    def close(self) -> None:
        self._p.terminate()


class WhisperAsr:
    def __init__(self, model: str = "tiny.en"):
        from faster_whisper import WhisperModel
        self._m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=4)

    def __call__(self, pcm: bytes) -> str:
        import numpy as np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segs, _ = self._m.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segs)


class KokoroTts:
    """Kokoro-82M via the system onnxruntime-gpu (CUDA). ~0.2x real time on the K1's Orin; CPU is ~2.8x."""

    def __init__(self, model: Path = HOME / "models" / "kokoro-v1.0.onnx", voices: Path = HOME / "models" / "voices-v1.0.bin",
                 voice: str = "af_heart"):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        sess = ort.InferenceSession(str(model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if sess.get_providers()[0] != "CUDAExecutionProvider":
            print("WARNING: Kokoro is on CPU, expect ~3x slower than real time", file=sys.stderr, flush=True)
        self._k, self._voice = Kokoro.from_session(sess, str(voices)), voice

    def __call__(self, text: str) -> bytes:
        import numpy as np
        samples, sr = self._k.create(text, voice=self._voice, speed=1.0, lang="en-us")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
        return buf.getvalue()


class AplayPlayer:
    def __init__(self, device: str = SPEAKER):
        self._dev, self._p = device, None

    def play(self, wav: bytes) -> None:
        self._p = subprocess.Popen(["aplay", "-q", "-D", self._dev, "-"], stdin=subprocess.PIPE)
        try:
            self._p.communicate(wav)
        except BrokenPipeError:
            pass

    def stop(self) -> None:
        if self._p and self._p.poll() is None:
            self._p.terminate()


DUMP = HOME / "dump"


def save_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm)


def _dumped(asr, seg):
    """Wrap ASR: keep the exact PCM the model receives (utt_*.wav) plus a .txt with level stats, threshold and the text."""
    import numpy as np

    def wrapped(pcm: bytes) -> str:
        text = asr(pcm)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        x = np.frombuffer(pcm, np.int16).astype(np.float64)
        save_wav(DUMP / f"utt_{stamp}.wav", pcm)
        (DUMP / f"utt_{stamp}.txt").write_text(
            f"text: {text!r}\nsec: {len(x) / RATE:.2f}\nrms: {np.sqrt(np.mean(x ** 2)):.0f}\npeak: {int(np.abs(x).max())}\n"
            f"start_rms_threshold: {seg._start_rms:.0f}\n")
        for old in sorted(DUMP.glob("utt_*.wav"))[:-30]:  # keep the last 30
            old.unlink(); old.with_suffix(".txt").unlink(missing_ok=True)
        return text
    return wrapped


def _timed(name, fn, size=len):
    def wrapped(x):
        t = time.time()
        out = fn(x)
        print(f"[{name}] {time.time() - t:.2f}s in={size(x)} out={out if isinstance(out, str) else len(out)}", file=sys.stderr, flush=True)
        return out
    return wrapped


def build(concurrent: bool = False) -> Audio:
    asr = _timed("asr", WhisperAsr(), size=lambda pcm: f"{len(pcm) / 2 / RATE:.1f}s audio")
    tts = _timed("tts", KokoroTts(), size=lambda t: f"{len(t)} chars")
    start_rms = float(os.environ.get("K1_CHAT_START_RMS", _saved_rms()))
    seg = Segmenter(RATE, FRAME_MS, start_rms=start_rms, end_silence_ms=450)
    return Audio(ArecordCapture(), _dumped(asr, seg), tts, AplayPlayer(), seg, concurrent=concurrent)


def _saved_rms(default: float = 150.0) -> float:
    try:
        return float((HOME / "start_rms").read_text())
    except (OSError, ValueError):
        return default


def mic_test(secs: int = 6) -> int:
    """Record all 6 mic-array channels while you speak; print level + ASR per channel and for the mono mix, keep wavs."""
    import numpy as np
    set_mic_gain()
    print(f"SPEAK NOW for {secs}s (a full sentence, at the distance you will use) ...", flush=True)
    raw = subprocess.run(["arecord", "-q", "-D", MIC, "-f", "S16_LE", "-r", str(RATE), "-c", "6", "-t", "raw", "-d", str(secs)],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw[: len(raw) // 12 * 12], np.int16).reshape(-1, 6)
    asr = WhisperAsr()
    clips = {f"ch{i}": x[:, i] for i in range(6)}
    clips["mean"] = x.astype(np.float32).mean(axis=1).astype(np.int16)
    for name, c in clips.items():
        c = np.ascontiguousarray(c, np.int16)
        save_wav(DUMP / f"mictest_{name}.wav", c.tobytes())
        rms = float(np.sqrt(np.mean(c.astype(np.float64) ** 2))); peak = int(np.abs(c).max())
        print(f"{name:5s} rms {rms:6.0f} peak {peak:6d}  ASR: {asr(c.tobytes())!r}", flush=True)
    return 0


def calibrate(quiet_s: int = 3, speak_s: int = 5) -> int:
    """Measure room noise, then speech, save a VAD threshold to ~/k1_chat/start_rms and transcribe the speech clip."""
    import statistics
    cap = ArecordCapture()

    def record(secs):
        frames = []
        for f in cap:
            frames.append(f)
            if len(frames) * FRAME_MS >= secs * 1000:
                return frames
        return frames

    def rms(f):
        return Segmenter._rms(f)

    print(f"Stay SILENT for {quiet_s}s ...", flush=True)
    cap.drain()
    quiet_frames = record(quiet_s)
    noise = sorted(rms(f) for f in quiet_frames)
    print(f"SPEAK NOW for {speak_s}s (say a sentence at normal volume, standing where you will talk) ...", flush=True)
    cap.drain()
    speech_frames = record(speak_s)
    cap.close()
    save_wav(DUMP / "calib_quiet.wav", b"".join(quiet_frames))
    save_wav(DUMP / "calib_speech.wav", b"".join(speech_frames))
    loud = sorted(rms(f) for f in speech_frames)
    n95, s_med, s_top = noise[int(len(noise) * .95) - 1], statistics.median(loud[len(loud) // 2:]), loud[-1]
    thr = max(60.0, n95 * 2.5)
    if thr > s_med * 0.5:
        thr = max(60.0, (n95 + s_med) / 2) if s_med > n95 else thr
    print(f"noise p95 rms {n95:.0f}   speech median-of-loud-half {s_med:.0f}   speech peak {s_top:.0f}")
    if s_top < n95 * 2:
        print("WARN: speech barely above noise: speak louder/closer, or the mic is quiet. Threshold NOT saved.")
        verdict = 1
    else:
        HOME.mkdir(parents=True, exist_ok=True)
        (HOME / "start_rms").write_text(f"{thr:.0f}")
        print(f"saved start_rms = {thr:.0f}")
        verdict = 0
    heard = WhisperAsr()(b"".join(speech_frames))
    (DUMP / "calib_report.txt").write_text(
        f"noise_p95_rms: {n95:.0f}\nspeech_median_loud_half_rms: {s_med:.0f}\nspeech_peak_rms: {s_top:.0f}\n"
        f"threshold_saved: {f'{thr:.0f}' if verdict == 0 else 'no'}\nasr_text: {heard!r}\n")
    print("ASR heard:", repr(heard), flush=True)
    return verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("sentence", nargs="?", default="Hello, I am a Booster K1. Please say something.")
    sm.add_argument("--timeout", type=int, default=20)
    sub.add_parser("listen")
    sub.add_parser("chat")  # needs OPENROUTER_API_KEY + K1_CHAT_MODEL in the environment
    sub.add_parser("bench")
    sub.add_parser("calibrate")
    sub.add_parser("mic-test")
    af = sub.add_parser("asr-file")
    af.add_argument("wav")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.cmd == "calibrate":
        return calibrate()
    if args.cmd == "mic-test":
        return mic_test()
    a = build(concurrent=args.cmd == "chat")
    try:
        if args.cmd == "bench":
            wav = a._tts("Hello, I am a Booster K1 robot. It is nice to meet you. How can I help you today?")
            for _ in range(2):
                a._tts("Hello, I am a Booster K1 robot. It is nice to meet you. How can I help you today?")
            import numpy as np
            with wave.open(io.BytesIO(wav)) as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
                n = int(len(x) * RATE / w.getframerate())
                pcm16 = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16).tobytes()
            print("ASR:", a._asr(pcm16), flush=True)
            print("ASR:", a._asr(pcm16), flush=True)
        elif args.cmd == "asr-file":
            with wave.open(args.wav) as w:
                assert w.getframerate() == RATE and w.getnchannels() == 1, "need 16 kHz mono wav"
                print("ASR:", a._asr(w.readframes(w.getnframes())), flush=True)
        elif args.cmd == "smoke":
            a.speak(args.sentence)
            print("listening...", flush=True)
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(args.timeout)
            try:
                heard = next(a.listen())
            except (TimeoutError, StopIteration):
                print(f"HEARD: <nothing> (mic peak rms {a._seg.peak:.0f}, threshold {a._seg._start_rms:.0f})", flush=True)
                return 1
            finally:
                signal.alarm(0)
            print(f"HEARD: {heard}", flush=True)
            a.speak(f"I heard: {heard}")
        elif args.cmd == "chat":
            from k1_chat.conversation import Conversation, run_async_voice_chat
            from k1_chat.llm import from_env

            def heard():
                for text in a.listen():
                    print(f"HEARD: {text}", flush=True)
                    yield text

            def say(text):
                t = time.time()
                print(f"SAID: {text}", flush=True)
                a.speak(text)
                print(f"[speak] {time.time() - t:.2f}s", file=sys.stderr, flush=True)

            run_async_voice_chat(heard(), Conversation(from_env(), history_file=HOME / "history.json"), say)
        else:
            for text in a.listen():
                print(f"HEARD: {text}", flush=True)
    finally:
        a._cap.close()
        a.stop_speaking()
    return 0


if __name__ == "__main__":
    sys.exit(main())
