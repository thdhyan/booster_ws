"""On-robot audio: arecord -> Segmenter -> faster-whisper, piper -> aplay. Run via robot/run.sh (needs the mic free).

    python -m k1_chat.robot_audio smoke "Hello, I am a K1."   # speak, hear one utterance, say it back
    python -m k1_chat.robot_audio listen                      # print every utterance until killed
"""
import argparse
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from k1_chat.audio import Audio, Segmenter

RATE, FRAME_MS = 16000, 30
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2
MIC = "plughw:CARD=Speaker,DEV=0"
SPEAKER = "plughw:CARD=Device,DEV=0"
HOME = Path(os.environ.get("K1_CHAT_HOME", Path.home() / "k1_chat"))


class ArecordCapture:
    def __init__(self, device: str = MIC):
        self._p = subprocess.Popen(["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def __iter__(self):
        while chunk := self._p.stdout.read(FRAME_BYTES):
            if len(chunk) == FRAME_BYTES:
                yield chunk

    def drain(self) -> None:
        fd = self._p.stdout.fileno()
        while select.select([fd], [], [], 0)[0]:
            os.read(fd, 65536)

    def close(self) -> None:
        self._p.terminate()


class WhisperAsr:
    def __init__(self, model: str = "base.en"):
        from faster_whisper import WhisperModel
        self._m = WhisperModel(model, device="cpu", compute_type="int8")

    def __call__(self, pcm: bytes) -> str:
        import numpy as np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segs, _ = self._m.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segs)


class PiperTts:
    def __init__(self, model: Path = HOME / "models" / "en_US-lessac-medium.onnx"):
        self._model = model

    def __call__(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:  # piper 1.8 writes nothing to stdout without -f
            r = subprocess.run([sys.executable, "-m", "piper", "-m", str(self._model), "-f", f.name],
                               input=text.encode(), capture_output=True)
            if r.returncode:
                raise RuntimeError(f"piper failed: {r.stderr.decode()[-300:]}")
            return Path(f.name).read_bytes()


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


def _timed(name, fn, size=len):
    def wrapped(x):
        t = time.time()
        out = fn(x)
        print(f"[{name}] {time.time() - t:.2f}s in={size(x)} out={out if isinstance(out, str) else len(out)}", file=sys.stderr, flush=True)
        return out
    return wrapped


def build() -> Audio:
    asr = _timed("asr", WhisperAsr(), size=lambda pcm: f"{len(pcm) / 2 / RATE:.1f}s audio")
    tts = _timed("tts", PiperTts(), size=lambda t: f"{len(t)} chars")
    return Audio(ArecordCapture(), asr, tts, AplayPlayer(), Segmenter(RATE, FRAME_MS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("sentence", nargs="?", default="Hello, I am a Booster K1. Please say something.")
    sm.add_argument("--timeout", type=int, default=20)
    sub.add_parser("listen")
    af = sub.add_parser("asr-file")
    af.add_argument("wav")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    a = build()
    try:
        if args.cmd == "asr-file":
            import wave
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
                print(f"HEARD: <nothing> (mic peak rms {a._seg.peak:.0f}, threshold 500)", flush=True)
                return 1
            finally:
                signal.alarm(0)
            print(f"HEARD: {heard}", flush=True)
            a.speak(f"I heard: {heard}")
        else:
            for text in a.listen():
                print(f"HEARD: {text}", flush=True)
    finally:
        a._cap.close()
        a.stop_speaking()
    return 0


if __name__ == "__main__":
    sys.exit(main())
