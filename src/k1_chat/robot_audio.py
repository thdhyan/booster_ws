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
        r = subprocess.run([sys.executable, "-m", "piper", "-m", str(self._model)], input=text.encode(),
                           capture_output=True, check=True)
        return r.stdout


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


def build() -> Audio:
    return Audio(ArecordCapture(), WhisperAsr(), PiperTts(), AplayPlayer(), Segmenter(RATE, FRAME_MS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("sentence", nargs="?", default="Hello, I am a Booster K1. Please say something.")
    sm.add_argument("--timeout", type=int, default=20)
    sub.add_parser("listen")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    a = build()
    try:
        if args.cmd == "smoke":
            a.speak(args.sentence)
            print("listening...", flush=True)
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(args.timeout)
            try:
                heard = next(a.listen())
            except (TimeoutError, StopIteration):
                print("HEARD: <nothing>", flush=True)
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
