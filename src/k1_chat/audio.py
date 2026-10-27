"""Local voice pipeline: mic frames -> VAD segmenter -> ASR text, and text -> TTS wav -> player.

Capture, ASR, TTS and player are injected so the logic is testable; the real ones live in
robot_audio.py and run on the K1. tests/test_audio.py defines the contract.
"""
import array
import math
import re
import time
from collections import deque
from typing import Callable, Iterable, Iterator, Protocol

# Whisper emits these on near-silence; they are never real user speech.
_HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "[blank_audio]", "(silence)"}


class Player(Protocol):
    def play(self, wav: bytes) -> None: ...
    def stop(self) -> None: ...


class Segmenter:
    """Energy VAD over fixed-size 16-bit mono PCM frames. feed() returns a finished utterance or None."""

    def __init__(self, rate=16000, frame_ms=30, start_rms=500.0, end_silence_ms=700, min_ms=300,
                 max_ms=15000, preroll_ms=300):
        ms = lambda v: max(1, v // frame_ms)
        self._start_rms = start_rms
        self._end, self._min, self._max = ms(end_silence_ms), ms(min_ms), ms(max_ms)
        self._pre = deque(maxlen=ms(preroll_ms))
        self.peak = 0.0  # loudest frame RMS seen, for tuning start_rms
        self.reset()

    def reset(self) -> None:
        self._pre.clear()
        self._buf: list[bytes] = []
        self._speech = 0
        self._quiet = 0

    @staticmethod
    def _rms(f: bytes) -> float:
        s = array.array("h", f)
        return math.sqrt(sum(x * x for x in s) / len(s)) if s else 0.0

    def feed(self, f: bytes) -> bytes | None:
        rms = self._rms(f)
        self.peak = max(self.peak, rms)
        loud = rms >= self._start_rms
        if not self._buf:
            if not loud:
                self._pre.append(f)
                return None
            self._buf = list(self._pre) + [f]
            self._pre.clear()
            self._speech, self._quiet = 1, 0
            return None
        self._buf.append(f)
        if loud:
            self._speech += 1
            self._quiet = 0
        else:
            self._quiet += 1
        if self._quiet >= self._end or len(self._buf) >= self._max:
            utt, ok = b"".join(self._buf), self._speech >= self._min
            self.reset()
            return utt if ok else None
        return None


class Audio:
    def __init__(self, capture: Iterable[bytes], asr: Callable[[bytes], str], tts: Callable[[str], bytes],
                 player: Player, segmenter: Segmenter, concurrent: bool = False, echo_tail_s: float = 0.5):
        self._cap, self._asr, self._tts, self._player, self._seg = capture, asr, tts, player, segmenter
        # concurrent: listen() runs in another thread while we speak; it alone reads the mic and discards our own voice
        self._concurrent, self._echo_tail, self._speaking, self._mute_until = concurrent, echo_tail_s, False, 0.0

    def listen(self) -> Iterator[str]:
        for f in self._cap:
            if self._speaking or time.monotonic() < self._mute_until:
                self._seg.reset()
                continue
            utt = self._seg.feed(f)
            if utt is None:
                continue
            text = self._asr(utt).strip()
            if text.lower().strip(".!? ") not in _HALLUCINATIONS:
                yield text

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._speaking = True
        try:
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):  # first audio starts after one sentence, not all
                if sentence:
                    self._player.play(self._tts(sentence))
        finally:
            self._mute_until = time.monotonic() + self._echo_tail
            self._speaking = False
        if not self._concurrent:
            self._seg.reset()
            self._cap.drain()  # drop what the mic heard of our own voice

    def stop_speaking(self) -> None:
        self._player.stop()
