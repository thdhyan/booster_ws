"""Mic -> ASR utterances and text -> TTS over the robot LUI service. tests/test_audio.py defines the contract.

The transport is injected: `call(api_id, body)` returns 0 on success, `asr_chunks()` yields dicts
{"text": str, "is_final": bool}. The chunk shape is an assumption until the AsrChunk idl is checked on the robot.
"""
import json
from enum import IntEnum
from typing import Iterator, Protocol


class AudioError(Exception):
    pass


class LuiApi(IntEnum):
    START_ASR = 1000
    STOP_ASR = 1001
    START_TTS = 1050
    STOP_TTS = 1051
    SEND_TTS_TEXT = 1052


class Transport(Protocol):
    def call(self, api_id: int, body: str) -> int: ...
    def asr_chunks(self) -> Iterator[dict]: ...


class Audio:
    def __init__(self, transport: Transport, voice: str = "zh_male_wennuanahu_moon_bigtts"):
        self._t, self._voice, self._tts_on = transport, voice, False

    def _call(self, api: LuiApi, body: str = "") -> None:
        rc = self._t.call(int(api), body)
        if rc != 0:
            raise AudioError(f"{api.name} failed rc={rc}")

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        if not self._tts_on:
            self._call(LuiApi.START_TTS, json.dumps({"voice_type": self._voice}))
            self._tts_on = True
        self._call(LuiApi.SEND_TTS_TEXT, json.dumps({"text": text}))

    def stop_speaking(self) -> None:
        self._call(LuiApi.STOP_TTS)
        self._tts_on = False

    def listen(self) -> Iterator[str]:
        self._call(LuiApi.START_ASR)
        try:
            for chunk in self._t.asr_chunks():
                text = chunk.get("text", "").strip()
                if chunk.get("is_final") and text:
                    yield text
        finally:
            self._call(LuiApi.STOP_ASR)
