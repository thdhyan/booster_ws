import json

import pytest
from k1_chat.audio import Audio, AudioError, LuiApi


class FakeTransport:
    def __init__(self, chunks=(), rc=0):
        self.calls, self.chunks, self.rc = [], list(chunks), rc

    def call(self, api_id: int, body: str) -> int:
        self.calls.append((api_id, body))
        return self.rc

    def asr_chunks(self):
        yield from self.chunks


def test_speak_starts_tts_once_then_sends_text():
    t = FakeTransport(); a = Audio(t, voice="v1")
    a.speak("hello"); a.speak("again")
    assert [c[0] for c in t.calls] == [LuiApi.START_TTS, LuiApi.SEND_TTS_TEXT, LuiApi.SEND_TTS_TEXT]
    assert json.loads(t.calls[0][1]) == {"voice_type": "v1"}
    assert json.loads(t.calls[1][1]) == {"text": "hello"}


def test_speak_empty_text_is_noop():
    t = FakeTransport(); Audio(t).speak("  ")
    assert t.calls == []


def test_speak_nonzero_rc_raises():
    with pytest.raises(AudioError):
        Audio(FakeTransport(rc=-1)).speak("hi")


def test_listen_yields_only_final_utterances_and_stops_asr():
    t = FakeTransport([{"text": "hel", "is_final": False}, {"text": "hello robot", "is_final": True},
                       {"text": "", "is_final": True}, {"text": "stop", "is_final": True}])
    a = Audio(t)
    assert list(a.listen()) == ["hello robot", "stop"]
    assert [c[0] for c in t.calls] == [LuiApi.START_ASR, LuiApi.STOP_ASR]


def test_listen_stops_asr_when_consumer_breaks():
    t = FakeTransport([{"text": "a", "is_final": True}, {"text": "b", "is_final": True}])
    for _ in Audio(t).listen():
        break
    assert t.calls[-1][0] == LuiApi.STOP_ASR


def test_stop_speaking_sends_stop_tts():
    t = FakeTransport(); a = Audio(t); a.speak("x"); a.stop_speaking()
    assert t.calls[-1][0] == LuiApi.STOP_TTS
    a.speak("y")  # TTS must be re-started after a stop
    assert t.calls[-2][0] == LuiApi.START_TTS


@pytest.mark.robot
def test_robot_speaks_fixed_sentence():
    pytest.skip("RobotTransport not written yet; needs the A2 LUI transport")
