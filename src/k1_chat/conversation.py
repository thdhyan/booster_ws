"""Voice conversation: transcript in, short spoken reply out. Speech only: nothing here can command the robot."""
import sys
import threading
from pathlib import Path
from typing import Callable, Iterable

SYSTEM = (Path(__file__).with_name("system_prompt.md")).read_text().strip()  # edit that file; k1voice.sh syncs it
FALLBACK = "Sorry, I could not think of an answer just now."


class Conversation:
    """Thread-safe. A reply is dropped (None) when a newer utterance arrived while the LLM was thinking; the newer call
    then carries both user turns merged into one message."""

    def __init__(self, llm, system: str = SYSTEM, max_turns: int = 6):
        self._llm, self._system, self._max = llm, system, max_turns
        self.history: list[dict] = []
        self._lock, self._seq = threading.Lock(), 0

    def reply(self, text: str) -> str | None:
        with self._lock:
            self._seq += 1
            mine = self._seq
            if self.history and self.history[-1]["role"] == "user":
                self.history[-1]["content"] += " " + text
            else:
                self.history.append({"role": "user", "content": text})
            self.history = self.history[-(2 * self._max - 1):]
            msgs = [{"role": "system", "content": self._system}, *map(dict, self.history)]
        ok = True
        try:
            answer = (self._llm.chat(msgs, max_tokens=400) or "").strip()
        except Exception as e:
            print(f"[llm] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            answer = ""
        if not answer:
            answer, ok = FALLBACK, False
        with self._lock:
            if mine != self._seq:
                return None
            if ok:
                self.history.append({"role": "assistant", "content": answer})
            else:
                self.history.pop()
            return answer


def run_voice_chat(transcripts: Iterable[str], conv: Conversation, say: Callable[[str], None]) -> None:
    for text in transcripts:
        say(conv.reply(text))


def run_async_voice_chat(transcripts: Iterable[str], conv: Conversation, say: Callable[[str], None]) -> None:
    """Keep listening while the LLM thinks and the robot speaks: each transcript starts its own LLM call in a thread,
    and a call superseded by a newer utterance is never spoken."""
    speak_lock = threading.Lock()
    threads: list[threading.Thread] = []

    def handle(text: str) -> None:
        answer = conv.reply(text)
        if answer is not None:
            with speak_lock:
                say(answer)

    for text in transcripts:
        t = threading.Thread(target=handle, args=(text,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
