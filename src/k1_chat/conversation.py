"""Voice conversation: transcript in, short spoken reply out. Speech only: nothing here can command the robot."""
import sys
from pathlib import Path
from typing import Callable, Iterable

SYSTEM = (Path(__file__).with_name("system_prompt.md")).read_text().strip()  # edit that file; k1voice.sh syncs it
FALLBACK = "Sorry, I could not think of an answer just now."


class Conversation:
    def __init__(self, llm, system: str = SYSTEM, max_turns: int = 6):
        self._llm, self._system, self._max = llm, system, max_turns
        self.history: list[dict] = []

    def reply(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-(2 * self._max - 1):]
        try:
            answer = (self._llm.chat([{"role": "system", "content": self._system}, *self.history], max_tokens=400) or "").strip()
        except Exception as e:
            print(f"[llm] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            self.history.pop()
            return FALLBACK
        self.history.append({"role": "assistant", "content": answer})
        return answer or FALLBACK


def run_voice_chat(transcripts: Iterable[str], conv: Conversation, say: Callable[[str], None]) -> None:
    for text in transcripts:
        say(conv.reply(text))
