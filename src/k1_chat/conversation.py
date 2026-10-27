"""Voice conversation: transcript in, short spoken reply out. Speech only: nothing here can command the robot."""
import json
import sys
import threading
from pathlib import Path
from typing import Callable, Iterable



def load_system(base: Path) -> str:
    """system_prompt.md (persona) followed by every knowledge/*.md in name order."""
    parts = [(base / "system_prompt.md").read_text().strip()]
    parts += [f.read_text().strip() for f in sorted((base / "knowledge").glob("*.md"))]
    return "\n\n".join(parts)


SYSTEM = load_system(Path(__file__).parent)  # edit system_prompt.md / knowledge/*.md; k1voice.sh syncs them
FALLBACK = "Sorry, I could not think of an answer just now."


class Conversation:
    """Thread-safe. A reply is dropped (None) when a newer utterance arrived while the LLM was thinking; the newer call
    then carries both user turns merged into one message."""

    def __init__(self, llm, system: str = SYSTEM, max_turns: int = 20, history_file: Path | None = None):
        self._llm, self._system, self._max, self._file = llm, system, max_turns, history_file
        self.history: list[dict] = self._load()
        self._lock, self._seq = threading.Lock(), 0

    def _load(self) -> list[dict]:
        try:
            h = json.loads(self._file.read_text()) if self._file else []
            return h[-(2 * self._max - 1):] if isinstance(h, list) else []
        except (OSError, ValueError):
            return []

    def _save(self) -> None:
        if self._file:
            self._file.write_text(json.dumps(self.history))

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
            answer = (self._llm.chat(msgs, max_tokens=2000) or "").strip()  # reasoning models spend part of this thinking
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
            self._save()
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
