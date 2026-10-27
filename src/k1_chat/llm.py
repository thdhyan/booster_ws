"""OpenRouter chat client (free models only). tests/test_llm.py defines the contract."""
import os
import time

import httpx


class LLMError(Exception):
    pass


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1",
                 retries: int = 3, backoff: float = 0.25, timeout: float = 60.0, sleep=time.sleep):
        if not model.endswith(":free"):
            raise ValueError(f"only ':free' models allowed, got {model!r}")
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.retries, self.backoff, self.timeout, self._sleep = retries, backoff, timeout, sleep

    def chat(self, messages: list[dict], **kw) -> str:
        body = {"model": self.model, "messages": messages, **kw}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last = "no attempt"
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(self.backoff * 2 ** (attempt - 1))
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
            except httpx.TransportError as e:
                last = f"transport error: {e}"
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
            try:
                return r.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError, ValueError) as e:
                raise LLMError(f"bad response shape: {e}") from e
        raise LLMError(f"gave up after {self.retries + 1} attempts: {last}")


def from_env(model_var: str = "K1_CHAT_MODEL") -> OpenRouterClient:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY not set")
    model = os.environ.get(model_var)
    if not model:
        raise LLMError(f"{model_var} not set")
    return OpenRouterClient(key, model)
