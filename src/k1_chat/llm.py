"""OpenRouter chat client (free models). TDD stub: tests/test_llm.py defines the contract."""


class LLMError(Exception):
    pass


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        raise NotImplementedError

    def chat(self, messages: list[dict], **kw) -> str:
        raise NotImplementedError


def from_env(model_var: str = "K1_CHAT_MODEL") -> OpenRouterClient:
    raise NotImplementedError
