import httpx, pytest, respx
from k1_chat.llm import LLMError, OpenRouterClient, from_env

URL = "https://openrouter.ai/api/v1/chat/completions"


@respx.mock
def test_chat_returns_text_and_sends_auth_and_model():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}))
    c = OpenRouterClient("sk-test", "x/y:free")
    assert c.chat([{"role": "user", "content": "hello"}]) == "hi"
    req = route.calls[0].request
    assert req.headers["authorization"] == "Bearer sk-test"
    assert b'"model":"x/y:free"' in req.content.replace(b" ", b"")


@respx.mock
def test_chat_retries_on_429_then_succeeds():
    respx.post(URL).mock(side_effect=[httpx.Response(429), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})])
    assert OpenRouterClient("k", "m:free").chat([{"role": "user", "content": "x"}]) == "ok"


@respx.mock
def test_chat_raises_llmerror_after_retries():
    respx.post(URL).mock(return_value=httpx.Response(500))
    with pytest.raises(LLMError):
        OpenRouterClient("k", "m:free").chat([{"role": "user", "content": "x"}])


def test_refuses_paid_model():
    with pytest.raises(ValueError):
        OpenRouterClient("k", "openrouter/auto")


def test_from_env_reads_key_and_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    monkeypatch.setenv("K1_CHAT_MODEL", "a/b:free")
    assert from_env() is not None


@pytest.mark.net
def test_live_free_model_answers():
    from dotenv import load_dotenv; load_dotenv()
    assert from_env().chat([{"role": "user", "content": "Reply with the word pong."}])
