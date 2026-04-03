"""Smoke test: verify the package imports correctly."""


def test_imports():
    from empujon_llm import LLMMultiplexer, LLMResponse, LLMProvider, LLMException
    assert LLMMultiplexer is not None
    assert LLMResponse is not None
    assert LLMProvider.AUTO.value == "auto"


def test_types():
    from empujon_llm import LLMMessage, LLMRequest
    msg = LLMMessage(role="user", content="hello")
    assert msg.role == "user"
    req = LLMRequest(model="gpt-5-mini", messages=[msg])
    assert req.model == "gpt-5-mini"
