import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx

from portfolio_chat.providers import (
    ChatProvider,
    Completion,
    EmbeddingBatch,
    FallbackProvider,
    ModelError,
    build_provider,
)
from tests.settings_factory import make_settings


class Primary:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        self.calls += 1
        if self.fail:
            raise ModelError("down")
        return EmbeddingBatch(vectors=[[1.0] for _ in texts], total_tokens=1)

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        if self.fail:
            raise ModelError("down")
        return Completion(text="primary", total_tokens=2)


class Secondary:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        self.calls += 1
        return EmbeddingBatch(vectors=[[0.0] for _ in texts], total_tokens=3)

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        return Completion(text="fallback", total_tokens=4)


def test_primary_success_skips_the_fallback() -> None:
    primary = Primary()
    fallback = Secondary()
    model = FallbackProvider(primary, fallback)

    assert model.embed(["KingPanda"]).vectors == [[1.0]]
    assert model.complete("system", "user").text == "primary"
    assert fallback.calls == 0


def test_primary_failure_uses_the_fallback() -> None:
    primary = Primary(fail=True)
    fallback = Secondary()
    model = FallbackProvider(primary, fallback)

    assert model.embed(["KingPanda"]).total_tokens == 3
    assert model.complete("system", "user").text == "fallback"
    assert primary.calls == 2
    assert fallback.calls == 2


def test_openai_is_primary_when_both_keys_exist(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, openai_api_key="sk-test")
    model = build_provider(settings)

    assert isinstance(model, FallbackProvider)
    assert isinstance(model.primary, ChatProvider)
    assert isinstance(model.fallback, ChatProvider)
    assert model.primary.name == "openai"
    assert model.fallback.name == "openrouter"


def test_missing_openai_key_uses_openrouter_only(tmp_path: Path) -> None:
    model = build_provider(make_settings(tmp_path))

    assert isinstance(model, ChatProvider)
    assert model.name == "openrouter"


def test_openai_chat_uses_gpt_4o_mini() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}},
        )

    provider = _provider("openai", handler)
    completion = provider.complete("system", "user")

    assert completion.text == "ok"
    assert seen["model"] == "gpt-4o-mini"
    assert seen["temperature"] == 0
    assert seen["max_tokens"] == 40
    assert "reasoning_effort" not in seen
    assert "reasoning" not in seen


def test_openrouter_chat_keeps_its_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}},
        )

    provider = _provider("openrouter", handler)
    provider.complete("system", "user")

    assert seen["model"] == "deepseek/deepseek-v4-flash"
    assert seen["temperature"] == 0
    assert seen["max_tokens"] == 40
    assert seen["reasoning"] == {"effort": "none"}
    assert "reasoning_effort" not in seen


def _provider(style: Literal["openai", "openrouter"], handler: Callable[[httpx.Request], httpx.Response]) -> ChatProvider:
    if style == "openai":
        return ChatProvider(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            embed_model="text-embedding-3-small",
            chat_model="gpt-4o-mini",
            max_output_tokens=40,
            style="openai",
            client=httpx.Client(
                base_url="https://api.openai.com/v1",
                transport=httpx.MockTransport(handler),
            ),
        )
    return ChatProvider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="or-test",
        embed_model="openai/text-embedding-3-small",
        chat_model="deepseek/deepseek-v4-flash",
        max_output_tokens=40,
        style="openrouter",
        client=httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            transport=httpx.MockTransport(handler),
        ),
    )
