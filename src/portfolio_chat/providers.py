import logging
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from portfolio_chat.settings import Settings

logger = logging.getLogger("portfolio_chat")


class ModelError(Exception):
    """The model call failed. The message stays off the client response."""


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    total_tokens: int


@dataclass(frozen=True)
class Completion:
    text: str
    total_tokens: int


class LanguageModel(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingBatch: ...

    def complete(self, system: str, user: str) -> Completion: ...


class ChatProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        embed_model: str,
        chat_model: str,
        max_output_tokens: int,
        style: Literal["openai", "openrouter"],
        client: httpx.Client | None = None,
    ) -> None:
        self.name = name
        self._embed_model = embed_model
        self._chat_model = chat_model
        self._max_output_tokens = max_output_tokens
        self._style = style
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=30.0,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        payload = self._post("/embeddings", {"model": self._embed_model, "input": texts})
        data = payload.get("data")
        if not isinstance(data, list):
            raise ModelError("unexpected response")
        rows = [row for row in data if isinstance(row, dict)]
        rows.sort(key=lambda row: int(row["index"]))
        vectors: list[list[float]] = []
        for row in rows:
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                raise ModelError("unexpected response")
            vectors.append([float(value) for value in embedding])
        return EmbeddingBatch(vectors=vectors, total_tokens=_total_tokens(payload))

    def complete(self, system: str, user: str) -> Completion:
        payload = self._post("/chat/completions", self._chat_body(system, user))
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelError("unexpected response")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ModelError("unexpected response")
        return Completion(text=_message_text(message.get("content")), total_tokens=_total_tokens(payload))

    def _chat_body(self, system: str, user: str) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self._chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._style == "openai":
            body["temperature"] = 0
            body["max_tokens"] = self._max_output_tokens
            return body
        body["temperature"] = 0
        body["max_tokens"] = self._max_output_tokens
        body["reasoning"] = {"effort": "none"}
        return body

    def _post(self, path: str, body: dict[str, object]) -> dict[str, object]:
        try:
            response = self._client.post(path, json=body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelError(str(error)) from error
        data = response.json()
        if not isinstance(data, dict):
            raise ModelError("unexpected response")
        return data


class FallbackProvider:
    """Try the primary provider. The second provider runs only after that call fails."""

    def __init__(self, primary: LanguageModel, fallback: LanguageModel) -> None:
        self.primary = primary
        self.fallback = fallback

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        try:
            return self.primary.embed(texts)
        except ModelError:
            logger.info("model fallback operation=embed")
            return self.fallback.embed(texts)

    def complete(self, system: str, user: str) -> Completion:
        try:
            return self.primary.complete(system, user)
        except ModelError:
            logger.info("model fallback operation=complete")
            return self.fallback.complete(system, user)


def build_provider(settings: Settings) -> ChatProvider | FallbackProvider:
    openai = _openai_provider(settings) if settings.openai_api_key else None
    openrouter = _openrouter_provider(settings) if settings.openrouter_api_key else None
    if openai and openrouter:
        return FallbackProvider(openai, openrouter)
    if openai:
        return openai
    return _openrouter_provider(settings)


def _openai_provider(settings: Settings) -> ChatProvider:
    return ChatProvider(
        name="openai",
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        embed_model=settings.openai_embed_model,
        chat_model=settings.openai_chat_model,
        max_output_tokens=settings.max_output_tokens,
        style="openai",
    )


def _openrouter_provider(settings: Settings) -> ChatProvider:
    return ChatProvider(
        name="openrouter",
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        embed_model=settings.openrouter_embed_model,
        chat_model=settings.openrouter_chat_model,
        max_output_tokens=settings.max_output_tokens,
        style="openrouter",
    )


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts).strip()
    raise ModelError("unexpected response")


def _total_tokens(payload: dict[str, object]) -> int:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    prompt_tokens = prompt if isinstance(prompt, int) else 0
    completion_tokens = completion if isinstance(completion, int) else 0
    return prompt_tokens + completion_tokens
