from dataclasses import dataclass

import httpx

from portfolio_chat.settings import Settings


class OpenRouterError(Exception):
    """The model call failed. The message stays off the client response."""


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    total_tokens: int


@dataclass(frozen=True)
class Completion:
    text: str
    total_tokens: int


class OpenRouter:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.Client(
            base_url=settings.openrouter_base_url.rstrip("/"),
            timeout=30.0,
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        )

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        payload = self._post("/embeddings", {"model": self._settings.embed_model, "input": texts})
        data = payload.get("data")
        if not isinstance(data, list):
            raise OpenRouterError("unexpected response")
        rows = [row for row in data if isinstance(row, dict)]
        rows.sort(key=lambda row: int(row["index"]))
        vectors: list[list[float]] = []
        for row in rows:
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                raise OpenRouterError("unexpected response")
            vectors.append([float(value) for value in embedding])
        return EmbeddingBatch(vectors=vectors, total_tokens=_total_tokens(payload))

    def complete(self, system: str, user: str) -> Completion:
        payload = self._post(
            "/chat/completions",
            {
                "model": self._settings.chat_model,
                "temperature": 0,
                "max_tokens": self._settings.max_output_tokens,
                "reasoning": {"effort": "none"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OpenRouterError("unexpected response")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise OpenRouterError("unexpected response")
        content = message.get("content")
        if not isinstance(content, str):
            raise OpenRouterError("unexpected response")
        return Completion(text=content.strip(), total_tokens=_total_tokens(payload))

    def _post(self, path: str, body: dict[str, object]) -> dict[str, object]:
        try:
            response = self._client.post(path, json=body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise OpenRouterError(str(error)) from error
        data = response.json()
        if not isinstance(data, dict):
            raise OpenRouterError("unexpected response")
        return data


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
