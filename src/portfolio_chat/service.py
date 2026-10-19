from typing import Protocol

from portfolio_chat.gate import Gate
from portfolio_chat.index import Index
from portfolio_chat.openrouter import Completion, EmbeddingBatch, OpenRouterError
from portfolio_chat.prompt import closing_answer, missing_answer, render_prompt
from portfolio_chat.schemas import Message
from portfolio_chat.settings import Settings


class NotReady(Exception):
    """The index has no sections yet."""


class LanguageModel(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingBatch: ...

    def complete(self, system: str, user: str) -> Completion: ...


class ChatService:
    def __init__(self, settings: Settings, index: Index, gate: Gate, model: LanguageModel) -> None:
        self.settings = settings
        self._index = index
        self._gate = gate
        self._model = model

    def ready(self) -> bool:
        return self._index.count() > 0

    def answer(self, messages: list[Message], client_ip: str) -> str:
        self._gate.admit(client_ip, messages)
        if not self.ready():
            raise NotReady
        embedded = self._model.embed([retrieval_text(messages)])
        hits = self._index.search(
            embedded.vectors[0],
            self.settings.top_k,
            self.settings.score_floor,
        )
        if not hits:
            self._gate.record(client_ip, embedded.total_tokens)
            return missing_answer(self.settings.whatsapp_url)
        system, user = render_prompt(self.settings, hits, messages)
        estimate = _estimate_tokens(system + user, self.settings.max_output_tokens)
        self._gate.reserve(estimate)
        try:
            completion = self._model.complete(system, user)
        except OpenRouterError:
            self._gate.record(client_ip, embedded.total_tokens + estimate)
            raise
        self._gate.record(client_ip, embedded.total_tokens + completion.total_tokens)
        text = completion.text.strip()
        user_turns = sum(1 for message in messages if message.role == "user")
        if user_turns >= self.settings.max_turns:
            text = f"{text}\n\n{closing_answer(self.settings.whatsapp_url)}"
        return text


def retrieval_text(messages: list[Message]) -> str:
    """Join the previous user turn so a follow-up keeps its subject."""
    users = [message.content for message in messages if message.role == "user"]
    current = users[-1]
    if len(users) == 1:
        return current
    return f"{users[-2]}\n{current}"


def _estimate_tokens(text: str, max_output: int) -> int:
    return max(1, len(text) // 4) + max_output
