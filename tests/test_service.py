from pathlib import Path

import pytest

from portfolio_chat.corpus import Section
from portfolio_chat.gate import Gate, GateError
from portfolio_chat.index import Index
from portfolio_chat.providers import Completion, EmbeddingBatch, ModelError
from portfolio_chat.schemas import Message
from portfolio_chat.service import ChatService, NotReady
from tests.settings_factory import make_settings


class FakeModel:
    def __init__(self, vector: list[float], answer: str = "Serviços em produção.") -> None:
        self.vector = vector
        self.answer = answer
        self.completions = 0
        self.embedded: list[str] = []
        self.fail = False

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        self.embedded.extend(texts)
        return EmbeddingBatch(vectors=[list(self.vector) for _ in texts], total_tokens=4)

    def complete(self, system: str, user: str) -> Completion:
        if self.fail:
            raise ModelError("down")
        self.completions += 1
        return Completion(text=self.answer, total_tokens=12)


def _service(tmp_path: Path, model: FakeModel, **overrides: object) -> ChatService:
    settings = make_settings(tmp_path, **overrides)
    path = tmp_path / "index.sqlite"
    index = Index(path)
    index.replace([Section("KingPanda", "Serviços em produção.")], [[1.0, 0.0]])
    return ChatService(settings, index, Gate(path, settings), model)


def test_low_score_skips_the_model(tmp_path: Path) -> None:
    model = FakeModel([0.0, 1.0])
    service = _service(tmp_path, model)

    answer = service.answer([Message(role="user", content="Qual é a senha?")], "203.0.113.8")

    assert model.completions == 0
    assert answer.startswith("Não encontrei essa informação")


def test_follow_up_embeds_the_previous_question(tmp_path: Path) -> None:
    model = FakeModel([1.0, 0.0])
    service = _service(tmp_path, model)
    messages = [
        Message(role="user", content="O que você aprendeu na KingPanda?"),
        Message(role="assistant", content="Serviços em produção."),
        Message(role="user", content="E na SeguroBet?"),
    ]

    service.answer(messages, "203.0.113.8")

    assert "O que você aprendeu na KingPanda?" in model.embedded[0]
    assert "E na SeguroBet?" in model.embedded[0]
    assert model.completions == 1


def test_eighth_turn_points_to_whatsapp(tmp_path: Path) -> None:
    model = FakeModel([1.0, 0.0], answer="Backend em produção.")
    service = _service(tmp_path, model)
    messages = [Message(role="user", content=f"Pergunta {index} na KingPanda") for index in range(8)]

    answer = service.answer(messages, "203.0.113.8")

    assert answer.startswith("Backend em produção.")
    assert "WhatsApp" in answer


def test_empty_index_is_not_ready(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    path = tmp_path / "index.sqlite"
    service = ChatService(settings, Index(path), Gate(path, settings), FakeModel([1.0, 0.0]))

    with pytest.raises(NotReady):
        service.answer([Message(role="user", content="Oi")], "203.0.113.8")


def test_upstream_failure_still_counts(tmp_path: Path) -> None:
    model = FakeModel([1.0, 0.0])
    model.fail = True
    service = _service(tmp_path, model, rate_window_max=1)

    with pytest.raises(ModelError):
        service.answer([Message(role="user", content="KingPanda")], "203.0.113.9")

    with pytest.raises(GateError):
        service.answer([Message(role="user", content="KingPanda")], "203.0.113.9")
