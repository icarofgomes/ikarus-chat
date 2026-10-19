from pathlib import Path

from fastapi.testclient import TestClient

from portfolio_chat.corpus import Section
from portfolio_chat.gate import Gate
from portfolio_chat.index import Index
from portfolio_chat.main import create_app
from portfolio_chat.openrouter import Completion, EmbeddingBatch
from portfolio_chat.service import ChatService
from tests.settings_factory import make_settings


class ReadyModel:
    def embed(self, texts: list[str]) -> EmbeddingBatch:
        return EmbeddingBatch(vectors=[[1.0, 0.0] for _ in texts], total_tokens=2)

    def complete(self, system: str, user: str) -> Completion:
        return Completion(text="Serviços em produção.", total_tokens=8)


def _client(tmp_path: Path) -> TestClient:
    settings = make_settings(tmp_path)
    path = tmp_path / "index.sqlite"
    index = Index(path)
    index.replace([Section("KingPanda", "Serviços em produção.")], [[1.0, 0.0]])
    service = ChatService(settings, index, Gate(path, settings), ReadyModel())
    return TestClient(create_app(service))


def test_missing_secret_is_unauthorized(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "KingPanda"}]},
    )

    assert response.status_code == 422


def test_wrong_secret_is_unauthorized(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/v1/chat",
        headers={"x-chat-secret": "nope"},
        json={"messages": [{"role": "user", "content": "KingPanda"}]},
    )

    assert response.status_code == 401


def test_chat_returns_the_answer(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/v1/chat",
        headers={"x-chat-secret": "secret", "x-chat-client-ip": "203.0.113.8"},
        json={"messages": [{"role": "user", "content": "O que você fez na KingPanda?"}]},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Serviços em produção."


def test_health_reports_ready(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is True
