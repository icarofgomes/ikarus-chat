import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from portfolio_chat.corpus import Section, load_corpus
from portfolio_chat.gate import Gate
from portfolio_chat.index import Hit, Index
from portfolio_chat.jobmatch import (
    JobMatchBudget,
    JobMatchBudgetError,
    JobMatchService,
    judge,
    merge_hits,
)
from portfolio_chat.main import create_app
from portfolio_chat.providers import Completion, EmbeddingBatch
from portfolio_chat.schemas import JobMatchRequest
from portfolio_chat.service import ChatService
from portfolio_chat.settings import Settings
from tests.settings_factory import make_settings

EXCERPT = "Serviços de RAG em produção na KingPanda."
ROOT = Path(__file__).resolve().parents[1]


class RubricModel:
    def __init__(self, payload: dict[str, object], *, vector: list[float] | None = None) -> None:
        self.payload = payload
        self.vector = vector or [1.0, 0.0]
        self.completed = 0
        self.embedded: list[str] = []
        self.prompt = ""

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        self.embedded = list(texts)
        return EmbeddingBatch(vectors=[self.vector for _ in texts], total_tokens=3)

    def complete(self, system: str, user: str) -> Completion:
        self.completed += 1
        self.prompt = user
        return Completion(text=json.dumps(self.payload), total_tokens=11)


def _axis(name: str, score: int, excerpt: str = EXCERPT) -> dict[str, object]:
    return {
        "name": name,
        "score": score,
        "requirement": name,
        "resume_excerpt": excerpt,
        "strength": "forte",
    }


def _rubric(*axes: dict[str, object]) -> dict[str, object]:
    return {"axes": list(axes), "gaps": ["Kubernetes"], "summary": "Cabe no backend."}


def _hits() -> list[Hit]:
    return [Hit(Section("KingPanda", EXCERPT), 0.9)]


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values = {
        "job_match_secret": "job-secret",
        "job_match_openrouter_api_key": "job-key",
        "job_match_score_floor": 0.35,
        "job_match_daily_token_budget": 100_000,
    }
    values.update(overrides)
    return make_settings(tmp_path, **values)


def _client(tmp_path: Path, model: RubricModel, **overrides: object) -> TestClient:
    settings = _settings(tmp_path, **overrides)
    path = tmp_path / "index.sqlite"
    index = Index(path)
    index.replace([Section("KingPanda", EXCERPT)], [[1.0, 0.0]])
    chat = ChatService(settings, index, Gate(path, settings), model)
    job_match = JobMatchService(settings, index, JobMatchBudget(path, settings), model)
    return TestClient(create_app(chat, job_match))


def _post(client: TestClient, secret: str = "job-secret") -> object:
    return client.post(
        "/v1/job-match",
        headers={"x-job-match-secret": secret},
        json={
            "url": "https://www.linkedin.com/posts/exemplo",
            "title": "Engenheiro de IA",
            "company": "Acme",
            "text": "Vaga de RAG em produção com Node.",
            "source": "search",
        },
    )


def test_missing_secret_is_unauthorized(tmp_path: Path) -> None:
    response = _client(tmp_path, RubricModel(_rubric(_axis("requisitos", 100)))).post(
        "/v1/job-match",
        json={
            "url": "https://www.linkedin.com/posts/exemplo",
            "title": "Engenheiro de IA",
            "text": "Vaga de RAG em produção.",
        },
    )

    assert response.status_code == 401


def test_wrong_secret_is_unauthorized(tmp_path: Path) -> None:
    response = _post(_client(tmp_path, RubricModel(_rubric())), secret="nope")

    assert response.status_code == 401


def test_unconfigured_route_stays_unavailable(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    path = tmp_path / "index.sqlite"
    index = Index(path)
    index.replace([Section("KingPanda", EXCERPT)], [[1.0, 0.0]])
    chat = ChatService(settings, index, Gate(path, settings), RubricModel(_rubric()))
    response = TestClient(create_app(chat)).post(
        "/v1/job-match",
        headers={"x-job-match-secret": "job-secret"},
        json={
            "url": "https://www.linkedin.com/posts/exemplo",
            "title": "Engenheiro de IA",
            "text": "Vaga de RAG em produção.",
        },
    )

    assert response.status_code == 503


def test_route_returns_aplicar_when_the_quote_is_in_the_resume(tmp_path: Path) -> None:
    payload = _rubric(
        _axis("requisitos", 100),
        _axis("stack", 100),
        _axis("senioridade", 100),
        _axis("dominio", 100),
    )
    response = _post(_client(tmp_path, RubricModel(payload)))

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "aplicar"
    assert body["match_percent"] == 100
    assert body["evidence"][0]["resume_excerpt"] == EXCERPT
    assert [axis["name"] for axis in body["axes"]] == [
        "requisitos",
        "stack",
        "senioridade",
        "dominio",
    ]
    assert sum(axis["points"] for axis in body["axes"]) == 100
    assert body["axes"][0]["accepted"] is True


def test_fake_excerpt_does_not_score(tmp_path: Path) -> None:
    payload = _rubric(_axis("requisitos", 100, "trabalhei na NASA por uma década inteira"))
    judged = judge(payload, _hits())

    assert judged.match_percent == 0
    assert judged.decision == "pular"
    assert judged.evidence == []
    assert judged.axes[0].accepted is False


def test_thresholds_split_aplicar_and_talvez() -> None:
    aplicar = judge(
        _rubric(_axis("requisitos", 100), _axis("stack", 100), _axis("senioridade", 50)),
        _hits(),
    )
    talvez = judge(_rubric(_axis("requisitos", 100), _axis("stack", 100)), _hits())

    assert aplicar.match_percent == 75
    assert aplicar.decision == "aplicar"
    assert talvez.match_percent == 65
    assert talvez.decision == "talvez"


def test_no_hits_skip_the_model(tmp_path: Path) -> None:
    model = RubricModel(_rubric(_axis("requisitos", 100)), vector=[0.0, 1.0])
    response = _post(_client(tmp_path, model))

    assert response.status_code == 200
    assert response.json()["decision"] == "pular"
    assert model.completed == 0


def test_daily_budget_blocks_the_call(tmp_path: Path) -> None:
    model = RubricModel(_rubric(_axis("requisitos", 100)))
    client = _client(tmp_path, model, job_match_daily_token_budget=10)

    response = _post(client)

    assert response.status_code == 429
    assert model.completed == 0


def test_budget_table_is_separate_from_the_chat(tmp_path: Path) -> None:
    settings = _settings(tmp_path, job_match_daily_token_budget=10)
    budget = JobMatchBudget(tmp_path / "index.sqlite", settings)
    with pytest.raises(JobMatchBudgetError):
        budget.reserve(11)


def test_corpus_path_defaults_to_the_private_file() -> None:
    assert Settings(_env_file=None).corpus_path == "private/curriculo.md"


def test_load_corpus_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "private" / "curriculo.md")


def test_ingest_stops_before_writing_the_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("ingest", ROOT / "scripts" / "ingest.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = tmp_path / "private" / "curriculo.md"
    monkeypatch.setattr(module, "Settings", lambda: _settings(tmp_path, corpus_path=str(missing)))

    with pytest.raises(SystemExit) as caught:
        module.main()

    assert "não grava o índice" in str(caught.value)
    assert not (tmp_path / "index.sqlite").exists()


def test_text_over_the_route_limit_is_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path, RubricModel(_rubric()), job_match_max_text_chars=20)
    response = client.post(
        "/v1/job-match",
        headers={"x-job-match-secret": "job-secret"},
        json={
            "url": "https://www.linkedin.com/posts/exemplo",
            "title": "Engenheiro de IA",
            "text": "Esta vaga passou do limite da rota.",
        },
    )

    assert response.status_code == 400


def test_request_accepts_a_long_job_text() -> None:
    job = JobMatchRequest(
        url="https://www.linkedin.com/posts/exemplo",
        title="Engenheiro",
        text="vaga " * 500,
    )

    assert len(job.text) > 400
    assert job.required == []
    assert job.also_accepts == []
    assert job.differentials == []
    assert job.exclusive is False


def test_term_lists_drop_blanks_and_keep_a_short_prefix() -> None:
    job = JobMatchRequest(
        url="https://example.com/vaga",
        title="Engenheiro",
        text="Vaga de backend com Node.",
        required=["  TypeScript  ", " ", "x" * 120, *[f"s{index}" for index in range(30)]],
    )

    assert job.required[0] == "TypeScript"
    assert len(job.required[1]) == 80
    assert len(job.required) == 24


def test_a_second_excerpt_keeps_the_axis_when_the_first_is_fake() -> None:
    requisitos = _axis("requisitos", 100, "trabalhei na NASA por uma década inteira")
    requisitos["excerpts"] = ["trabalhei na NASA por uma década inteira", EXCERPT]
    judged = judge(_rubric(requisitos), _hits())

    assert judged.match_percent == 40
    assert judged.evidence[0].resume_excerpt == EXCERPT
    assert judged.axes[0].accepted is True
    assert judged.axes[0].excerpts == [EXCERPT]


def test_inapplicable_domain_renormalizes_the_other_axes() -> None:
    dominio = _axis("dominio", 100)
    dominio["applicable"] = False
    judged = judge(
        _rubric(
            _axis("requisitos", 100),
            _axis("stack", 100),
            _axis("senioridade", 100),
            dominio,
        ),
        _hits(),
    )

    assert judged.match_percent == 100
    assert judged.decision == "aplicar"
    found = {axis.name: axis for axis in judged.axes}
    assert found["dominio"].applicable is False
    assert found["dominio"].points == 0
    assert found["requisitos"].points + found["stack"].points + found["senioridade"].points == 100
    assert [item.requirement for item in judged.evidence] == ["requisitos", "stack", "senioridade"]


def test_merge_keeps_half_the_slots_for_each_query() -> None:
    first = [
        Hit(Section("Stack", "node"), 0.9),
        Hit(Section("Quem", "tipo"), 0.8),
        Hit(Section("A", "a"), 0.7),
        Hit(Section("B", "b"), 0.6),
        Hit(Section("C", "c"), 0.5),
    ]
    second = [
        Hit(Section("Stack", "node melhor"), 0.95),
        Hit(Section("Escola", "aula"), 0.4),
        Hit(Section("Projeto", "professor"), 0.39),
        Hit(Section("Monitoria", "pais"), 0.38),
        Hit(Section("Formacao", "economia"), 0.37),
    ]

    merged = merge_hits([first, second], 8)

    titles = {hit.section.title for hit in merged}
    stack = next(hit for hit in merged if hit.section.title == "Stack")
    assert len(merged) == 8
    assert {"Escola", "Projeto", "Monitoria", "Formacao"} <= titles
    assert stack.score == 0.95
    assert stack.section.body == "node melhor"


def test_embed_drops_chrome_and_hashtags(tmp_path: Path) -> None:
    model = RubricModel(_rubric(_axis("requisitos", 100)))
    client = _client(tmp_path, model)
    response = client.post(
        "/v1/job-match",
        headers={"x-job-match-secret": "job-secret"},
        json={
            "url": "https://www.linkedin.com/posts/exemplo",
            "title": "Engenheiro de IA",
            "company": "Acme",
            "text": (
                "Publicação no feed\n"
                "Seguir o time no Git.\n"
                "Vaga de RAG em produção com Node.\n"
                "Seguir\n"
                "há 6 d\n"
                "#Node #React #Docker\n"
                "Exibir tradução"
            ),
            "required": ["Node"],
            "also_accepts": [".NET"],
            "differentials": ["Next.js"],
            "exclusive": False,
        },
    )

    assert response.status_code == 200
    assert model.embedded == [
        "Engenheiro de IA\nNode, .NET, Next.js",
        "Seguir o time no Git.\nVaga de RAG em produção com Node.",
    ]
    assert "required: Node" in model.prompt
    assert "also_accepts: .NET" in model.prompt
    assert "differentials: Next.js" in model.prompt
    assert "exclusive: false" in model.prompt
    assert "Publicação no feed" not in model.prompt
    assert "#Node" not in model.prompt
    assert "há 6 d" not in model.prompt
