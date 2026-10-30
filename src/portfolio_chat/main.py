import hmac
import logging
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request

from portfolio_chat.gate import Gate, GateError
from portfolio_chat.index import Index
from portfolio_chat.jobmatch import (
    JobMatchBudget,
    JobMatchBudgetError,
    JobMatchInputError,
    JobMatchModelError,
    JobMatchService,
)
from portfolio_chat.providers import ModelError, build_job_match_provider, build_provider
from portfolio_chat.prompt import unavailable_answer
from portfolio_chat.schemas import ChatRequest, ChatResponse, JobMatchRequest, JobMatchResponse
from portfolio_chat.service import ChatService, NotReady
from portfolio_chat.settings import Settings

logger = logging.getLogger("portfolio_chat")
NOT_READY = "O chat ainda não está pronto."


def build_service(settings: Settings) -> ChatService:
    path = Path(settings.index_path)
    return ChatService(
        settings,
        Index(path),
        Gate(path, settings),
        build_provider(settings),
    )


def build_job_match(settings: Settings) -> JobMatchService | None:
    if not settings.job_match_secret or not settings.job_match_openrouter_api_key:
        return None
    path = Path(settings.index_path)
    return JobMatchService(
        settings,
        Index(path),
        JobMatchBudget(path, settings),
        build_job_match_provider(settings),
    )


def create_app(
    service: ChatService | None = None,
    job_match: JobMatchService | None = None,
) -> FastAPI:
    if service is None:
        settings = Settings()
        if not settings.chat_shared_secret:
            raise RuntimeError("CHAT_SHARED_SECRET is required")
        service = build_service(settings)
        job_match = build_job_match(settings)
    elif not service.settings.chat_shared_secret:
        raise RuntimeError("CHAT_SHARED_SECRET is required")

    app = FastAPI(title="Portfolio chat")

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {"status": "ok", "ready": service.ready()}

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(
        body: ChatRequest,
        request: Request,
        x_chat_secret: str = Header(),
    ) -> ChatResponse:
        if not hmac.compare_digest(x_chat_secret, service.settings.chat_shared_secret):
            raise HTTPException(status_code=401, detail="unauthorized")
        try:
            answer = service.answer(body.messages, _client_ip(request))
        except GateError as error:
            logger.info("chat rejected status=%s", error.status_code)
            raise HTTPException(
                status_code=error.status_code,
                detail={"answer": error.answer},
            ) from error
        except NotReady as error:
            logger.info("chat rejected status=503")
            raise HTTPException(status_code=503, detail={"answer": NOT_READY}) from error
        except ModelError as error:
            logger.info("chat rejected status=502")
            raise HTTPException(
                status_code=502,
                detail={"answer": unavailable_answer(service.settings.whatsapp_url)},
            ) from error
        logger.info("chat result=ok")
        return ChatResponse(answer=answer)

    @app.post("/v1/job-match", response_model=JobMatchResponse)
    def match_job(
        body: JobMatchRequest,
        x_job_match_secret: str = Header(default=""),
    ) -> JobMatchResponse:
        if job_match is None:
            raise HTTPException(status_code=503, detail="job-match não configurado")
        if not hmac.compare_digest(x_job_match_secret, job_match.settings.job_match_secret):
            raise HTTPException(status_code=401, detail="unauthorized")
        try:
            result = job_match.match(body)
        except JobMatchInputError as error:
            raise HTTPException(status_code=400, detail="texto longo demais") from error
        except JobMatchBudgetError as error:
            raise HTTPException(status_code=429, detail="orçamento do job-match esgotado") from error
        except (ModelError, JobMatchModelError) as error:
            logger.info("job-match rejected status=502")
            raise HTTPException(status_code=502, detail="não foi possível julgar a vaga") from error
        logger.info("job-match result=%s percent=%s", result.decision, result.match_percent)
        return result

    return app


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-chat-client-ip", "").strip()
    if forwarded:
        return forwarded[:64]
    if request.client is None:
        return "unknown"
    return request.client.host
