import hmac
import logging
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request

from portfolio_chat.gate import Gate, GateError
from portfolio_chat.index import Index
from portfolio_chat.providers import ModelError, build_provider
from portfolio_chat.prompt import unavailable_answer
from portfolio_chat.schemas import ChatRequest, ChatResponse
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


def create_app(service: ChatService | None = None) -> FastAPI:
    if service is None:
        settings = Settings()
        if not settings.chat_shared_secret:
            raise RuntimeError("CHAT_SHARED_SECRET is required")
        service = build_service(settings)
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

    return app


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-chat-client-ip", "").strip()
    if forwarded:
        return forwarded[:64]
    if request.client is None:
        return "unknown"
    return request.client.host
