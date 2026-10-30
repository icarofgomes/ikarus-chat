from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

USER_MAX_CHARS = 400
REPLY_MAX_CHARS = 2_000


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("empty message")
        return cleaned

    @model_validator(mode="after")
    def limit_length(self) -> "Message":
        limit = USER_MAX_CHARS if self.role == "user" else REPLY_MAX_CHARS
        if len(self.content) > limit:
            raise ValueError("message is too long")
        return self


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=16)


class ChatResponse(BaseModel):
    answer: str


class JobMatchRequest(BaseModel):
    url: str
    title: str
    company: str = ""
    text: str
    source: Literal["search", "vagas"] = "search"

    @field_validator("url", "title", "text", "company")
    @classmethod
    def strip_field(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def require_job_text(self) -> "JobMatchRequest":
        if not self.url or not self.title or not self.text:
            raise ValueError("url, title and text are required")
        return self


class Evidence(BaseModel):
    requirement: str
    resume_excerpt: str
    strength: Literal["forte", "media", "fraca"]


class JobMatchResponse(BaseModel):
    match_percent: int
    decision: Literal["aplicar", "talvez", "pular"]
    evidence: list[Evidence]
    gaps: list[str]
    summary: str
