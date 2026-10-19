from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=400)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("empty message")
        return cleaned


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=16)


class ChatResponse(BaseModel):
    answer: str
