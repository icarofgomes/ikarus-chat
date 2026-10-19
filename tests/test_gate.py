from datetime import UTC, datetime
from pathlib import Path

import pytest

from portfolio_chat.gate import Gate, GateError
from portfolio_chat.schemas import Message
from tests.settings_factory import make_settings


def test_rate_window_closes(tmp_path: Path) -> None:
    gate = Gate(tmp_path / "index.sqlite", make_settings(tmp_path))
    message = [Message(role="user", content="O que você fez na KingPanda?")]

    for _ in range(6):
        gate.admit("203.0.113.8", message)
        gate.record("203.0.113.8", 1)

    with pytest.raises(GateError) as caught:
        gate.admit("203.0.113.8", message)
    assert caught.value.status_code == 429


def test_daily_cap_survives_a_fresh_window(tmp_path: Path) -> None:
    clock = {"now": datetime(2026, 10, 16, 12, tzinfo=UTC).timestamp()}
    gate = Gate(tmp_path / "index.sqlite", make_settings(tmp_path), clock=lambda: clock["now"])
    message = [Message(role="user", content="E na SeguroBet?")]

    for _ in range(20):
        gate.admit("203.0.113.8", message)
        gate.record("203.0.113.8", 1)
        clock["now"] += 601

    with pytest.raises(GateError) as caught:
        gate.admit("203.0.113.8", message)
    assert caught.value.status_code == 429


def test_ninth_turn_is_refused(tmp_path: Path) -> None:
    gate = Gate(tmp_path / "index.sqlite", make_settings(tmp_path))
    messages = [Message(role="user", content=f"Pergunta {index}") for index in range(9)]

    with pytest.raises(GateError) as caught:
        gate.admit("203.0.113.8", messages)
    assert "WhatsApp" in caught.value.answer


def test_overlong_message_is_refused(tmp_path: Path) -> None:
    gate = Gate(tmp_path / "index.sqlite", make_settings(tmp_path))
    message = Message.model_construct(role="user", content="a" * 401)

    with pytest.raises(GateError) as caught:
        gate.admit("203.0.113.8", [message])
    assert caught.value.status_code == 400


def test_token_budget_is_checked_before_the_call(tmp_path: Path) -> None:
    gate = Gate(
        tmp_path / "index.sqlite",
        make_settings(tmp_path, daily_token_budget=100),
    )
    gate.record("203.0.113.8", 90)

    with pytest.raises(GateError):
        gate.reserve(20)
