import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from portfolio_chat.prompt import paused_answer, too_long_answer
from portfolio_chat.schemas import Message
from portfolio_chat.settings import Settings


class GateError(Exception):
    def __init__(self, answer: str, status_code: int) -> None:
        self.answer = answer
        self.status_code = status_code
        super().__init__(answer)


class Gate:
    def __init__(
        self,
        path: Path,
        settings: Settings,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or _unix_now
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, timeout=5, check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                ip TEXT NOT NULL,
                at_unix INTEGER NOT NULL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_tokens (
                day TEXT PRIMARY KEY,
                tokens INTEGER NOT NULL
            )
            """
        )
        self._db.commit()

    def admit(self, client_ip: str, messages: list[Message]) -> None:
        if messages[-1].role != "user":
            raise GateError(too_long_answer(), 400)
        for message in messages:
            if len(message.content) > self._settings.max_message_chars:
                raise GateError(too_long_answer(), 400)
        user_turns = sum(1 for message in messages if message.role == "user")
        if user_turns > self._settings.max_turns or len(messages) > self._settings.max_messages:
            raise GateError(paused_answer(self._settings.whatsapp_url), 429)
        now = int(self._clock())
        paused = paused_answer(self._settings.whatsapp_url)
        with self._lock:
            window_start = now - self._settings.rate_window_seconds
            window_count = self._count(
                "SELECT COUNT(*) FROM requests WHERE ip = ? AND at_unix > ?",
                (client_ip, window_start),
            )
            daily_count = self._count(
                "SELECT COUNT(*) FROM requests WHERE ip = ? AND at_unix >= ?",
                (client_ip, _day_start(now)),
            )
            spent = self._count(
                "SELECT tokens FROM daily_tokens WHERE day = ?",
                (_day(now),),
            )
            if window_count >= self._settings.rate_window_max:
                raise GateError(paused, 429)
            if daily_count >= self._settings.daily_max:
                raise GateError(paused, 429)
            if spent >= self._settings.daily_token_budget:
                raise GateError(paused, 429)

    def reserve(self, estimated_tokens: int) -> None:
        now = int(self._clock())
        with self._lock:
            spent = self._count(
                "SELECT tokens FROM daily_tokens WHERE day = ?",
                (_day(now),),
            )
        if spent + estimated_tokens > self._settings.daily_token_budget:
            raise GateError(paused_answer(self._settings.whatsapp_url), 429)

    def record(self, client_ip: str, tokens: int) -> None:
        now = int(self._clock())
        with self._lock:
            self._db.execute(
                "INSERT INTO requests (ip, at_unix) VALUES (?, ?)",
                (client_ip, now),
            )
            self._db.execute(
                """
                INSERT INTO daily_tokens (day, tokens) VALUES (?, ?)
                ON CONFLICT(day) DO UPDATE SET tokens = tokens + excluded.tokens
                """,
                (_day(now), tokens),
            )
            self._db.commit()

    def _count(self, sql: str, params: tuple[object, ...]) -> int:
        row = self._db.execute(sql, params).fetchone()
        if row is None or row[0] is None:
            return 0
        return int(row[0])


def _unix_now() -> float:
    return datetime.now(UTC).timestamp()


def _day(unix: int) -> str:
    return datetime.fromtimestamp(unix, UTC).date().isoformat()


def _day_start(unix: int) -> int:
    moment = datetime.fromtimestamp(unix, UTC)
    start = datetime(moment.year, moment.month, moment.day, tzinfo=UTC)
    return int(start.timestamp())
