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
            CREATE TABLE IF NOT EXISTS token_spend (
                at_unix INTEGER NOT NULL,
                tokens INTEGER NOT NULL
            )
            """
        )
        self._db.commit()

    def admit(self, client_ip: str, messages: list[Message]) -> None:
        if messages[-1].role != "user":
            raise GateError(too_long_answer(), 400)
        for message in messages:
            limit = (
                self._settings.max_message_chars
                if message.role == "user"
                else self._settings.max_reply_chars
            )
            if len(message.content) > limit:
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
            if window_count >= self._settings.rate_window_max:
                raise GateError(paused, 429)
            if daily_count >= self._settings.daily_max:
                raise GateError(paused, 429)
            if self._over_budget(now, 0):
                raise GateError(paused, 429)

    def reserve(self, estimated_tokens: int) -> None:
        """Count the call before the model runs, so parallel requests share one budget."""
        now = int(self._clock())
        with self._lock:
            if self._over_budget(now, estimated_tokens):
                raise GateError(paused_answer(self._settings.whatsapp_url), 429)
            self._add_tokens(now, estimated_tokens)
            self._db.commit()

    def record(self, client_ip: str, tokens: int, *, reserved: int = 0) -> None:
        """Replace a reservation with the real spend and count the request."""
        now = int(self._clock())
        with self._lock:
            self._add_tokens(now, tokens - reserved)
            self._db.execute(
                "INSERT INTO requests (ip, at_unix) VALUES (?, ?)",
                (client_ip, now),
            )
            self._db.execute(
                "DELETE FROM token_spend WHERE at_unix < ?",
                (_day_start(now) - 3_600,),
            )
            self._db.commit()

    def _over_budget(self, now: int, extra: int) -> bool:
        minute = self._spent_since(now - 60)
        hour = self._spent_since(now - 3_600)
        day = self._spent_since(_day_start(now))
        if extra:
            return (
                minute + extra > self._settings.token_minute_budget
                or hour + extra > self._settings.token_hour_budget
                or day + extra > self._settings.daily_token_budget
            )
        return (
            minute >= self._settings.token_minute_budget
            or hour >= self._settings.token_hour_budget
            or day >= self._settings.daily_token_budget
        )

    def _spent_since(self, start: int) -> int:
        return self._count(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_spend WHERE at_unix >= ?",
            (start,),
        )

    def _add_tokens(self, now: int, tokens: int) -> None:
        if tokens == 0:
            return
        self._db.execute(
            "INSERT INTO token_spend (at_unix, tokens) VALUES (?, ?)",
            (now, tokens),
        )

    def _count(self, sql: str, params: tuple[object, ...]) -> int:
        row = self._db.execute(sql, params).fetchone()
        if row is None or row[0] is None:
            return 0
        return int(row[0])


def _unix_now() -> float:
    return datetime.now(UTC).timestamp()


def _day_start(unix: int) -> int:
    moment = datetime.fromtimestamp(unix, UTC)
    start = datetime(moment.year, moment.month, moment.day, tzinfo=UTC)
    return int(start.timestamp())
