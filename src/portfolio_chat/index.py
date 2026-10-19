import math
import sqlite3
import struct
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from portfolio_chat.corpus import Section


@dataclass(frozen=True)
class Hit:
    section: Section
    score: float


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])


def unpack_vector(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


class Index:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, timeout=5, check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        self._db.commit()

    def count(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()
        if row is None:
            return 0
        return int(row[0])

    def replace(self, sections: Sequence[Section], vectors: Sequence[Sequence[float]]) -> None:
        if len(sections) != len(vectors):
            raise ValueError("each section needs one vector")
        with self._lock:
            self._db.execute("DELETE FROM chunks")
            self._db.executemany(
                "INSERT INTO chunks (title, body, embedding) VALUES (?, ?, ?)",
                [
                    (section.title, section.body, pack_vector(vector))
                    for section, vector in zip(sections, vectors, strict=True)
                ],
            )
            self._db.commit()

    def search(self, vector: Sequence[float], top_k: int, floor: float) -> list[Hit]:
        rows = self._db.execute("SELECT title, body, embedding FROM chunks").fetchall()
        hits: list[Hit] = []
        for title, body, blob in rows:
            stored = unpack_vector(blob)
            score = cosine(vector, stored)
            if score >= floor:
                hits.append(Hit(Section(title=str(title), body=str(body)), score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
