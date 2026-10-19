from pathlib import Path

from portfolio_chat.corpus import Section
from portfolio_chat.index import Index


def test_search_respects_the_floor(tmp_path: Path) -> None:
    index = Index(tmp_path / "index.sqlite")
    index.replace(
        [Section("KingPanda", "Backend."), Section("SeguroBet", "Full stack.")],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    hits = Index(tmp_path / "index.sqlite").search([1.0, 0.0], top_k=4, floor=0.5)

    assert [hit.section.title for hit in hits] == ["KingPanda"]
    assert index.search([0.0, 1.0], top_k=4, floor=0.5)[0].section.title == "SeguroBet"
    assert index.search([0.2, 0.2], top_k=4, floor=0.99) == []
