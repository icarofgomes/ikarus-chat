from pathlib import Path

from portfolio_chat.corpus import split_sections


def test_splits_on_headings() -> None:
    sections = split_sections(
        """
# Ignorado

## KingPanda

Serviços em produção.

## SeguroBet

Único desenvolvedor.
"""
    )

    assert [section.title for section in sections] == ["KingPanda", "SeguroBet"]
    assert sections[0].body == "Serviços em produção."


def test_public_corpus_keeps_the_jobs() -> None:
    markdown = Path("corpus/curriculo.md").read_text(encoding="utf-8")
    titles = [section.title for section in split_sections(markdown)]

    assert "KingPanda" in titles
    assert "SeguroBet" in titles
    assert "Case Juliana Camioto" in titles
