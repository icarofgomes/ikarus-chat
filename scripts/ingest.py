from pathlib import Path

from portfolio_chat.corpus import load_corpus, split_sections
from portfolio_chat.index import Index
from portfolio_chat.providers import build_provider
from portfolio_chat.settings import Settings


def main() -> None:
    settings = Settings()
    corpus = Path(settings.corpus_path)
    try:
        markdown = load_corpus(corpus)
    except FileNotFoundError as error:
        raise SystemExit(
            f"{error}. O ingest usa private/curriculo.md e não grava o índice."
        ) from error
    if not settings.openai_api_key and not settings.openrouter_api_key:
        raise SystemExit("OPENAI_API_KEY or OPENROUTER_API_KEY is required")
    sections = split_sections(markdown)
    if not sections:
        raise SystemExit("corpus has no sections")
    embedded = build_provider(settings).embed([section.text() for section in sections])
    Index(Path(settings.index_path)).replace(sections, embedded.vectors)
    print(f"indexed {len(sections)} sections into {settings.index_path}")


if __name__ == "__main__":
    main()
