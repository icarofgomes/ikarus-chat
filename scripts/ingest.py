from pathlib import Path

from portfolio_chat.corpus import split_sections
from portfolio_chat.index import Index
from portfolio_chat.openrouter import OpenRouter
from portfolio_chat.settings import Settings


def main() -> None:
    settings = Settings()
    if not settings.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    markdown = Path(settings.corpus_path).read_text(encoding="utf-8")
    sections = split_sections(markdown)
    if not sections:
        raise SystemExit("corpus has no sections")
    embedded = OpenRouter(settings).embed([section.text() for section in sections])
    Index(Path(settings.index_path)).replace(sections, embedded.vectors)
    print(f"indexed {len(sections)} sections into {settings.index_path}")


if __name__ == "__main__":
    main()
