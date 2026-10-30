import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    title: str
    body: str

    def text(self) -> str:
        return f"{self.title}\n{self.body}"


def load_corpus(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"corpus não encontrado: {path}")
    return path.read_text(encoding="utf-8")


def split_sections(markdown: str) -> list[Section]:
    matches = list(HEADING.finditer(markdown))
    sections: list[Section] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        title = match.group(1).strip()
        body = markdown[start:end].strip()
        if title and body:
            sections.append(Section(title=title, body=body))
    return sections
