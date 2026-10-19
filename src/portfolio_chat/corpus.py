import re
from dataclasses import dataclass

HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    title: str
    body: str

    def text(self) -> str:
        return f"{self.title}\n{self.body}"


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
