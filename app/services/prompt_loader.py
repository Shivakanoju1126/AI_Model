from __future__ import annotations

from pathlib import Path


class PromptLibrary:
    """Loads prompt sections from a markdown file for maintainability."""

    def __init__(self, markdown_path: str) -> None:
        self.markdown_path = Path(markdown_path)
        self.sections = self._parse_markdown(self.markdown_path)

    def get(self, section: str, fallback: str = "") -> str:
        return self.sections.get(section.strip().lower(), fallback)

    @staticmethod
    def _parse_markdown(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}

        sections: dict[str, list[str]] = {}
        current = "default"
        sections[current] = []

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                current = line.removeprefix("## ").strip().lower()
                sections.setdefault(current, [])
                continue
            sections.setdefault(current, []).append(line)

        return {key: "\n".join(value).strip() for key, value in sections.items()}
