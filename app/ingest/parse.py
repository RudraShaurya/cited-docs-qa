import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
from docx import Document as DocxDocument

HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
HEADING_STYLE_RE = re.compile(r"^Heading\s*\d*$", re.IGNORECASE)


@dataclass
class ParsedBlock:
    page: int | None
    section: str | None
    text: str


def _dehyphenate(text: str) -> str:
    return HYPHEN_BREAK_RE.sub(r"\1\2", text)


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Drop lines that repeat verbatim across most pages (headers/footers)."""
    if len(pages) < 3:
        return pages

    line_counts: dict[str, int] = {}
    per_page_lines = [p.splitlines() for p in pages]
    for lines in per_page_lines:
        for line in {line.strip() for line in lines if line.strip()}:
            line_counts[line] = line_counts.get(line, 0) + 1

    threshold = max(3, int(len(pages) * 0.6))
    repeated = {line for line, count in line_counts.items() if count >= threshold}

    cleaned_pages = []
    for lines in per_page_lines:
        kept = [line for line in lines if line.strip() not in repeated]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


def parse_pdf(path: Path) -> list[ParsedBlock]:
    doc = fitz.open(path)
    try:
        raw_pages = [page.get_text() for page in doc]
    finally:
        doc.close()

    cleaned_pages = _strip_repeated_lines(raw_pages)

    blocks = []
    for page_number, text in enumerate(cleaned_pages, start=1):
        text = _dehyphenate(text).strip()
        if text:
            blocks.append(ParsedBlock(page=page_number, section=None, text=text))
    return blocks


def parse_docx(path: Path) -> list[ParsedBlock]:
    doc = DocxDocument(path)

    blocks: list[ParsedBlock] = []
    current_section: str | None = None
    buffer: list[str] = []

    def flush():
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(ParsedBlock(page=None, section=current_section, text=text))
            buffer.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if HEADING_STYLE_RE.match(para.style.name or ""):
            flush()
            current_section = text
            continue
        buffer.append(text)

    flush()
    return blocks


def parse_document(path: Path) -> list[ParsedBlock]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}")
