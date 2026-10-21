import re
from dataclasses import dataclass

import tiktoken

from app.ingest.parse import ParsedBlock

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


@dataclass
class Sentence:
    page: int | None
    section: str | None
    text: str
    tokens: int


@dataclass
class Chunk:
    chunk_index: int
    page: int | None
    section: str | None
    text: str
    tokens: int


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _flatten_sentences(blocks: list[ParsedBlock]) -> list[Sentence]:
    sentences = []
    for block in blocks:
        for raw in _split_sentences(block.text):
            sentences.append(
                Sentence(page=block.page, section=block.section, text=raw, tokens=count_tokens(raw))
            )
    return sentences


def chunk_document(
    blocks: list[ParsedBlock],
    chunk_size_tokens: int = 450,
    chunk_overlap_tokens: int = 60,
) -> list[Chunk]:
    """Pack sentences into chunks, never splitting mid-sentence.

    A chunk boundary is forced when the running token count would exceed
    chunk_size_tokens, or when the heading (section) changes. Overlap only
    carries across a size boundary, never across a heading boundary, since
    repeating the tail of one section at the top of an unrelated one adds
    noise without helping retrieval.
    """
    sentences = _flatten_sentences(blocks)

    chunks: list[Chunk] = []
    current: list[Sentence] = []
    current_tokens = 0
    last_section: str | None = None
    section_seen = False

    def finalize() -> None:
        if not current:
            return
        text = " ".join(s.text for s in current)
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                page=current[0].page,
                section=current[0].section,
                text=text,
                tokens=sum(s.tokens for s in current),
            )
        )

    for sentence in sentences:
        section_changed = section_seen and sentence.section != last_section
        would_overflow = bool(current) and (current_tokens + sentence.tokens > chunk_size_tokens)

        if section_changed or would_overflow:
            overlap_carry: list[Sentence] = []
            if would_overflow and not section_changed:
                carry_tokens = 0
                for s in reversed(current):
                    if carry_tokens + s.tokens > chunk_overlap_tokens:
                        break
                    overlap_carry.insert(0, s)
                    carry_tokens += s.tokens

            finalize()
            current = list(overlap_carry)
            current_tokens = sum(s.tokens for s in current)

        current.append(sentence)
        current_tokens += sentence.tokens
        last_section = sentence.section
        section_seen = True

    finalize()
    return chunks
