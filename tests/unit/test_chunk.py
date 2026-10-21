from app.ingest.chunk import chunk_document, count_tokens
from app.ingest.parse import ParsedBlock


def make_block(text: str, page: int | None = 1, section: str | None = None) -> ParsedBlock:
    return ParsedBlock(page=page, section=section, text=text)


def test_single_small_block_is_one_chunk():
    block = make_block("This is one sentence. This is another sentence.", page=3)
    chunks = chunk_document([block], chunk_size_tokens=450, chunk_overlap_tokens=60)

    assert len(chunks) == 1
    assert chunks[0].page == 3
    assert "This is one sentence." in chunks[0].text
    assert "This is another sentence." in chunks[0].text


def test_never_splits_mid_sentence():
    # Build a block whose sentences are individually short but together exceed
    # a tiny chunk_size_tokens, forcing a split, and check every chunk's text
    # consists of whole sentences only (ends with terminal punctuation).
    sentences = [f"This is sentence number {i} in the document." for i in range(20)]
    block = make_block(" ".join(sentences), page=1)

    chunks = chunk_document([block], chunk_size_tokens=30, chunk_overlap_tokens=5)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.strip().endswith(".")
        # Every piece between ". " separators must be a complete, known
        # sentence (minus its trailing period) - never a partial one.
        pieces = [p.strip() for p in chunk.text.rstrip(".").split(". ")]
        known_sentences_no_period = {s.rstrip(".") for s in sentences}
        for piece in pieces:
            assert piece in known_sentences_no_period


def test_overlap_carries_trailing_sentences_into_next_chunk():
    sentences = [f"Sentence {i} has some filler words to pad it out a bit." for i in range(12)]
    block = make_block(" ".join(sentences), page=2)

    chunks = chunk_document([block], chunk_size_tokens=40, chunk_overlap_tokens=15)

    assert len(chunks) >= 2
    # The tail of chunk N should reappear at the head of chunk N+1.
    first_chunk_sentences = chunks[0].text.split(". ")
    last_sentence_of_first = first_chunk_sentences[-1].strip().rstrip(".")
    assert last_sentence_of_first in chunks[1].text


def test_page_and_section_survive_chunking():
    block_a = make_block("Content from page one.", page=1, section="Intro")
    block_b = make_block("Content from page two.", page=2, section="Intro")

    chunks = chunk_document([block_a, block_b], chunk_size_tokens=450, chunk_overlap_tokens=0)

    assert chunks[0].page == 1
    assert chunks[0].section == "Intro"


def test_heading_change_forces_new_chunk_without_crossing_overlap():
    block_a = make_block("Text under heading A that is reasonably long for a sentence.",
                          page=1, section="A")
    block_b = make_block("Text under heading B that is reasonably long for a sentence.",
                          page=2, section="B")

    chunks = chunk_document([block_a, block_b], chunk_size_tokens=450, chunk_overlap_tokens=60)

    assert len(chunks) == 2
    assert chunks[0].section == "A"
    assert chunks[1].section == "B"
    # No bleed-through from A's text into B's chunk.
    assert "heading A" not in chunks[1].text


def test_count_tokens_is_positive_and_monotonic_ish():
    short = count_tokens("Hello world.")
    longer = count_tokens("Hello world. This sentence has quite a few more words in it.")
    assert short > 0
    assert longer > short


def test_empty_blocks_produce_no_chunks():
    assert chunk_document([], chunk_size_tokens=450, chunk_overlap_tokens=60) == []
    assert chunk_document([make_block("")], chunk_size_tokens=450, chunk_overlap_tokens=60) == []
