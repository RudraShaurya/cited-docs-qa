import app.answer as answer_module
from app.answer import REFUSAL_MESSAGE, parse_and_validate_citations, should_refuse_before_generation
from app.llm.base import FakeLLM


def make_chunk(chunk_id=1, document_id="doc-1", title="Fact Sheet", page=2,
                section=None, text="Some chunk text.", **extra):
    chunk = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "title": title,
        "page": page,
        "section": section,
        "text": text,
    }
    chunk.update(extra)
    return chunk


def test_citation_parsing_extracts_valid_markers():
    chunks = [make_chunk(chunk_id=1, text="Chunk one text."), make_chunk(chunk_id=2, text="Chunk two text.")]
    answer = "Something is true [S1]. Something else is also true [S2]."

    citations, refused = parse_and_validate_citations(answer, chunks)

    assert refused is False
    assert len(citations) == 2
    assert citations[0].marker == 1
    assert citations[1].marker == 2


def test_citation_parsing_drops_out_of_range_markers():
    chunks = [make_chunk(chunk_id=1)]
    answer = "This cites a source that doesn't exist [S5]."

    citations, refused = parse_and_validate_citations(answer, chunks)

    assert citations == []
    assert refused is True


def test_citation_parsing_dedupes_repeated_markers():
    chunks = [make_chunk(chunk_id=1)]
    answer = "First mention [S1]. Second mention of the same source [S1]."

    citations, refused = parse_and_validate_citations(answer, chunks)

    assert len(citations) == 1
    assert refused is False


def test_answer_with_no_citations_is_treated_as_refusal():
    chunks = [make_chunk(chunk_id=1)]
    answer = "This is a confident-sounding answer with no citation markers at all."

    citations, refused = parse_and_validate_citations(answer, chunks)

    assert citations == []
    assert refused is True


def test_literal_refusal_message_passes_through_as_refusal():
    chunks = [make_chunk(chunk_id=1)]

    citations, refused = parse_and_validate_citations(REFUSAL_MESSAGE, chunks)

    assert citations == []
    assert refused is True


def test_refuse_before_generation_on_empty_retrieval():
    assert should_refuse_before_generation([]) is True


def test_refuse_before_generation_on_low_similarity_and_no_keyword_hit():
    chunks = [make_chunk(similarity=0.1)]
    assert should_refuse_before_generation(chunks) is True


def test_no_refusal_when_similarity_is_high():
    chunks = [make_chunk(similarity=0.8)]
    assert should_refuse_before_generation(chunks) is False


def test_no_refusal_when_similarity_low_but_keyword_hit_present():
    chunks = [make_chunk(similarity=0.1, rank=0.9)]
    assert should_refuse_before_generation(chunks) is False


def test_no_refusal_in_keyword_only_mode():
    # keyword-only results carry no "similarity" key at all
    chunks = [make_chunk(rank=0.3)]
    assert should_refuse_before_generation(chunks) is False


def test_full_answer_flow_with_strong_retrieval(monkeypatch):
    chunks = [make_chunk(chunk_id=1, similarity=0.9, text="The answer is 42.")]
    monkeypatch.setattr(answer_module, "retrieve", lambda question, k, mode: chunks)

    llm = FakeLLM(response="The answer is 42 [S1].")
    result = answer_module.answer_question("What is the answer?", llm, mode="hybrid")

    assert result.refused is False
    assert result.answer == "The answer is 42 [S1]."
    assert len(result.citations) == 1
    assert result.citations[0].document_id == "doc-1"
    assert "retrieve" in result.timings_ms
    assert "generate" in result.timings_ms


def test_full_answer_flow_refuses_on_weak_retrieval(monkeypatch):
    chunks = [make_chunk(chunk_id=1, similarity=0.05)]
    monkeypatch.setattr(answer_module, "retrieve", lambda question, k, mode: chunks)

    llm = FakeLLM()
    result = answer_module.answer_question("Unanswerable question?", llm, mode="hybrid")

    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    # The LLM should never have been called when retrieval is this weak.
    assert llm.last_user is None


def test_full_answer_flow_refuses_when_llm_gives_uncited_answer(monkeypatch):
    chunks = [make_chunk(chunk_id=1, similarity=0.9, text="Some strong chunk.")]
    monkeypatch.setattr(answer_module, "retrieve", lambda question, k, mode: chunks)

    llm = FakeLLM(response="A confident answer with no citation.")
    result = answer_module.answer_question("A question?", llm, mode="hybrid")

    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
