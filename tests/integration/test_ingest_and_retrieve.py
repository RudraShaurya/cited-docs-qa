from app.ingest.pipeline import ingest_file
from app.retrieval.embed import backfill_missing_embeddings
from app.retrieval.hybrid import retrieve
from tests.integration.fixtures import make_pdf


def test_reingesting_same_file_is_a_noop(clean_db, tmp_path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Long service leave accrues after ten years."])

    first = ingest_file(pdf_path, title="Doc")
    second = ingest_file(pdf_path, title="Doc")

    assert first["status"] == "ingested"
    assert first["chunks"] == 1
    assert second["status"] == "skipped"

    with clean_db.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 1


def test_vector_search_finds_the_right_document(clean_db, tmp_path):
    cat_pdf = make_pdf(tmp_path / "cats.pdf", ["Cats are small domesticated carnivorous mammals."])
    leave_pdf = make_pdf(
        tmp_path / "leave.pdf",
        ["An employee accrues long service leave after a continuous period of service."],
    )

    ingest_file(cat_pdf, title="About Cats")
    ingest_file(leave_pdf, title="Leave Policy")
    backfill_missing_embeddings()

    results = retrieve("How much long service leave do I get?", k=3, mode="vector")

    assert results
    assert results[0]["title"] == "Leave Policy"


def test_keyword_search_finds_exact_term(clean_db, tmp_path):
    pdf_a = make_pdf(tmp_path / "a.pdf", ["General workplace safety guidance for employers."])
    pdf_b = make_pdf(tmp_path / "b.pdf", ["The NES sets minimum entitlements for all employees."])

    ingest_file(pdf_a, title="Safety")
    ingest_file(pdf_b, title="NES Guide")
    backfill_missing_embeddings()

    results = retrieve("NES", k=3, mode="keyword")

    assert results
    assert results[0]["title"] == "NES Guide"


def test_hybrid_mode_returns_fused_results(clean_db, tmp_path):
    pdf_a = make_pdf(tmp_path / "a.pdf", ["General workplace safety guidance for employers."])
    pdf_b = make_pdf(tmp_path / "b.pdf", ["The NES sets minimum entitlements for all employees."])

    ingest_file(pdf_a, title="Safety")
    ingest_file(pdf_b, title="NES Guide")
    backfill_missing_embeddings()

    results = retrieve("NES entitlements", k=3, mode="hybrid")

    assert results
    assert any(r["title"] == "NES Guide" for r in results)
    assert "rrf_score" in results[0]
