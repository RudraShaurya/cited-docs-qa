import io

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from app.llm.base import FakeLLM
from tests.integration.fixtures import make_pdf


@pytest.fixture
def client(clean_db, monkeypatch):
    monkeypatch.setattr(
        api_module,
        "get_llm_client",
        lambda: FakeLLM(response="Long service leave accrues over time [S1]."),
    )
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_ui_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Cited Document Q&A" in response.text


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database"] == "ok"


def test_upload_rejects_bad_file_type(client):
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_then_ask_returns_cited_answer(client, tmp_path):
    pdf_path = make_pdf(
        tmp_path / "leave.pdf",
        ["An employee accrues long service leave after a continuous period of service."],
    )

    with pdf_path.open("rb") as f:
        upload = client.post(
            "/documents", files={"file": ("leave.pdf", f, "application/pdf")}
        )
    assert upload.status_code == 200
    assert upload.json()["status"] == "ingested"

    ask = client.post("/ask", json={"question": "How does long service leave accrue?"})
    assert ask.status_code == 200
    body = ask.json()
    assert body["refused"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["page"] == 1


def test_documents_listing(client, tmp_path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Some content about workplace rules."])
    with pdf_path.open("rb") as f:
        client.post("/documents", files={"file": ("doc.pdf", f, "application/pdf")})

    response = client.get("/documents")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["chunk_count"] >= 1
