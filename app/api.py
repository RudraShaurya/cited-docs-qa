import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.answer import AnswerResult, answer_question
from app.config import settings
from app.db import health_check, run_migrations
from app.ingest.pipeline import SUPPORTED_SUFFIXES, ingest_file
from app.llm.factory import get_llm_client
from app.logging import log_event, new_request_id
from app.retrieval.embed import backfill_missing_embeddings

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_DIR = Path("/tmp/cited-docs-qa-uploads")
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    run_migrations()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Cited Document Q&A", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None
    mode: str = "hybrid"


class CitationResponse(BaseModel):
    marker: int
    document_id: str
    title: str
    page: int | None
    section: str | None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    refused: bool
    retrieval: list[dict]
    timings_ms: dict[str, float]
    model: str


class DocumentResponse(BaseModel):
    id: str
    title: str
    chunk_count: int
    ingested_at: str


class IngestResponse(BaseModel):
    status: str
    chunks: int


@app.get("/health")
def health():
    db_ok = health_check()
    llm_ok = True
    llm_error = None
    try:
        get_llm_client()
    except Exception as exc:
        llm_ok = False
        llm_error = str(exc)

    status = "ok" if db_ok and llm_ok else "degraded"
    return {
        "status": status,
        "database": "ok" if db_ok else "unreachable",
        "llm_provider": settings.llm_provider,
        "llm_configured": llm_ok,
        "llm_error": llm_error,
    }


@app.post("/documents", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Use one of: {sorted(SUPPORTED_SUFFIXES)}",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = UPLOAD_DIR / temp_name
    temp_path.write_bytes(contents)

    try:
        result = ingest_file(temp_path, title=Path(file.filename or temp_name).stem)
        if result["status"] == "ingested":
            backfill_missing_embeddings()
    finally:
        temp_path.unlink(missing_ok=True)

    return IngestResponse(status=result["status"], chunks=result["chunks"])


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents():
    from app.db import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.title, d.ingested_at, COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id, d.title, d.ingested_at
            ORDER BY d.ingested_at DESC
            """
        ).fetchall()

    return [
        DocumentResponse(
            id=str(r[0]), title=r[1], ingested_at=r[2].isoformat(), chunk_count=r[3]
        )
        for r in rows
    ]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    new_request_id()
    t0 = time.monotonic()

    try:
        llm = get_llm_client()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"LLM not configured: {exc}") from exc

    try:
        result: AnswerResult = answer_question(
            request.question, llm, mode=request.mode, top_k=request.top_k
        )
    except Exception as exc:
        log_event(
            "ask_failed",
            question_len=len(request.question),
            mode=request.mode,
            error=str(exc),
        )
        raise HTTPException(status_code=503, detail="The language model is unavailable.") from exc

    total_ms = (time.monotonic() - t0) * 1000
    log_event(
        "ask",
        question_len=len(request.question),
        mode=request.mode,
        chunks_retrieved=len(result.retrieval),
        refused=result.refused,
        timings_ms=result.timings_ms,
        total_ms=total_ms,
    )

    return AskResponse(
        answer=result.answer,
        citations=[CitationResponse(**vars(c)) for c in result.citations],
        refused=result.refused,
        retrieval=[
            {
                "chunk_id": c["chunk_id"],
                "document_id": str(c["document_id"]),
                "page": c.get("page"),
                "similarity": c.get("similarity"),
                "rank": c.get("rank"),
                "rrf_score": c.get("rrf_score"),
            }
            for c in result.retrieval
        ],
        timings_ms=result.timings_ms,
        model=result.model,
    )
