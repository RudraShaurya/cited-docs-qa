import argparse
import hashlib
import sys
from pathlib import Path

from app.config import settings
from app.db import get_pool, run_migrations
from app.ingest.chunk import chunk_document
from app.ingest.parse import parse_document

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def ingest_file(path: Path, title: str | None = None) -> dict:
    """Parse, chunk and store one file. Returns a summary dict.

    Skips (and reports) files whose sha256 is already in `documents`,
    so re-running ingestion over the same folder is a no-op for them.
    """
    digest = sha256_of(path)
    pool = get_pool()

    with pool.connection() as conn:
        existing = conn.execute(
            "SELECT id FROM documents WHERE sha256 = %s", (digest,)
        ).fetchone()
        if existing:
            return {"file": path.name, "status": "skipped", "chunks": 0}

        blocks = parse_document(path)
        chunks = chunk_document(
            blocks,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )

        doc_id = conn.execute(
            """
            INSERT INTO documents (title, source_url, licence, sha256)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (title or path.stem, None, None, digest),
        ).fetchone()[0]

        for c in chunks:
            conn.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, page, section, text)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (doc_id, c.chunk_index, c.page, c.section, c.text),
            )

        conn.commit()
        return {"file": path.name, "status": "ingested", "chunks": len(chunks)}


def ingest_directory(directory: Path) -> None:
    run_migrations()

    files = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        print(f"No supported files (.pdf, .docx) found under {directory}")
        return

    ingested = skipped = 0
    total_chunks = 0

    for path in files:
        result = ingest_file(path)
        if result["status"] == "ingested":
            ingested += 1
            total_chunks += result["chunks"]
            print(f"  ingested  {result['file']}  ({result['chunks']} chunks)")
        else:
            skipped += 1
            print(f"  skipped   {result['file']}  (already ingested)")

    print(
        f"\nDone. {ingested} file(s) ingested ({total_chunks} chunks), "
        f"{skipped} skipped, {len(files)} total."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF/DOCX files into the document store.")
    parser.add_argument("directory", type=Path, help="Directory to scan for .pdf/.docx files")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    ingest_directory(args.directory)


if __name__ == "__main__":
    main()
