from app.db import get_pool
from app.retrieval.embed import embed_query

_SELECT = """
    SELECT
        c.id, c.document_id, c.page, c.section, c.text,
        d.title, d.source_url,
        c.embedding <=> %(q)s AS distance
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(q)s
    LIMIT %(k)s
"""


def vector_search(question: str, k: int = 20) -> list[dict]:
    query_vector = embed_query(question)
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(_SELECT, {"q": query_vector, "k": k}).fetchall()

    return [
        {
            "chunk_id": r[0],
            "document_id": r[1],
            "page": r[2],
            "section": r[3],
            "text": r[4],
            "title": r[5],
            "source_url": r[6],
            "distance": float(r[7]),
            "similarity": 1.0 - float(r[7]),
        }
        for r in rows
    ]
