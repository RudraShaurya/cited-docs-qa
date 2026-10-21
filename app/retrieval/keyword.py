from app.db import get_pool

_SELECT = """
    SELECT
        c.id, c.document_id, c.page, c.section, c.text,
        d.title, d.source_url,
        ts_rank_cd(c.tsv, websearch_to_tsquery('english', %(q)s)) AS rank
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.tsv @@ websearch_to_tsquery('english', %(q)s)
    ORDER BY rank DESC
    LIMIT %(k)s
"""


def keyword_search(question: str, k: int = 20) -> list[dict]:
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(_SELECT, {"q": question, "k": k}).fetchall()

    return [
        {
            "chunk_id": r[0],
            "document_id": r[1],
            "page": r[2],
            "section": r[3],
            "text": r[4],
            "title": r[5],
            "source_url": r[6],
            "rank": float(r[7]),
        }
        for r in rows
    ]
