from sentence_transformers import SentenceTransformer

from app.config import settings
from app.db import get_pool

# BAAI/bge-* models expect this instruction prepended to queries (not passages)
# for retrieval tasks; see the model card.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    model = get_model()
    vector = model.encode(
        BGE_QUERY_INSTRUCTION + text, normalize_embeddings=True, show_progress_bar=False
    )
    return vector.tolist()


def backfill_missing_embeddings(batch_size: int = 64) -> int:
    """Embed every chunk whose `embedding` column is still NULL. Returns count embedded."""
    pool = get_pool()
    total = 0

    while True:
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, text FROM chunks WHERE embedding IS NULL LIMIT %s",
                (batch_size,),
            ).fetchall()
            if not rows:
                break

            ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]
            vectors = embed_passages(texts)

            for chunk_id, vector in zip(ids, vectors):
                conn.execute(
                    "UPDATE chunks SET embedding = %s WHERE id = %s", (vector, chunk_id)
                )
            conn.commit()
            total += len(rows)

    return total


if __name__ == "__main__":
    count = backfill_missing_embeddings()
    print(f"Embedded {count} chunk(s).")
