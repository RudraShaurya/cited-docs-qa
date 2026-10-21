from app.retrieval.keyword import keyword_search
from app.retrieval.vector import vector_search

RRF_K = 60


def reciprocal_rank_fusion(result_lists: list[list[dict]], k_constant: int = RRF_K) -> list[dict]:
    """Fuse several ranked result lists by Reciprocal Rank Fusion.

    Each input list must be sorted best-first and each item must carry a
    "chunk_id" key. Works purely on ranks, so a vector cosine-distance list
    and a text-rank list can be fused without comparing incomparable scores.
    """
    scores: dict[int, float] = {}
    merged: dict[int, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_constant + rank)
            merged[chunk_id] = {**item, **merged.get(chunk_id, {})}

    fused = [{**merged[chunk_id], "rrf_score": score} for chunk_id, score in scores.items()]
    fused.sort(key=lambda row: row["rrf_score"], reverse=True)
    return fused


def retrieve(question: str, k: int = 5, mode: str = "hybrid", fetch_k: int = 20) -> list[dict]:
    """Retrieve the top-k chunks for a question using the given mode."""
    if mode == "vector":
        return vector_search(question, k=fetch_k)[:k]
    if mode == "keyword":
        return keyword_search(question, k=fetch_k)[:k]
    if mode == "hybrid":
        vector_results = vector_search(question, k=fetch_k)
        keyword_results = keyword_search(question, k=fetch_k)
        fused = reciprocal_rank_fusion([vector_results, keyword_results])
        return fused[:k]
    raise ValueError(f"Unknown retrieval mode: {mode!r}. Use 'vector', 'keyword' or 'hybrid'.")
