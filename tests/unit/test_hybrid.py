from app.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_boosts_items_appearing_in_both_lists():
    vector_results = [
        {"chunk_id": 1, "text": "a", "similarity": 0.9},
        {"chunk_id": 2, "text": "b", "similarity": 0.8},
        {"chunk_id": 3, "text": "c", "similarity": 0.7},
    ]
    keyword_results = [
        {"chunk_id": 3, "text": "c", "rank": 0.5},
        {"chunk_id": 1, "text": "a", "rank": 0.4},
    ]

    fused = reciprocal_rank_fusion([vector_results, keyword_results])
    fused_ids = [r["chunk_id"] for r in fused]

    # chunk 1 (rank 1 in both lists) should outrank chunk 3 (rank 1 vector, rank 1 keyword too)
    # and both should outrank chunk 2, which only appears once.
    assert fused_ids[0] in (1, 3)
    assert 2 in fused_ids
    assert fused_ids.index(2) > fused_ids.index(1)


def test_rrf_merges_fields_from_both_lists():
    vector_results = [{"chunk_id": 1, "text": "a", "similarity": 0.9}]
    keyword_results = [{"chunk_id": 1, "text": "a", "rank": 0.4}]

    fused = reciprocal_rank_fusion([vector_results, keyword_results])

    assert fused[0]["similarity"] == 0.9
    assert fused[0]["rank"] == 0.4


def test_rrf_on_single_list_preserves_order():
    results = [{"chunk_id": i, "text": str(i)} for i in [5, 2, 8]]
    fused = reciprocal_rank_fusion([results])
    assert [r["chunk_id"] for r in fused] == [5, 2, 8]


def test_rrf_on_disjoint_lists_keeps_both():
    vector_results = [{"chunk_id": 1, "text": "a"}]
    keyword_results = [{"chunk_id": 2, "text": "b"}]

    fused = reciprocal_rank_fusion([vector_results, keyword_results])

    assert {r["chunk_id"] for r in fused} == {1, 2}


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
