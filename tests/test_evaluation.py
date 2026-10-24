from rag.evaluation import retrieval_metrics


def test_retrieval_metrics():
    metrics = retrieval_metrics([
        {"expected_pages": [3], "retrieved_chunks": [{"vector_rank": 2, "chunk": {"page_start": 3}}]},
        {"expected_pages": [4], "retrieved_chunks": []},
    ])
    assert metrics == {"recall_at_k": 0.5, "mrr": 0.25}
