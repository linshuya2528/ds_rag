from rag.schemas import Chunk, RetrievedChunk
from rag.validator import parse_json_response, validate_payload


def _evidence():
    chunk = Chunk("c1", "d1", "book.pdf", 2, 2, 1, "evidence", 8)
    return [RetrievedChunk(chunk, 1, 0.1)]


def test_valid_payload_cites_supplied_evidence():
    payload, issues = parse_json_response('{"answer":"yes","citations":[{"chunk_id":"c1"}],"insufficient_evidence":false}')
    assert not issues
    assert validate_payload(payload, _evidence())["passed"]


def test_validator_rejects_unknown_citation():
    validation = validate_payload(
        {"answer": "yes", "citations": [{"chunk_id": "not-there"}], "insufficient_evidence": False},
        _evidence(),
    )
    assert not validation["passed"]
