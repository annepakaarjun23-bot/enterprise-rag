from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.api.main import app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert "qdrant" in body["checks"]
    assert "postgres" in body["checks"]


def test_chat_non_streaming(client):
    r = client.post(
        "/chat",
        json={
            "message": "How do I add memory to a LangGraph agent?",
            "retrieval_mode": "hybrid",
            "top_k": 5,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"], "empty answer"
    assert len(body["chunks"]) > 0
    assert body["duration_ms"] > 0
    assert body["session_id"]


def test_chat_streaming(client):
    with client.stream(
        "POST",
        "/chat/stream",
        json={
            "message": "What is create_agent?",
            "retrieval_mode": "hybrid",
            "top_k": 5,
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        events = []
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            import json
            events.append(json.loads(line[len("data: "):]))

    types = [e["type"] for e in events]
    assert "session" in types
    assert "done" in types
    # at least one token event should have fired
    assert "token" in types or "error" in types


def test_chat_request_validation(client):
    # too-long message -> 422
    r = client.post("/chat", json={"message": "x" * 5000})
    assert r.status_code == 422

    # invalid retrieval mode -> 422
    r = client.post("/chat", json={"message": "hi", "retrieval_mode": "bogus"})
    assert r.status_code == 422


def test_request_id_header(client):
    r = client.get("/health")
    assert "x-request-id" in r.headers
    assert len(r.headers["x-request-id"]) > 0