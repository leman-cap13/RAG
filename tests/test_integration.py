from unittest.mock import AsyncMock

from conftest import FAKE_ANSWER

import api as api_module
from config import settings
from rag.session import get_history


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_index_is_idempotent(client):
    first = await client.post("/index")
    assert first.status_code == 200
    results = first.json()
    assert any(r["file"] == "sample.txt" and r["status"] == "indexed" and r["chunks"] > 0 for r in results)

    second = await client.post("/index")
    assert second.status_code == 200
    assert all(r["status"] == "skipped" for r in second.json())


async def test_ask_happy_path_runs_full_pipeline(client):
    await client.post("/index")

    response = await client.post(
        "/ask", json={"question": "ADA Universiteti nə vaxt təsis edilib?", "top_k": 2}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == FAKE_ANSWER
    assert "sample.txt" in body["sources"]
    assert len(body["context"]) > 0
    assert body["session_id"]  # a session_id is minted even when the caller doesn't send one


async def test_ask_empty_question_rejected(client):
    response = await client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


async def test_ask_second_identical_call_is_served_from_cache(client):
    await client.post("/index")
    # Two separate (session-less) callers asking the same first-turn question
    # should both get the cached answer, even though each gets its own session_id.
    payload = {"question": "Universitetdə hansı proqramlar var?", "top_k": 2}

    first = await client.post("/ask", json=payload)
    assert first.status_code == 200

    second = await client.post("/ask", json=payload)
    assert second.status_code == 200

    first_body, second_body = first.json(), second.json()
    assert first_body["answer"] == second_body["answer"]
    assert first_body["sources"] == second_body["sources"]
    assert first_body["session_id"] != second_body["session_id"]


async def test_conversation_memory_persists_within_session(client):
    await client.post("/index")
    session_id = "test-conversation-memory-session"

    first = await client.post(
        "/ask", json={"question": "ADA nə vaxt yaranıb?", "session_id": session_id}
    )
    assert first.status_code == 200
    assert first.json()["session_id"] == session_id

    second = await client.post(
        "/ask", json={"question": "bəs orda hansı proqramlar var?", "session_id": session_id}
    )
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id

    history = get_history(session_id)
    assert [turn["role"] for turn in history] == ["user", "assistant", "user", "assistant"]
    assert history[2]["content"] == "bəs orda hansı proqramlar var?"


async def test_ask_rejected_when_queue_is_full(client, monkeypatch):
    monkeypatch.setattr(
        api_module.rabbit_client, "queue_depth", AsyncMock(return_value=settings.rabbitmq_max_queue_depth)
    )

    response = await client.post("/ask", json={"question": "Bu sual keşdə deyil, unikal sualdır."})

    assert response.status_code == 503


async def test_delete_unknown_source_returns_404(client):
    response = await client.delete("/sources/does-not-exist.txt")
    assert response.status_code == 404
