"""Tests for the API. The LLM is replaced with a fake, so tests are fast and need no Ollama."""
from fastapi.testclient import TestClient

import docagent.api as api
from docagent.schema import Receipt

client = TestClient(api.app)
FAKE = Receipt(company="ABC SDN BHD", date="25/12/2018", address="JOHOR", total=9.0)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_returns_receipt(monkeypatch):
    monkeypatch.setattr(api, "extract", lambda text: FAKE)
    response = client.post("/extract", json={"text": "ABC SDN BHD\nTOTAL 9.00"})
    assert response.status_code == 200
    assert response.json()["total"] == 9.0


def test_failed_extraction_returns_422(monkeypatch):
    monkeypatch.setattr(api, "extract", lambda text: None)
    response = client.post("/extract", json={"text": "unreadable"})
    assert response.status_code == 422


def test_empty_text_is_rejected():
    response = client.post("/extract", json={"text": ""})
    assert response.status_code == 422