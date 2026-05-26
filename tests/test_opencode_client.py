"""Tests for opencode client — extraction and env-var logic."""

import os
from opencode_client import OpencodeClient, get_llm_client


def test_extract_single_text():
    client = OpencodeClient()
    response = {"parts": [{"type": "text", "text": "Hello, world!"}]}
    assert client._extract_text(response) == "Hello, world!"


def test_extract_multiple_text_parts():
    client = OpencodeClient()
    response = {
        "parts": [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
    }
    assert client._extract_text(response) == "First part\nSecond part"


def test_extract_ignores_non_text_parts():
    client = OpencodeClient()
    response = {
        "parts": [
            {"type": "tool_use", "name": "read_file", "input": {"path": "test.md"}},
            {"type": "text", "text": "Actual response"},
        ]
    }
    assert client._extract_text(response) == "Actual response"


def test_extract_no_text_parts():
    client = OpencodeClient()
    response = {"parts": [{"type": "tool_use", "name": "think"}]}
    assert client._extract_text(response) == ""


def test_extract_no_parts():
    client = OpencodeClient()
    response = {"info": {"id": "123"}}
    assert client._extract_text(response) == ""


# ── get_llm_client ────────────────────────────────────────────────────────────

def test_get_llm_client_returns_none_for_gemini():
    os.environ["LLM_PROVIDER"] = "gemini"
    assert get_llm_client() is None


def test_get_llm_client_returns_none_when_not_set(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert get_llm_client() is None


def test_get_llm_client_returns_none_when_server_unreachable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "opencode")
    monkeypatch.setenv("OPENCODE_URL", "http://localhost:1")
    assert get_llm_client() is None
