"""
opencode_client.py — Python HTTP client for the opencode server API.

Connects to `opencode serve` (HTTP) and provides a generate_content interface
that mirrors the Gemini API, allowing drop-in replacement in query.py and agent.py.

Usage:
    client = OpencodeClient(base_url="http://opencode:4096")
    response = client.generate_content(
        prompt="Hello",
        system_instruction="You are helpful",
        temperature=0.1,
    )
"""

import json
import os
from typing import Any, Optional
from urllib.parse import urljoin

import requests


class OpencodeClient:
    """A lightweight client for the opencode server's chat API."""

    def __init__(
        self,
        base_url: str = "http://localhost:4096",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session_id: Optional[str] = None
        self.default_provider = os.getenv("OPENCODE_PROVIDER", "opencode-go")
        raw_model = os.getenv("OPENCODE_MODEL", "")
        self.default_model: Optional[str] = raw_model.strip() or None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Get or create a persistent session."""
        if self._session_id is None:
            self._session_id = self._create_session()
        return self._session_id

    def _create_session(self) -> str:
        """Create a new opencode session, return its ID."""
        resp = requests.post(
            urljoin(self.base_url, "/session"),
            json={"title": "Obsidian Recall"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["id"]

    def reset_session(self) -> None:
        """Force a new session on the next request."""
        self._session_id = None

    # ------------------------------------------------------------------
    # Generation — matches Gemini-like interface
    # ------------------------------------------------------------------

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Send a prompt to opencode and return the text response.

        This mirrors the Gemini API's `models.generate_content()` interface
        for easy substitution in query.py and agent.py.
        """
        sid = self.session_id

        # Build parts
        parts = [{"type": "text", "text": prompt}]

        # Send the message
        body: dict[str, Any] = {"parts": parts}

        if system_instruction is not None:
            body["system"] = system_instruction

        if temperature is not None:
            body["temperature"] = temperature

        # Specify provider/model (server picks default if model omitted)
        if self.default_model:
            body["model"] = {
                "providerID": self.default_provider,
                "modelID": self.default_model,
            }

        resp = requests.post(
            urljoin(self.base_url, f"/session/{sid}/message"),
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract text from response parts
        return self._extract_text(data)

    @staticmethod
    def _extract_text(response_data: dict) -> str:
        """Extract text content from an opencode message response."""
        parts = response_data.get("parts", [])
        text_parts = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "\n".join(text_parts)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """Check if the opencode server is reachable."""
        try:
            resp = requests.get(
                urljoin(self.base_url, "/global/health"),
                timeout=5,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # Programmatic authentication (for remote servers without auth.json)
    # ------------------------------------------------------------------

    def authenticate(self, provider_id: str, api_key: str) -> bool:
        """
        Set an API key for a provider via the auth API.

        This removes the need to mount auth.json — the key is stored
        in the server's runtime state. Useful for remote deployments.
        """
        try:
            resp = requests.put(
                urljoin(self.base_url, f"/auth/{provider_id}"),
                json={"type": "api", "key": api_key},
                timeout=10,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False


# ------------------------------------------------------------------
# Helper: get the appropriate LLM client based on env var
# ------------------------------------------------------------------

def get_llm_client() -> Optional[OpencodeClient]:
    """
    Return an OpencodeClient if LLM_PROVIDER=opencode, else None.

    The caller falls back to Gemini when this returns None.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider != "opencode":
        return None

    base_url = os.getenv(
        "OPENCODE_URL",
        "http://localhost:4096",
    ).rstrip("/")

    client = OpencodeClient(base_url=base_url)

    # Quick health check — log but don't fail
    if not client.health():
        import logging
        logging.warning(
            "opencode server not reachable at %s — falling back to Gemini. "
            "Make sure `opencode serve` is running.",
            base_url,
        )
        return None

    # Programmatic auth: set API key from env var (no auth.json mount needed)
    api_key = os.getenv("OPENCODE_GO_API_KEY", "").strip()
    if api_key:
        provider_id = os.getenv("OPENCODE_PROVIDER", "opencode-go")
        client.authenticate(provider_id, api_key)

    return client
