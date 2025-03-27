"""Optional local Ollama backend (uses structured outputs via the `format` field)."""

from __future__ import annotations

import httpx

from .base import LLMError


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0,
                 transport: httpx.BaseTransport | None = None):
        self.name = f"ollama:{model}"
        self.model = model
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def generate(self, *, system: str, prompt: str, json_schema: dict | None = None) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "format": json_schema or "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        try:
            resp = self._client.post("/api/chat", json=body)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        if not content:
            raise LLMError("ollama returned an empty reply")
        return content
