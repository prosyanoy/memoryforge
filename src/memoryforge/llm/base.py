"""Pluggable LLM interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..config import Settings


class LLMError(Exception):
    """The model backend failed (network, HTTP error, empty reply)."""


class LLMClient(Protocol):
    name: str

    def generate(self, *, system: str, prompt: str, json_schema: dict | None = None) -> str:
        """Return the raw text completion (expected to be a JSON document)."""
        ...


def build_llm(settings: "Settings") -> LLMClient:
    if settings.llm_backend == "ollama":
        from .ollama import OllamaClient

        return OllamaClient(settings.ollama_url, settings.ollama_model, timeout=settings.llm_timeout_s)
    from .stub import StubLLM

    return StubLLM()
