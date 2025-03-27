"""Runtime configuration, read from MEMORYFORGE_* environment variables (and .env)."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMORYFORGE_", env_file=".env", extra="ignore")

    # storage
    db_path: str = "./data/memoryforge.db"  # ":memory:" is supported

    # queue
    queue_backend: Literal["memory", "kafka"] = "memory"
    kafka_bootstrap: str = "localhost:19092"
    kafka_topic: str = "conversation.events"
    kafka_group: str = "memoryforge"
    run_worker: bool = True

    # extraction
    llm_backend: Literal["stub", "ollama"] = "stub"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_timeout_s: float = 60.0
    prompt_version: str = "v2"
    extraction_attempts: int = 2
    min_confidence: float = 0.5
    default_country_code: str = "1"

    # retention ("what to forget")
    retention_days: int = 365
    resolved_issue_ttl_days: int = 30

    # embeddings / vector search
    embedder: Literal["hash", "fastembed"] = "hash"
    embedding_dim: int = 384  # used by the hash embedder; fastembed reports its own
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    vector_backend: Literal["auto", "faiss", "numpy"] = "auto"

    # ranking
    recency_half_life_days: float = 30.0
    weight_similarity: float = 0.6
    weight_recency: float = 0.25
    weight_confidence: float = 0.15

    # observability
    log_format: Literal["console", "json"] = "console"
    log_level: str = "INFO"
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    span_buffer_size: int = 5000

    # access
    admin_token: str = ""
