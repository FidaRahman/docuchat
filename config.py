"""
config.py — Centralized application settings for DocuChat (free stack).

Uses pydantic-settings to load and validate environment variables from .env.
All other modules import from here — never read os.environ directly elsewhere.

Free stack:
  - LLM:        Groq  (GROQ_API_KEY)   — llama-3.3-70b-versatile
  - Embeddings: HuggingFace sentence-transformers — all-MiniLM-L6-v2 (local, no key)
  - Vector DB:  FAISS (unchanged, local)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Pydantic validates types at startup so misconfigurations fail fast
    rather than at the first API call.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Groq (free LLM API) ---
    groq_api_key: str = Field(..., description="Groq API key — get one free at console.groq.com")

    chat_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model used for chat completions",
    )

    # --- Local embeddings (no API key needed) ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model for local embeddings",
    )

    # --- Chunking ---
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=4000,
        description="Character size of each text chunk (character-based splitter, no tiktoken needed)",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlapping characters between adjacent chunks",
    )

    # --- Retrieval ---
    top_k_results: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of most-relevant chunks retrieved per query",
    )

    # --- Session management ---
    max_history_turns: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Max conversation turns (user+bot pairs) stored per session",
    )

    # --- Persistence ---
    faiss_index_path: str = Field(
        default="faiss_index",
        description="Directory where FAISS index files are saved",
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Uvicorn bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Uvicorn bind port")

    # --- LLM behaviour ---
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0 = deterministic, best for RAG)",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum tokens in the LLM's response",
    )

    @field_validator("groq_api_key")
    @classmethod
    def api_key_must_not_be_placeholder(cls, v: str) -> str:
        """Reject the placeholder string that ships in .env.example."""
        if v.strip() in ("your_groq_api_key_here", "", "gsk_..."):
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com, "
                "then copy .env.example to .env and add it."
            )
        return v.strip()

    @property
    def faiss_index_dir(self) -> Path:
        """Resolved Path object for the FAISS persistence directory."""
        return Path(self.faiss_index_path)

    @property
    def system_prompt(self) -> str:
        """
        System prompt injected into every chat completion request.
        Enforces grounded, citation-backed answers and prevents hallucination.
        """
        return (
            "You are a document assistant. Answer questions ONLY using the context "
            "provided below. If the answer is not in the context, say: "
            "'I couldn't find information about that in the uploaded documents.' "
            "Do not use prior knowledge. Be concise and accurate. "
            "Always cite which part of the context you used."
        )

    @property
    def no_context_reply(self) -> str:
        """Standard reply when retrieval returns no usable chunks."""
        return "I couldn't find information about that in the uploaded documents."


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Using lru_cache means the .env file is read exactly once per process
    lifetime, which is efficient and avoids repeated disk I/O.
    """
    return Settings()


# Module-level convenience alias so callers can do:
#   from config import settings
settings: Settings = get_settings()
