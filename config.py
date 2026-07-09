"""
config.py - Centralized application settings for DocuChat.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str = Field(..., description="Groq API key")

    chat_model: str = Field(default="llama-3.3-70b-versatile")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    top_k_results: int = Field(default=4)
    max_history_turns: int = Field(default=6)
    faiss_index_path: str = Field(default="faiss_index")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=1024)

    @field_validator("groq_api_key")
    @classmethod
    def groq_key_must_not_be_placeholder(cls, v: str) -> str:
        if v.strip() in ("your_groq_api_key_here", "", "gsk_..."):
            raise ValueError("GROQ_API_KEY is not set.")
        return v.strip()

    @property
    def faiss_index_dir(self) -> Path:
        return Path(self.faiss_index_path)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a document assistant. Answer questions ONLY using the context "
            "provided below. If the answer is not in the context, say: "
            "'I couldn't find information about that in the uploaded documents.' "
            "Do not use prior knowledge. Be concise and accurate. "
            "Always cite which part of the context you used."
        )

    @property
    def no_context_reply(self) -> str:
        return "I couldn't find information about that in the uploaded documents."


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
