"""App settings, loaded from environment / .env via pydantic-settings."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    airouter_api_key: str = ""
    
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    qdrant_url: str = "http://localhost:7333"
    qdrant_api_key: str = ""
    qdrant_collection_leaves: str = "chunks_md_codesafe_v1"

    embedding_base_url: str = "https://api.airouter.in/v1"
    embedding_api_key: str = ""               
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64            

    chunk_strategy: str = "md_codesafe_v1"


settings = Settings()