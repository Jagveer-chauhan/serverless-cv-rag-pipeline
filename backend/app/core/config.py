"""Core configuration and settings management."""
import os
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    PROJECT_NAME: str = "Serverless CV Parsing & RAG Pipeline"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Strict SLA Threshold
    SLA_TARGET_MS: float = 5000.0

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://cv-rag-frontend.onrender.com",
        "*"
    ]

    # Supabase (PostgreSQL + pgvector)
    # Default to sqlite or test fallback if SUPABASE_DB_URL is not provided
    SUPABASE_DB_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
    SUPABASE_KEY: str = ""

    # Hugging Face Serverless Inference API
    HF_API_KEY: str = ""
    HF_MODEL_NAME: str = "google/gemma-3-4b-it"
    HF_API_URL: str = "https://api-inference.huggingface.co/models/google/gemma-3-4b-it"

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Keepalive Security (Optional)
    KEEPALIVE_SECRET: str = ""

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return ["*"]


settings = Settings()
