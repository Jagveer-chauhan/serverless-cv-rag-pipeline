"""Core configuration and settings management."""
import os
from pathlib import Path
from typing import List, Union, Optional
from dotenv import load_dotenv
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Locate repository paths and explicitly load .env
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BACKEND_DIR.parent

_candidate_env_paths = [
    BACKEND_DIR / ".env",
    ROOT_DIR / ".env",
    Path(".env"),
    Path("backend/.env")
]

# Load from the first existing .env file into os.environ
for env_path in _candidate_env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        break


class Settings(BaseSettings):
    """Application Settings strictly sourced from .env or environment variables."""
    model_config = SettingsConfigDict(
        env_file=[str(p) for p in _candidate_env_paths if p.exists()] or None,
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

    # Supabase (PostgreSQL + pgvector) - STRICTLY LOADED FROM .env
    SUPABASE_DB_URL: str = Field(default="", description="PostgreSQL connection string loaded strictly from .env")
    SUPABASE_KEY: str = Field(default="", description="Supabase API key loaded strictly from .env")

    # Hugging Face Serverless Inference API - STRICTLY LOADED FROM .env
    HF_API_KEY: str = Field(default="", description="Hugging Face API token loaded strictly from .env")
    HF_MODEL_NAME: str = Field(default="google/gemma-3-4b-it", description="Model name loaded from .env")
    HF_BASE_URL: str = Field(default="https://api-inference.huggingface.co", description="Hugging Face base inference URL loaded from .env")
    HF_API_URL: str = Field(default="", description="Optional direct custom LLM API URL loaded from .env")
    HF_EMBEDDING_API_URL: str = Field(default="", description="Optional direct custom embedding API URL loaded from .env")

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model name loaded from .env")
    EMBEDDING_DIM: int = 384

    # Keepalive Security - STRICTLY LOADED FROM .env
    KEEPALIVE_SECRET: str = Field(default="", description="Optional keepalive secret token from .env")

    @property
    def hf_llm_url(self) -> str:
        """Dynamic LLM inference endpoint URL."""
        if self.HF_API_URL and self.HF_API_URL.strip():
            return self.HF_API_URL.strip()
        return f"{self.HF_BASE_URL.rstrip('/')}/models/{self.HF_MODEL_NAME}"

    @property
    def hf_embedding_url(self) -> str:
        """Dynamic Embedding inference endpoint URL."""
        if self.HF_EMBEDDING_API_URL and self.HF_EMBEDDING_API_URL.strip():
            return self.HF_EMBEDDING_API_URL.strip()
        return f"{self.HF_BASE_URL.rstrip('/')}/models/{self.EMBEDDING_MODEL_NAME}"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return ["*"]


settings = Settings()
