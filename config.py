"""Configurações centrais da aplicação via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # MinIO
    MINIO_ENDPOINT: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str

    # PostgreSQL
    DATABASE_URL: str

    # Milvus
    MILVUS_HOST: str
    MILVUS_PORT: int
    MILVUS_COLLECTION: str

    # HuggingFace
    HUGGINGFACE_API_TOKEN: str
    HF_LLM_MODEL: str
    HF_EMBED_MODEL: str
    LLM_PROVIDER: str = "huggingface"

    # OpenAI (opcional)
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_REASONING_EFFORT: str | None = None
    OPENAI_TIMEOUT_SECONDS: int = 120

    # RAG Pipeline
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    TOP_K_RETRIEVAL: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()