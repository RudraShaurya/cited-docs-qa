from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str

    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    chunk_size_tokens: int = 450
    chunk_overlap_tokens: int = 60

    retrieval_top_k: int = 5
    refusal_min_similarity: float = 0.35


settings = Settings()
