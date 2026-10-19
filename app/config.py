import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "auto"))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", ""))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploaded_docs")

    def resolve_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "mock"

    def resolve_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider
        if self.openai_api_key:
            return "openai"
        return "local-tfidf-svd"


settings = Settings()
