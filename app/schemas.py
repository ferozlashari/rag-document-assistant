from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=10)


class SourceInfo(BaseModel):
    doc_id: str
    chunk_id: str
    page: int | None
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    llm_provider: str
    embedding_provider: str


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    filename: str


class DocumentsResponse(BaseModel):
    documents: list[str]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    version: str
