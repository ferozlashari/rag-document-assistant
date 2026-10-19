"""
FastAPI application for the Intelligent RAG Document Assistant.

Run with:  uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs

With no OPENAI_API_KEY set, embeddings use the built-in LocalTfidfEmbeddings
and generation uses the built-in MockProvider, so /upload + /chat work fully
offline out of the box for demos.
"""

from __future__ import annotations

import os
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.embeddings import build_embedding_provider
from app.llm_providers import build_provider
from app.rag_pipeline import RAGPipeline
from app.schemas import (
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from app.vectorstore import NumpyVectorStore

app = FastAPI(
    title="Intelligent RAG Document Assistant",
    description="Upload documents, then ask questions grounded in their contents.",
    version=__version__,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

os.makedirs(settings.upload_dir, exist_ok=True)

_embedding_provider_name = settings.resolve_embedding_provider()
_llm_provider_name = settings.resolve_llm_provider()
_llm_api_key = settings.anthropic_api_key if _llm_provider_name == "anthropic" else settings.openai_api_key

embedder = build_embedding_provider(_embedding_provider_name, api_key=settings.openai_api_key)
llm = build_provider(_llm_provider_name, api_key=_llm_api_key, model=settings.model_name)
pipeline = RAGPipeline(embedder=embedder, store=NumpyVectorStore(), llm=llm)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=llm.name,
        embedding_provider=embedder.name,
        version=__version__,
    )


@app.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    doc_id = f"{os.path.splitext(file.filename)[0]}_{uuid.uuid4().hex[:8]}"
    dest_path = os.path.join(settings.upload_dir, f"{doc_id}{os.path.splitext(file.filename)[1]}")

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks_created = pipeline.ingest_file(
        dest_path, doc_id=doc_id, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    if chunks_created == 0:
        raise HTTPException(status_code=422, detail="No extractable text found in the uploaded document.")

    return IngestResponse(doc_id=doc_id, chunks_created=chunks_created, filename=file.filename)


@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest) -> QueryResponse:
    result = pipeline.query(req.question, k=req.k)
    sources = [
        SourceInfo(
            doc_id=s.chunk.doc_id,
            chunk_id=s.chunk.chunk_id,
            page=s.chunk.page,
            score=round(s.score, 4),
            text=s.chunk.text,
        )
        for s in result.sources
    ]
    return QueryResponse(
        answer=result.answer,
        sources=sources,
        llm_provider=llm.name,
        embedding_provider=embedder.name,
    )


@app.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    return DocumentsResponse(documents=pipeline.ingested_documents, total_chunks=len(pipeline.store))
