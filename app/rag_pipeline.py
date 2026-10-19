"""
The RAG pipeline: ties together loaders, embeddings, vectorstore, and the LLM
provider into ingest() and query() operations.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.embeddings import BaseEmbeddingProvider
from app.llm_providers import BaseLLMProvider, build_rag_prompt
from app.loaders import load_and_chunk
from app.vectorstore import BaseVectorStore, RetrievedChunk


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    def __init__(self, embedder: BaseEmbeddingProvider, store: BaseVectorStore, llm: BaseLLMProvider):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self._ingested_docs: list[str] = []

    def ingest_file(self, path: str, doc_id: str, chunk_size: int = 500, overlap: int = 100) -> int:
        """Load, chunk, embed, and store a document. Returns number of chunks added."""
        chunks = load_and_chunk(path, doc_id=doc_id, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self._ingested_docs.append(doc_id)
        return len(chunks)

    def query(self, question: str, k: int = 4) -> RAGAnswer:
        if not self._ingested_docs:
            return RAGAnswer(answer="No documents have been ingested yet.", sources=[])

        query_vector = self.embedder.embed_query(question)
        retrieved = self.store.search(query_vector, k=k)

        if not retrieved:
            return RAGAnswer(answer="I couldn't find anything relevant in the ingested documents.", sources=[])

        prompt = build_rag_prompt(question, retrieved)
        answer = self.llm.generate(prompt)
        return RAGAnswer(answer=answer, sources=retrieved)

    @property
    def ingested_documents(self) -> list[str]:
        return list(self._ingested_docs)
