"""
Vector store abstraction.

- NumpyVectorStore: a real, fully offline, in-memory vector store using
  NumPy cosine similarity. This is what gets exercised in this project's
  tests since it needs no external services.
- ChromaVectorStore: production adapter to ChromaDB (persistent, requires
  the chromadb package). Same interface, so swapping is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.loaders import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class BaseVectorStore:
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[RetrievedChunk]:
        raise NotImplementedError


class NumpyVectorStore(BaseVectorStore):
    """In-memory cosine-similarity vector store. No external dependencies beyond NumPy."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("Number of vectors must match number of chunks")
        self._chunks.extend(chunks)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[RetrievedChunk]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1e-10
        scores = (self._vectors @ query_vector) / norms

        top_k_idx = np.argsort(scores)[::-1][:k]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top_k_idx if scores[i] > 0]

    def __len__(self) -> int:
        return len(self._chunks)


class ChromaVectorStore(BaseVectorStore):
    """Production adapter to a persistent ChromaDB collection. Requires the chromadb package."""

    def __init__(self, collection_name: str = "documents", persist_dir: str = "./chroma_db"):
        import chromadb  # lazy import; not installed in this sandbox

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "page": c.page or 0} for c in chunks],
        )

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[RetrievedChunk]:
        results = self._collection.query(query_embeddings=[query_vector.tolist()], n_results=k)
        out = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            chunk = Chunk(
                doc_id=meta["doc_id"],
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                page=meta.get("page"),
            )
            # Chroma returns distance; convert to a similarity-like score.
            distance = results["distances"][0][i]
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - distance))
        return out
