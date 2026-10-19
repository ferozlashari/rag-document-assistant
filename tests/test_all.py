"""
Test suite covering loaders.py, embeddings.py (LocalTfidfEmbeddings), vectorstore.py
(NumpyVectorStore), llm_providers.py (MockProvider), and the full rag_pipeline.py.

Run with:  python3 tests/test_all.py
No network, no API key, no FastAPI/ChromaDB required - matches what actually
runs in CI. main.py additionally requires `fastapi` to import.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.loaders import load_and_chunk, chunk_pages
from app.embeddings import LocalTfidfEmbeddings
from app.vectorstore import NumpyVectorStore
from app.llm_providers import MockProvider
from app.rag_pipeline import RAGPipeline

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "app", "sample_pdfs", "company_handbook.pdf")

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


print("== loaders: chunking ==")
chunks = chunk_pages("doc1", [(1, "Sentence one. Sentence two. Sentence three. " * 10)], chunk_size=100, overlap=20)
check("produces multiple chunks", len(chunks) > 1)
check("chunks respect rough size target", all(len(c.text) < 300 for c in chunks))

print("== loaders: real PDF extraction ==")
pdf_chunks = load_and_chunk(SAMPLE_PDF, doc_id="handbook", chunk_size=150, overlap=30)
check("extracts text from real PDF", len(pdf_chunks) >= 4)
check("preserves page numbers", any(c.page == 2 for c in pdf_chunks))

print("== embeddings: LocalTfidfEmbeddings ==")
embedder = LocalTfidfEmbeddings(n_components=32)
vectors = embedder.embed_documents([c.text for c in pdf_chunks])
check("returns one vector per chunk", vectors.shape[0] == len(pdf_chunks))
qvec = embedder.embed_query("vacation days")
check("query vector matches doc vector dimensionality", qvec.shape[0] == vectors.shape[1])

print("== vectorstore: NumpyVectorStore ==")
store = NumpyVectorStore()
store.add(pdf_chunks, vectors)
check("store length matches chunks added", len(store) == len(pdf_chunks))
results = store.search(qvec, k=2)
check("search returns results", len(results) > 0)
check("results are sorted by descending score", results[0].score >= results[-1].score)

print("== full RAG pipeline (mirrors main.py) ==")
pipeline = RAGPipeline(embedder=LocalTfidfEmbeddings(n_components=32), store=NumpyVectorStore(), llm=MockProvider())
n = pipeline.ingest_file(SAMPLE_PDF, doc_id="handbook", chunk_size=150, overlap=30)
check("ingestion returns chunk count", n >= 4)

result = pipeline.query("How many vacation days do employees get?", k=2)
check("answers vacation question correctly", "15" in result.answer)
check("returns source citations", len(result.sources) > 0)

result = pipeline.query("What is the meal reimbursement limit?", k=2)
check("answers reimbursement question correctly", "75" in result.answer)

empty_pipeline = RAGPipeline(embedder=LocalTfidfEmbeddings(), store=NumpyVectorStore(), llm=MockProvider())
result = empty_pipeline.query("anything")
check("handles empty pipeline gracefully", "No documents" in result.answer)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
