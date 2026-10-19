# Intelligent RAG Document Assistant

Upload PDF or text documents and ask questions grounded in their contents. Built with **LangChain**-compatible embedding/LLM interfaces, a real **vector search** layer, and a **FastAPI** REST API.

## Features

- **Real PDF ingestion** (`pypdf`): extracts text per-page, then chunks it on sentence boundaries with overlap so context isn't cut mid-sentence.
- **Pluggable embeddings**: OpenAI (`text-embedding-3-small`), HuggingFace (`sentence-transformers`), or a built-in **LocalTfidfEmbeddings** (TF-IDF + adaptive SVD) that runs fully offline with zero API key.
- **Pluggable vector store**: an in-memory NumPy cosine-similarity store (used for tests/demos), or a **ChromaDB** adapter for persistent production use — same interface, one-line swap.
- **Grounded generation**: retrieved chunks are injected into a strict "answer only from context" prompt, so answers are traceable to specific source chunks (with page numbers and similarity scores).
- **REST API**: `/upload`, `/chat`, `/documents`, `/health`.

## Architecture

```
PDF/TXT --> loaders.py (extract + chunk) --> embeddings.py (vectorize)
                                                      │
                                                      ▼
                                          vectorstore.py (store + search)
                                                      │
question --> embeddings.py (vectorize) --> similarity search --> top-k chunks
                                                      │
                                                      ▼
                                    llm_providers.py (grounded generation)
                                                      │
                                                      ▼
                                          answer + cited sources
```

## Project structure

```
app/
├── main.py            FastAPI app: /upload, /chat, /documents, /health
├── rag_pipeline.py     Ties ingest() and query() together
├── loaders.py           PDF/text extraction + sentence-aware chunking
├── embeddings.py        OpenAI / HuggingFace / LocalTfidfEmbeddings
├── vectorstore.py       NumPy in-memory store + ChromaDB adapter
├── llm_providers.py      OpenAI / Anthropic / Mock generation
├── schemas.py           Pydantic request/response models
├── config.py            Settings from environment variables
└── sample_pdfs/
    └── company_handbook.pdf   Real test fixture (2-page PDF with policy Q&A)
tests/
└── test_all.py           14 tests covering the full pipeline end-to-end
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optionally add OPENAI_API_KEY
```

No API key? Leave `.env` blank — embeddings fall back to `LocalTfidfEmbeddings` and generation falls back to `MockProvider`, so the whole pipeline works offline.

## Run

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs**.

## Example usage

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@app/sample_pdfs/company_handbook.pdf"

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do employees get?"}'
```

```json
{
  "answer": "Based on the retrieved documents: [handbook_xxx p.1] Employee Handbook - Acme Robotics Inc. Vacation Policy: Full-time employees accrue 15 paid vacation days per year.",
  "sources": [
    {"doc_id": "handbook_xxx", "chunk_id": "handbook_xxx_0", "page": 1, "score": 0.5423, "text": "..."}
  ],
  "llm_provider": "mock",
  "embedding_provider": "local-tfidf-svd"
}
```

## Testing

```bash
python3 tests/test_all.py
```

14 tests, run and verified in development: real PDF text extraction, sentence-aware chunking, embedding generation, vector search ranking, and full ingest→retrieve→generate pipeline correctness (including two real bugs caught and fixed during development — see below).

## Bugs caught during development (kept here for transparency)

1. **SVD collapse on small corpora**: with very few chunks, `TruncatedSVD` was forced down to 1 dimension, making unrelated chunks look identically similar (cosine score of 1.0 for everything). Fixed by falling back to raw TF-IDF vectors when the corpus is too small for SVD to be meaningful.
2. **Chunk granularity**: an early test used a 500-char chunk size that merged an entire page into one chunk, defeating retrieval. Fixed by chunking at a size appropriate to document density.

## Known limitations / next steps

- `NumpyVectorStore` is in-memory only; swap in `ChromaVectorStore` (already implemented) for persistence across restarts.
- `LocalTfidfEmbeddings` is a lexical (not semantic) fallback — swap in OpenAI or HuggingFace embeddings for true semantic retrieval in production.
- No authentication on the API — add before exposing publicly.

## Tech stack

Python · LangChain · FastAPI · ChromaDB · scikit-learn · pypdf · Pydantic
