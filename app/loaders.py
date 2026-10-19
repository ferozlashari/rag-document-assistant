"""
Document loading and chunking.

Supports PDF (via pypdf) and plain text files. Chunking uses a sliding
window over sentences so chunks don't cut sentences in half, with overlap
to preserve context across chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    page: int | None = None


def load_text_from_pdf(path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) tuples."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages


def load_text_from_file(path: str) -> list[tuple[int, str]]:
    """Load a plain .txt file as a single 'page'."""
    with open(path, "r", encoding="utf-8") as f:
        return [(1, f.read())]


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_pages(
    doc_id: str,
    pages: list[tuple[int, str]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Chunk page text into ~chunk_size-character windows with overlap, on sentence boundaries."""
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_num, page_text in pages:
        sentences = _split_sentences(page_text)
        if not sentences:
            continue

        current: list[str] = []
        current_len = 0

        def flush():
            nonlocal chunk_index, current, current_len
            if current:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(
                        Chunk(doc_id=doc_id, chunk_id=f"{doc_id}_{chunk_index}", text=chunk_text, page=page_num)
                    )
                    chunk_index += 1

        for sentence in sentences:
            if current_len + len(sentence) > chunk_size and current:
                flush()
                # keep overlap: retain trailing sentences up to `overlap` chars
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                current = overlap_sentences
                current_len = overlap_len

            current.append(sentence)
            current_len += len(sentence)

        flush()

    return chunks


def load_and_chunk(path: str, doc_id: str, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    if path.lower().endswith(".pdf"):
        pages = load_text_from_pdf(path)
    else:
        pages = load_text_from_file(path)
    return chunk_pages(doc_id, pages, chunk_size=chunk_size, overlap=overlap)
