"""
Embedding provider abstraction.

- OpenAIEmbeddings: production path via langchain_openai (requires API key + network).
- HuggingFaceEmbeddings: production path via sentence-transformers (requires the
  model weights to be downloaded once; requires network on first run).
- LocalTfidfEmbeddings: a real, fully offline embedding provider using
  scikit-learn's TF-IDF + SVD (a form of Latent Semantic Analysis) to produce
  dense vectors. This is what lets the whole RAG pipeline be run and tested
  in environments with no network access and no API key.

All providers implement embed_documents(texts) -> list[vector] and
embed_query(text) -> vector, so vectorstore.py is provider-agnostic.
"""

from __future__ import annotations

import numpy as np


class BaseEmbeddingProvider:
    name = "base"
    dim: int = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class OpenAIEmbeddings(BaseEmbeddingProvider):
    """Production embeddings via OpenAI. Requires OPENAI_API_KEY + network."""

    name = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings  # lazy import

        self._client = LCOpenAIEmbeddings(model=model, api_key=api_key)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array(self._client.embed_documents(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(self._client.embed_query(text))


class HuggingFaceEmbeddings(BaseEmbeddingProvider):
    """Production embeddings via a local sentence-transformers model. Requires model download."""

    name = "huggingface"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array(self._model.encode(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(self._model.encode([text])[0])


class LocalTfidfEmbeddings(BaseEmbeddingProvider):
    """
    Fully offline embedding provider: TF-IDF (char n-grams, robust to plurals/typos)
    reduced to a dense vector space via Truncated SVD (i.e. LSA).

    Must be fit on the document corpus first (via embed_documents), then embed_query
    projects new text into the same learned space.
    """

    name = "local-tfidf-svd"

    def __init__(self, n_components: int = 64):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.n_components = n_components
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self._svd = None
        self._use_svd = False
        self._fitted = False
        self.dim = n_components

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD

        tfidf = self._vectorizer.fit_transform(texts)

        # SVD only helps (and is only numerically stable) with enough documents
        # relative to the target dimensionality. For small corpora, skip it and
        # use the dense TF-IDF vectors directly - degenerate 1-D SVD projections
        # would otherwise make unrelated chunks look identically similar.
        max_safe_components = min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if max_safe_components >= 8:
            self._svd = TruncatedSVD(n_components=max_safe_components, random_state=42)
            vectors = self._svd.fit_transform(tfidf)
            self._use_svd = True
            self.dim = vectors.shape[1]
        else:
            self._svd = None
            self._use_svd = False
            vectors = tfidf.toarray()
            self.dim = vectors.shape[1]

        self._fitted = True
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LocalTfidfEmbeddings must embed a document corpus before embedding queries.")
        tfidf = self._vectorizer.transform([text])
        if self._use_svd:
            return self._svd.transform(tfidf)[0]
        return tfidf.toarray()[0]


def build_embedding_provider(provider_name: str, api_key: str | None = None) -> BaseEmbeddingProvider:
    if provider_name == "openai":
        return OpenAIEmbeddings(api_key=api_key)
    if provider_name == "huggingface":
        return HuggingFaceEmbeddings()
    return LocalTfidfEmbeddings()
