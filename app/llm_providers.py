"""
LLM provider abstraction for the RAG chain's generation step.

Same pattern as Project #1: OpenAI/Anthropic for production, plus a
deterministic MockProvider that extracts a grounded answer directly from
retrieved context, so the full RAG pipeline can be run and tested offline.
"""

from __future__ import annotations

from app.vectorstore import RetrievedChunk

RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the answer isn't in the context, say you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{r.chunk.doc_id} p.{r.chunk.page}] {r.chunk.text}" for r in retrieved)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


class BaseLLMProvider:
    name = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIProvider(BaseLLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from langchain_openai import ChatOpenAI  # lazy import

        self._client = ChatOpenAI(model=model, api_key=api_key, temperature=0)

    def generate(self, prompt: str) -> str:
        return self._client.invoke(prompt).content


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        from langchain_anthropic import ChatAnthropic  # lazy import

        self._client = ChatAnthropic(model=model, api_key=api_key, temperature=0)

    def generate(self, prompt: str) -> str:
        return self._client.invoke(prompt).content


class MockProvider(BaseLLMProvider):
    """
    Deterministic, offline generation: extracts the most relevant sentence(s)
    from the provided context rather than calling a real LLM. This lets the
    complete retrieve -> prompt -> generate pipeline run without an API key.
    """

    name = "mock"

    def generate(self, prompt: str) -> str:
        if "Context:\n\n\nQuestion" in prompt or "Context:\n\nQuestion" in prompt:
            return "I don't have enough information in the provided documents to answer that."

        context_part = prompt.split("Context:\n")[1].split("\n\nQuestion:")[0]
        question = prompt.split("Question: ")[1].split("\n\nAnswer:")[0]

        if not context_part.strip():
            return "I don't have enough information in the provided documents to answer that."

        # Naive extractive answer: return the context, prefixed for transparency.
        return f"Based on the retrieved documents: {context_part.strip()[:400]}"


def build_provider(provider_name: str, api_key: str | None, model: str) -> BaseLLMProvider:
    if provider_name == "openai":
        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini")
    if provider_name == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-6")
    return MockProvider()
