"""Tool: search the support knowledge base (mock RAG retrieval).

The retrieval strategy is pluggable:
  * LexicalRetriever   - deterministic IDF-weighted token overlap, no network.
                         The default, so the agent runs anywhere with zero setup.
  * EmbeddingRetriever - OpenAI embeddings + cosine similarity, enabled with
                         RETRIEVER=embedding for semantically richer search.

Swapping or adding a retriever touches neither the tool nor the agent, which is
the main point: the KB is the one place real production systems most need to
evolve (bigger corpus, hybrid search, reranking).
"""
from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..config import DATA_DIR, settings
from .registry import Tool

# Tokenize latin words and Thai characters. Note: Thai is written without spaces,
# so a whole Thai phrase collapses into one token and the KB corpus here is English;
# genuine cross-lingual matching relies on the query being in English (the mock's
# _kb_query does this) or on the embedding retriever. This mainly keeps Thai input
# from breaking tokenization rather than making Thai keyword search effective.
_WORD = re.compile(r"[a-zA-Z0-9฀-๿]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _doc_text(doc: dict[str, Any]) -> str:
    return f"{doc.get('title', '')} {doc.get('body', '')} {' '.join(doc.get('tags', []))}"


class Retriever(ABC):
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        ...


class LexicalRetriever(Retriever):
    """IDF-weighted token-overlap scoring, normalised to [0, 1]."""

    def __init__(self, docs: list[dict[str, Any]]):
        super().__init__(docs)
        self._doc_tokens = [set(_tokens(_doc_text(d))) for d in docs]
        n = len(docs)
        df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((n + 1) / (c + 0.5)) + 1.0 for t, c in df.items()}

    def search(self, query: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        q = set(_tokens(query))
        denom = sum(self._idf.get(t, 1.0) for t in q) or 1.0
        scored: list[tuple[dict[str, Any], float]] = []
        for doc, toks in zip(self.docs, self._doc_tokens):
            overlap = q & toks
            raw = sum(self._idf.get(t, 1.0) for t in overlap)
            scored.append((doc, raw / denom))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(d, round(s, 3)) for d, s in scored[:top_k] if s > 0]


class EmbeddingRetriever(Retriever):
    """Semantic search via OpenAI embeddings and cosine similarity."""

    def __init__(self, docs: list[dict[str, Any]]):
        super().__init__(docs)
        from openai import OpenAI  # lazy: the lexical path needs no SDK/key

        self._client = OpenAI()
        self._doc_vecs = self._embed([_doc_text(d) for d in docs])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=settings.embedding_model, input=texts)
        return [item.embedding for item in resp.data]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        qv = self._embed([query])[0]
        scored = [(d, self._cosine(qv, dv)) for d, dv in zip(self.docs, self._doc_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(d, round(max(s, 0.0), 3)) for d, s in scored[:top_k]]


def build_retriever(docs: list[dict[str, Any]], offline: bool = False) -> Retriever:
    use_embedding = (
        settings.retriever == "embedding"
        and settings.has_api_key
        and not settings.use_mock
        and not offline  # --mock / offline always forces the no-network retriever
    )
    return EmbeddingRetriever(docs) if use_embedding else LexicalRetriever(docs)


class KnowledgeBaseTool(Tool):
    name = "search_knowledge_base"
    description = (
        "Search the support knowledge base (FAQs, help docs, runbooks) for "
        "articles relevant to the customer's problem. Returns the most relevant "
        "documents with a relevance score; use the content to craft the reply."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused search query describing the issue.",
            },
            "top_k": {
                "type": "integer",
                "description": "How many documents to return (default 3).",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    def __init__(self, source: Path | None = None, offline: bool = False):
        path = source or (DATA_DIR / "knowledge_base.json")
        self._docs: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        self._retriever = build_retriever(self._docs, offline=offline)

    def run(self, query: str, top_k: int = 3) -> dict[str, Any]:
        hits = self._retriever.search(query, top_k)
        return {
            "query": query,
            "results": [
                {
                    "doc_id": d["id"],
                    "title": d["title"],
                    "snippet": d["body"][:240],
                    "relevance": score,
                }
                for d, score in hits
            ],
        }
