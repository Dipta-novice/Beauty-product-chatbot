from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_QUERY_REWRITER: ChatOpenAI | None = None
_CROSS_ENCODER: CrossEncoder | None = None


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = text.replace("’", "'").replace("™", " ").replace("®", " ")
    text = re.sub(r"sk[\s\-]?ii", "sk ii", text)
    text = re.sub(r"[^a-z0-9$.\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_json_block(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(match.group(0))


def _get_query_rewriter() -> ChatOpenAI:
    global _QUERY_REWRITER
    if _QUERY_REWRITER is None:
        _QUERY_REWRITER = ChatOpenAI(
            model=REWRITE_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
        )
    return _QUERY_REWRITER


def _get_cross_encoder() -> CrossEncoder:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder(RERANKER_MODEL)
    return _CROSS_ENCODER


def _fallback_rewrite(query: str) -> Dict[str, Any]:
    normalized = _normalize_text(query)
    quoted_match = re.search(r"['\"]([^'\"]+)['\"]", query)
    quoted_product = _normalize_text(quoted_match.group(1)) if quoted_match else ""

    price_match = re.search(r"(?:under|below|less than)\s*\$?\s*(\d+(?:\.\d+)?)", normalized)
    max_price = float(price_match.group(1)) if price_match else None

    return {
        "original_query": query,
        "normalized_query": normalized,
        "rewritten_query": normalized,
        "product_name": quoted_product,
        "brand": "",
        "category": "",
        "skin_types": [],
        "max_price": max_price,
        "intent": "product_lookup",
    }


def rewrite_query(query: str) -> Dict[str, Any]:
    prompt = f"""
You rewrite beauty-product search queries for retrieval.
Return only valid JSON with this exact schema:
{{
  "rewritten_query": "string",
  "product_name": "string",
  "brand": "string",
  "category": "string",
  "skin_types": ["string"],
  "max_price": number or null,
  "intent": "string"
}}

Rules:
- Preserve exact product names when present.
- Normalize punctuation variants like SK-II, SPF, CC+ and quoted product names.
- Expand the query so retrieval works better on catalog rows.
- Extract brand/category/skin_types/max_price when possible.
- Use concise retrieval-friendly wording.
- If a field is unknown, use empty string, [] or null.

User query: {query}
""".strip()

    try:
        response = _get_query_rewriter().invoke(prompt)
        payload = _extract_json_block(response.content if hasattr(response, "content") else str(response))
        rewritten_query = str(payload.get("rewritten_query") or query).strip()
        return {
            "original_query": query,
            "normalized_query": _normalize_text(query),
            "rewritten_query": rewritten_query,
            "product_name": _normalize_text(payload.get("product_name")),
            "brand": _normalize_text(payload.get("brand")),
            "category": _normalize_text(payload.get("category")),
            "skin_types": [_normalize_text(item) for item in payload.get("skin_types", []) if _normalize_text(item)],
            "max_price": _parse_float(payload.get("max_price")),
            "intent": str(payload.get("intent") or "product_lookup").strip(),
        }
    except Exception:
        return _fallback_rewrite(query)


class HybridProductRetriever:
    def __init__(self, docs: List[Document], k: int = 8):
        self.docs = docs
        self.k = k
        self._indexed_docs = [self._prepare_doc(doc) for doc in docs]

        bm25_docs = [
            Document(
                page_content=indexed["search_text"],
                metadata={**indexed["doc"].metadata, "_doc_index": idx},
            )
            for idx, indexed in enumerate(self._indexed_docs)
        ]
        self.bm25 = BM25Retriever.from_documents(bm25_docs)
        self.bm25.k = max(k * 2, 12)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(
            indexed["search_text"] for indexed in self._indexed_docs
        )

    def _prepare_doc(self, doc: Document) -> Dict[str, Any]:
        metadata = dict(doc.metadata)
        name = str(metadata.get("name") or metadata.get("product") or "")
        brand = str(metadata.get("brand") or "")
        category = str(metadata.get("category") or metadata.get("label") or "")
        ingredients = str(metadata.get("ingredients") or "")
        benefits = str(metadata.get("benefits") or "")
        skin_types = " ".join(metadata.get("skin_types", []))

        search_text = " ".join(
            part
            for part in [
                _normalize_text(name),
                _normalize_text(brand),
                _normalize_text(category),
                _normalize_text(ingredients),
                _normalize_text(benefits),
                _normalize_text(skin_types),
                _normalize_text(doc.page_content),
            ]
            if part
        )

        return {
            "doc": doc,
            "search_text": search_text,
            "name_norm": _normalize_text(name),
            "brand_norm": _normalize_text(brand),
            "category_norm": _normalize_text(category),
            "skin_types": {value.lower() for value in metadata.get("skin_types", [])},
            "price": metadata.get("price"),
            "rank": metadata.get("rank", 0.0),
        }

    def _score_doc(
        self,
        doc_index: int,
        rewrite: Dict[str, Any],
        bm25_rank: Dict[int, int],
        tfidf_scores,
    ) -> float:
        indexed = self._indexed_docs[doc_index]
        score = 0.0

        if doc_index in bm25_rank:
            score += max(0.0, 4.0 - (0.25 * bm25_rank[doc_index]))

        score += float(tfidf_scores[doc_index]) * 5.0

        product_name = rewrite.get("product_name", "")
        brand = rewrite.get("brand", "")
        category = rewrite.get("category", "")
        query_norm = rewrite["normalized_query"]

        if product_name:
            if product_name == indexed["name_norm"]:
                score += 12.0
            elif product_name in indexed["name_norm"]:
                score += 8.0

        if brand:
            if brand == indexed["brand_norm"]:
                score += 4.0
            elif brand in indexed["brand_norm"]:
                score += 2.0

        if category and category == indexed["category_norm"]:
            score += 2.0

        if rewrite["skin_types"]:
            score += len(indexed["skin_types"].intersection(rewrite["skin_types"])) * 1.5

        price = indexed["price"]
        max_price = rewrite["max_price"]
        if max_price is not None and isinstance(price, (int, float)):
            if price <= max_price:
                score += 2.0
            else:
                score -= min(3.0, (price - max_price) / 25.0)

        score += SequenceMatcher(None, query_norm, indexed["name_norm"]).ratio() * 3.0

        rank = indexed["rank"]
        if isinstance(rank, (int, float)):
            score += float(rank) / 5.0

        return score

    def retrieve(self, query: str, k: int | None = None) -> Tuple[List[Document], Dict[str, Any]]:
        top_k = k or self.k
        rewrite = rewrite_query(query)

        bm25_results = self.bm25.invoke(rewrite["rewritten_query"])
        bm25_rank: Dict[int, int] = {}
        for rank, doc in enumerate(bm25_results):
            doc_idx = doc.metadata.get("_doc_index")
            if not isinstance(doc_idx, int):
                continue
            if doc_idx not in bm25_rank:
                bm25_rank[doc_idx] = rank

        query_vector = self.vectorizer.transform([rewrite["rewritten_query"]])
        tfidf_scores = (self.doc_vectors @ query_vector.T).toarray().ravel()

        scored_docs = []
        for idx in range(len(self._indexed_docs)):
            score = self._score_doc(idx, rewrite, bm25_rank, tfidf_scores)
            if score > 0:
                scored_docs.append((score, self._indexed_docs[idx]["doc"]))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]], rewrite

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs, _ = self.retrieve(query, self.k)
        return docs

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


def build_hybrid_retriever(docs: List[Document], k: int = 8) -> HybridProductRetriever:
    return HybridProductRetriever(docs, k=k)


def rerank(query: str, docs: List[Document], top_k: int = 5) -> Tuple[List[Document], Dict[str, Any]]:
    rewrite = rewrite_query(query)
    if not docs:
        return [], rewrite

    cross_encoder = _get_cross_encoder()
    pairs = [(rewrite["rewritten_query"], doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    ranked: List[Tuple[float, Document]] = []
    for score, doc in zip(scores, docs):
        boost = 0.0
        product_name = rewrite.get("product_name", "")
        brand = rewrite.get("brand", "")
        name_norm = _normalize_text(doc.metadata.get("name") or doc.metadata.get("product"))
        brand_norm = _normalize_text(doc.metadata.get("brand"))

        if product_name and product_name == name_norm:
            boost += 0.75
        elif product_name and product_name in name_norm:
            boost += 0.35

        if brand and brand == brand_norm:
            boost += 0.2

        ranked.append((float(score) + boost, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]], rewrite
