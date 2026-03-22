from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from vector_store import build_hybrid_retriever, rerank

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SUPPORT_MESSAGE = "For queries regarding offers, returns, or shipping, please contact our support team at 1800-XXX-XXXX."

SYSTEM_PROMPT = """You are a cosmetics store assistant.

Rules:
- Answer only from the retrieved product context.
- If the escalation classifier has determined the user is asking about offers, discounts, returns, complaints, refund policy, shipping, delivery, or free shipping, reply exactly:
  "{support_message}"
- If the user asks for a specific product and it exists in the context, prioritize that exact product.
- If the answer is not supported by the context, say that the product or detail is not available in the database.
- Keep answers concise, factual, and friendly.

When relevant, include:
- Product name and brand
- Price
- Rating
- Skin type suitability
- Benefits
- Ingredients

Retrieved context:
{context}
"""


def _build_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )


def _should_escalate(query: str, classifier_llm: ChatOpenAI) -> bool:
    prompt = f"""
Classify this customer message for a cosmetics support chatbot.

Return only YES or NO.

Return YES if the query is mainly about:
- offers
- discounts
- coupons
- deals
- returns
- refunds
- exchanges
- shipping
- delivery status
- complaints

Return NO for normal product discovery, ingredients, skincare advice, or suitability questions.

User query: {query}
""".strip()

    try:
        result = classifier_llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        return text.strip().upper().startswith("YES")
    except Exception:
        return False


def _format_docs(docs: List[Any]) -> str:
    if not docs:
        return "No matching products found."

    sections = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        skin_types = ", ".join(metadata.get("skin_types", [])) or "Not specified"
        section = (
            f"[Result {index}]\n"
            f"Name: {metadata.get('name', 'Unknown')}\n"
            f"Brand: {metadata.get('brand', 'Unknown')}\n"
            f"Category: {metadata.get('category', 'Unknown')}\n"
            f"Price: ${metadata.get('price', 'N/A')}\n"
            f"Rating: {metadata.get('rank', 'N/A')}/5\n"
            f"Skin types: {skin_types}\n"
            f"Benefits: {metadata.get('benefits', 'N/A')}\n"
            f"Offers: {metadata.get('offers', 'N/A')}\n"
            f"Return policy: {metadata.get('return_policy', 'N/A')}\n"
            f"Ingredients: {metadata.get('ingredients', 'N/A')}\n"
        )
        sections.append(section)
    return "\n".join(sections)


def build_rag_chain(docs: List, vs=None):
    retriever = build_hybrid_retriever(docs)
    llm = _build_llm(temperature=0.1)
    classifier_llm = _build_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    answer_chain = prompt | llm | StrOutputParser()

    return {
        "retriever": retriever,
        "answer_chain": answer_chain,
        "classifier_llm": classifier_llm,
    }


def run_query(chain: Dict[str, Any], query: str, chat_history: List[Any], docs=None, vs=None) -> Dict[str, Any]:
    if _should_escalate(query, chain["classifier_llm"]):
        return {"answer": SUPPORT_MESSAGE, "source_documents": []}

    retriever = chain["retriever"]
    answer_chain = chain["answer_chain"]

    retrieved_docs, rewrite = retriever.retrieve(query, k=10)
    reranked_docs, _ = rerank(query, retrieved_docs, top_k=5)
    context = _format_docs(reranked_docs)

    formatted_history = []
    for message in chat_history:
        if isinstance(message, (HumanMessage, AIMessage)):
            formatted_history.append(message)

    answer = answer_chain.invoke(
        {
            "support_message": SUPPORT_MESSAGE,
            "context": context,
            "question": query,
            "chat_history": formatted_history,
        }
    )

    return {
        "answer": answer,
        "source_documents": reranked_docs,
        "query_rewrite": rewrite,
    }
