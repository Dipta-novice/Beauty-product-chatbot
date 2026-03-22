from document_builder import build_documents
from langchain_community.retrievers import BM25Retriever

# Load your exact data
docs = build_documents("cosmetics_enriched.csv")
bm25 = BM25Retriever.from_documents(docs)
bm25.k = 3

# Test SK-II specifically
results = bm25.invoke("SK-II Facial Treatment Essence")
print("FOUND:", len([d for d in results if "SK-II" in d.page_content]))
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.metadata.get('brand', 'N/A')} - {doc.page_content[:100]}")
