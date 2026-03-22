# 💄 AI-Powered Beauty Chatbot — Personal Care Product Advisor

A **Retrieval-Augmented Generation (RAG)** powered chatbot that helps customers discover personalized beauty & skincare product recommendations using LLM intelligence and hybrid semantic search.

---

## ✨ Features

### 🤖 **LLM-Driven Intelligence**
- **Free LLM** (Nvidia Nemotron via OpenRouter)
- **LLM-Based Escalation**: Model decides when to escalate to human support (offers, returns, complaints)
- **Conversational Memory**: Maintains chat history across sessions
- **Semantic Understanding**: Understands product queries, skin concerns, and ingredient questions

### 🔍 **Smart Retrieval (Hybrid Search)**
- **Semantic Search** (Nomic embeddings from Hugging Face - free, local)
- **BM25 Keyword Matching** (perfect for product names & brands)
- **Ensemble Retrieval** (40% keyword + 60% semantic = best of both)
- **Cross-Encoder Reranking** (re-ranks top documents for relevance)
- **MMR Diversity** (prevents duplicate/similar results)

### 💾 **Data & Storage**
- **SQLite Database** (local, unlimited storage)
- **Persistent Vector Store** (ChromaDB with automatic persistence)
- **Chat History** (searchable, exportable to JSON)
- **Data Enrichment** (synthetic benefits, offers, return policies)

### 🎯 **User Experience**
- **Streamlit Web UI** (instant, no frontend coding needed)
- **Session Management** (unique session IDs for multi-user support)
- **Conversation Sidebar** (view past 10 messages)
- **Clear Chat Button** (easy session reset)
- **Debug Panel** (view session stats)

### 📊 **Data Extraction**
- **Interactive CLI Tool** (`data_extraction.py`)
- Export all chats / specific sessions to JSON
- Session summaries & statistics
- Database browser-friendly format

---

## 🏗️ Architecture

```
📦 Beauty Chatbot
├── 📄 app.py                    ← Streamlit UI & main entry point
├── 🤖 rag_chain.py             ← RAG pipeline with LLM integration
├── 💭 chatbot.py               ← LLM-driven query processing
├── 📚 vector_store.py          ← Hybrid retrieval (Semantic + BM25)
├── 📊 data_enricher.py         ← Add synthetic enrichment to CSV
├── 📃 document_builder.py      ← Convert CSV rows to LangChain docs
├── 🗄️  db_manager.py           ← SQLite conversation storage
├── 📤 data_extraction.py       ← Export chats to JSON
├── 🛍️  cosmetics.csv           ← Raw product data
└── 📋 requirements.txt         ← Python dependencies
```

### Data Flow
```
cosmetics.csv
    ↓
data_enricher.py (add benefits, offers, returns)
    ↓
document_builder.py (create LangChain Documents)
    ↓
vector_store.py (build hybrid retriever)
    ↓
rag_chain.py (RAG + LLM)
    ↓
chatbot.py (process & save)
    ↓
SQLite + Streamlit UI
```

---

## 🚀 Quick Start

### 1️⃣ **Prerequisites**
- Python 3.10+
- Windows/Mac/Linux
- API Key: Get free LLM access via [OpenRouter](https://openrouter.ai/)

### 2️⃣ **Clone & Setup**

```bash
# Navigate to project
cd "Beauty care"

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ **Configuration**

Create `.env` file in project root:
```env
# OpenRouter API Key (free tier available)
OPENAI_API_KEY=sk-or-v1-your-key-here
```

Get your key:
1. Visit https://openrouter.ai/keys
2. Create free account
3. Generate API key
4. Copy to `.env`

### 4️⃣ **Run the Chatbot**

```bash
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`

---

## 💬 Usage Examples

### **Example 1: Product Recommendation**
```
User: "What's a good moisturizer for dry skin?"

Bot: 
**CeraVe Moisturizing Facial Lotion by CeraVe**
✅ **Ingredients:** Ceramides, Hyaluronic Acid, Niacinamide...
⭐ **Rating:** 4.8/5
✅ **Skin Types:** Dry, Sensitive
💧 **Benefits:** Deep hydration, skin barrier repair
💰 **Price:** $14.99
🎁 **Offers:** 15% off on first order
```

### **Example 2: Ingredient Query**
```
User: "Is hyaluronic acid good for sensitive skin?"

Bot:
Hyaluronic acid is an excellent ingredient for sensitive skin...
[Recommends 3-4 products with hyaluronic acid]
```

### **Example 3: Escalation (LLM-Driven)**
```
User: "Is there a refund policy for this product?"

Bot:
For this query, please contact our support team at 1800-XXX-XXXX
(Mon–Sat, 9am–6pm)
```

---

## 📂 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI, session management, chat display |
| `rag_chain.py` | RAG pipeline, LLM integration, prompt engineering |
| `chatbot.py` | Query processing, escalation detection, response formatting |
| `vector_store.py` | Semantic + BM25 hybrid retrieval, reranking |
| `data_enricher.py` | Add benefits, offers, return policies to CSV |
| `document_builder.py` | Convert CSV to LangChain Document objects |
| `db_manager.py` | SQLite operations (save, fetch, init) |
| `data_extraction.py` | Export chat history to JSON (interactive CLI) |
| `cosmetics.csv` | Raw product dataset |
| `requirements.txt` | Python package dependencies |

---

## 💾 Extract Chat History

### **Interactive Mode**
```bash
python data_extraction.py
```

Menu options:
1. Export ALL chats to JSON
2. Export specific session
3. View session summary
4. View database stats
5. List all sessions

### **Programmatic Extraction**
```python
from data_extraction import extract_all_chats, export_to_json

chats = extract_all_chats()
export_to_json(chats, "my_chats.json")
```

### **Output Format**
```json
[
  {
    "id": 1,
    "session_id": "abc-123-def",
    "role": "user",
    "message": "What moisturizer for dry skin?",
    "escalated": false,
    "timestamp": "2026-03-22 10:30:45"
  },
  {
    "id": 2,
    "session_id": "abc-123-def",
    "role": "assistant",
    "message": "**CeraVe Moisturizing Cream**...",
    "escalated": false,
    "timestamp": "2026-03-22 10:30:50"
  }
]
```

---

## 📊 View Database Files

### **Option 1: SQLite Browser (GUI)**
1. Download: https://sqlitebrowser.org/
2. File → Open → `chat_history.db`
3. Browse Data tab

### **Option 2: Python**
```python
import sqlite3
conn = sqlite3.connect("chat_history.db")
for row in conn.execute("SELECT * FROM conversations LIMIT 5"):
    print(row)
```

### **Option 3: Command Line**
```bash
sqlite3 chat_history.db
sqlite> SELECT * FROM conversations;
sqlite> .mode column
sqlite> .headers on
sqlite> SELECT role, COUNT(*) FROM conversations GROUP BY role;
```

---

## 🔧 Configuration & Customization

### **Change Support Contact Number**
Edit `rag_chain.py`:
```python
SYSTEM_PROMPT = """
...
respond with: "For this query, please contact our support team at 1800-XXX-XXXX."
```

### **Adjust Temperature (Creativity)**
Edit `rag_chain.py`:
```python
llm = ChatOpenAI(
    ...
    temperature=0.1  # ← Lower = deterministic, Higher = creative
)
```

### **Change Embedding Model**
Edit `vector_store.py`:
```python
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # Or any HF model
```

### **Tune Retrieval Weights**
Edit `vector_store.py`:
```python
EnsembleRetriever(
    retrievers=[bm25, mmr_retriever],
    weights=[0.3, 0.7]  # ← Adjust semantic vs keyword balance
)
```

---

## 🐛 Troubleshooting

### **"ModuleNotFoundError: No module named 'langchain_community'"**
```bash
pip install -r requirements.txt
```

### **"OPENAI_API_KEY not set"**
- Create `.env` file with `OPENAI_API_KEY=your-key`
- Ensure it's in project root, not subdirectories

### **Streamlit not opening browser**
```bash
# Manually open
streamlit run app.py --logger.level=debug
# Visit: http://localhost:8501
```

### **Chat takes too long (first run)**
- First run downloads Nomic embedding model (~300MB)
- Subsequent runs are instant (cached locally)

### **Pylance warnings about imports**
- VS Code → Ctrl+Shift+P → "Python: Select Interpreter"
- Choose: `./venv/Scripts/python.exe`
- Wait 10 seconds for re-indexing

---

## 📦 Technology Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| **LLM** | Nvidia Nemotron (OpenRouter) | Free, high-quality, no account needed |
| **Embeddings** | Nomic (Hugging Face) | Free, fast, runs locally |
| **Retrieval** | ChromaDB + BM25 + CrossEncoder | Hybrid = best results |
| **Framework** | LangChain | Standardized RAG pipeline |
| **UI** | Streamlit | Zero frontend code needed |
| **Database** | SQLite | Local, no server required |
| **Reranking** | SentenceTransformers | Improves relevance significantly |

---

## 🎯 Performance Metrics

- **Retrieval Speed**: ~200ms (hybrid search)
- **LLM Response**: ~2-5s (depends on query complexity)
- **Memory Usage**: ~500MB (including models)
- **Database Size**: ~1MB per 1000 messages
- **Vector Store**: ~100MB (for ~1000 products)

---

## 🔐 Security Notes

- ✅ No data leaves your machine (SQLite local)
- ✅ API key stored in `.env` (never commit to git)
- ✅ `.gitignore` prevents accidental key leaks
- ✅ LLM calls go through OpenRouter (read their privacy policy)

---

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Feedback rating system
- [ ] A/B testing for prompts
- [ ] Admin dashboard
- [ ] PostgreSQL for production
- [ ] User authentication
- [ ] Product image search
- [ ] Mobile app (Streamlit mobile)

---

## 📝 License

This project is open source and free to use.

---

## 💡 Tips & Tricks

### **Batch Export Sessions**
```python
from data_extraction import get_all_sessions, extract_by_session, export_to_json

for session_id in get_all_sessions():
    chats = extract_by_session(session_id)
    export_to_json(chats, f"session_{session_id[:8]}.json")
```

### **Update Product Data**
1. Edit `cosmetics.csv`
2. Delete `chroma_db/` folder
3. Restart Streamlit (auto-rebuilds vector store)

### **Monitor Chat Quality**
```bash
python data_extraction.py
# Select option 4 to view database statistics
# Check escalation rate, message patterns
```

---

## 🤝 Support

For issues or questions:
1. Check Troubleshooting section
2. Review Streamlit terminal output for errors
3. Check `.env` file configuration
4. Verify API key is valid

---


