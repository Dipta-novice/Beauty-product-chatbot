import streamlit as st
import uuid
from data_enricher import enrich_data
from document_builder import build_documents
from rag_chain import build_rag_chain
from chatbot import process_query
from db_manager import get_connection, init_db, fetch_history

st.set_page_config(page_title="💄 Beauty Advisor", layout="wide")

# ── Init session state ──────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_system():
    """Load the hybrid RAG system."""
    # Generate enriched data
    enrich_data("cosmetics.csv", "cosmetics_enriched.csv")
    docs = build_documents("cosmetics_enriched.csv")

    chain = build_rag_chain(docs)
    
    # Database
    conn = get_connection()
    init_db(conn)
    
    return chain, conn, docs, None

chain, db_conn, docs, _ = load_system()

# ── UI ──────────────────────────────────────────────────────────
st.title("💄 Personal Care Product Advisor")
st.caption("Powered by hybrid retrieval with query rewriting and reranking.")

# Sidebar — Past conversation history
with st.sidebar:
    st.header("🕘 Conversation History")
    history = fetch_history(db_conn, st.session_state.session_id)
    
    if history:
        st.markdown("**Recent chats:**")
        for h in history[-10:]:
            role_display = h.get('role', 'user').title()
            content_preview = h.get('content', 'No content')[:60] + "..."
            with st.expander(f"{role_display}: {content_preview}"):
                st.write(h.get('content', 'No content'))
    else:
        st.info("No chat history yet. Start chatting!")

# Clear chat button
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about a product, ingredient, or skin concern..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching products..."):
            chat_history = st.session_state.messages[:-1]
            
            try:
                response = process_query(
                    prompt, chain, chat_history, db_conn, st.session_state.session_id
                )
            except Exception as e:
                response = f"Sorry, something went wrong: {str(e)}"
                st.error(f"Error: {str(e)}")
        
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

