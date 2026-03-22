from db_manager import save_message
from rag_chain import run_query
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any

def process_query(query: str, chain: Any, chat_history: List, db_conn: Any, session_id: str) -> str:
    """LLM-driven query process with proper message formatting.
    
    - Save user message to DB
    - Convert chat_history to LangChain messages  
    - Run RAG+LLM chain
    - Save assistant response
    """
    # Save user message
    save_message(db_conn, session_id, "user", query)
    
    # ✅ Convert chat_history to proper LangChain messages
    formatted_history = convert_chat_history(chat_history)
    
    # Run RAG chain
    result = run_query(chain, query, formatted_history, docs=None, vs=None)
    answer = result.get("answer", "Sorry, I could not generate an answer right now.")
    
    # ✅ Ensure answer is a string (convert AIMessage or other objects)
    if hasattr(answer, 'content'):
        answer = answer.content
    answer = str(answer).strip()
    
    # Detect escalations (support contact)
    escalated = "please contact our support team" in answer.lower()
    save_message(db_conn, session_id, "assistant", answer, escalated=escalated)
    
    return answer

def convert_chat_history(chat_history: List) -> List:
    """Convert DB chat_history to LangChain message objects."""
    messages = []
    for i, entry in enumerate(chat_history):
        if isinstance(entry, dict):
            role = entry.get('role', 'user')
            content = entry.get('content', '')
        elif isinstance(entry, tuple) and len(entry) == 2:
            role, content = entry
        elif isinstance(entry, str):
            content = entry
            role = 'user' if i % 2 == 0 else 'assistant'
        else:
            continue
            
        if role.lower() in ['user', 'human', 'question']:
            messages.append(HumanMessage(content=content))
        elif role.lower() in ['assistant', 'ai', 'answer']:
            messages.append(AIMessage(content=content))
    
    return messages
