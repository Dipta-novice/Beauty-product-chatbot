"""
Extract chat history from SQLite database to JSON file.
Usage: python data_extraction.py
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict
import os


DB_PATH = "chat_history.db"


def get_connection() -> sqlite3.Connection:
    """Get SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn


def extract_all_chats() -> List[Dict]:
    """Extract ALL conversations from SQLite."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found: {DB_PATH}")
        return []
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, session_id, role, message, escalated, created_at
            FROM conversations
            ORDER BY created_at ASC
        """)
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    # Format as dictionaries
    chats = [
        {
            "id": row["id"],
            "session_id": row["session_id"],
            "role": row["role"],
            "message": row["message"],
            "escalated": bool(row["escalated"]),
            "timestamp": row["created_at"]
        }
        for row in rows
    ]
    
    return chats


def extract_by_session(session_id: str) -> List[Dict]:
    """Extract chat history for a specific session."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, session_id, role, message, escalated, created_at
            FROM conversations
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    chats = [
        {
            "id": row["id"],
            "session_id": row["session_id"],
            "role": row["role"],
            "message": row["message"],
            "escalated": bool(row["escalated"]),
            "timestamp": row["created_at"]
        }
        for row in rows
    ]
    
    return chats


def get_all_sessions() -> List[str]:
    """Get list of all unique session IDs."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT DISTINCT session_id FROM conversations ORDER BY session_id")
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    return [row[0] for row in rows]


def export_to_json(data: List[Dict], filename: str = "chat_history.json"):
    """Save chat history to JSON file."""
    output_path = filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(data)} messages to {output_path}")
    return output_path


def export_sessions_summary(filename: str = "sessions_summary.json"):
    """Export summary of all sessions."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                session_id,
                COUNT(*) as message_count,
                SUM(CASE WHEN role='user' THEN 1 ELSE 0 END) as user_messages,
                SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END) as assistant_messages,
                SUM(CASE WHEN escalated=1 THEN 1 ELSE 0 END) as escalated_count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time
            FROM conversations
            GROUP BY session_id
            ORDER BY start_time DESC
        """)
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    summary = [
        {
            "session_id": row["session_id"],
            "total_messages": row["message_count"],
            "user_messages": row["user_messages"],
            "assistant_messages": row["assistant_messages"],
            "escalations": row["escalated_count"],
            "start_time": row["start_time"],
            "end_time": row["end_time"]
        }
        for row in rows
    ]
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(summary)} session summaries to {filename}")
    return filename


def print_db_info():
    """Print database statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Total messages
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_msgs = cursor.fetchone()[0]
        
        # Sessions
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
        total_sessions = cursor.fetchone()[0]
        
        # User vs Assistant
        cursor.execute("SELECT role, COUNT(*) FROM conversations GROUP BY role")
        role_counts = cursor.fetchall()
        
        # Escalations
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE escalated=1")
        escalations = cursor.fetchone()[0]
        
    finally:
        conn.close()
    
    print("\n" + "="*60)
    print("📊 DATABASE STATISTICS")
    print("="*60)
    print(f"📁 Database file: {DB_PATH}")
    print(f"💬 Total messages: {total_msgs}")
    print(f"🔄 Unique sessions: {total_sessions}")
    
    for role, count in role_counts:
        print(f"  - {role}: {count}")
    
    print(f"⚠️  Escalations: {escalations}")
    print("="*60 + "\n")


def interactive_menu():
    """Interactive menu for data extraction."""
    while True:
        print("\n📊 CHAT HISTORY EXTRACTION MENU")
        print("="*50)
        print("1. Export ALL chat history to JSON")
        print("2. Export specific session to JSON")
        print("3. View all sessions summary")
        print("4. View database statistics")
        print("5. List all session IDs")
        print("6. Exit")
        print("="*50)
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == "1":
            all_chats = extract_all_chats()
            export_to_json(all_chats, "chat_history_all.json")
            
        elif choice == "2":
            sessions = get_all_sessions()
            if not sessions:
                print("❌ No sessions found!")
                continue
            
            print("\nAvailable sessions:")
            for i, sid in enumerate(sessions, 1):
                print(f"{i}. {sid[:20]}...")
            
            session_num = input("Enter session number: ").strip()
            try:
                session_id = sessions[int(session_num) - 1]
                chats = extract_by_session(session_id)
                export_to_json(chats, f"chat_{session_id[:8]}.json")
            except (ValueError, IndexError):
                print("❌ Invalid selection!")
        
        elif choice == "3":
            export_sessions_summary("sessions_summary.json")
            
        elif choice == "4":
            print_db_info()
        
        elif choice == "5":
            sessions = get_all_sessions()
            print("\n📋 All Session IDs:")
            for i, sid in enumerate(sessions, 1):
                print(f"{i}. {sid}")
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option!")


if __name__ == "__main__":
    print("🗄️  BEAUTY CHATBOT — DATA EXTRACTION TOOL\n")
    
    # Show menu
    interactive_menu()
