import psycopg2
import json
import os
from datetime import datetime

# Connection details from your .env or direct
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://user:password@localhost:5432/beauty_care")

conn = psycopg2.connect(POSTGRES_DSN)
cur = conn.cursor()

# Fetch all conversations
cur.execute("""
    SELECT session_id, role, message, escalated, created_at 
    FROM conversations 
    ORDER BY created_at DESC
""")

rows = cur.fetchall()
cur.close()
conn.close()

# Save to JSON
history = [
    {
        "session_id": row[0],
        "role": row[1],
        "message": row[2],
        "escalated": row[3],
        "timestamp": row[4].isoformat() if row[4] else None
    }
    for row in rows
]

with open("chat_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"✅ Exported {len(history)} messages to chat_history.json")