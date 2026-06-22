import sqlite3

conn = sqlite3.connect("emotion_history.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emotion TEXT,
    method TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully")