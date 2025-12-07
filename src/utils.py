import sqlite3
import datetime
import pandas as pd
import os

DB_NAME = "documind.db"

# --- DATABASE FUNCTIONS ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  upload_date TIMESTAMP,
                  filename TEXT,
                  file_blob BLOB,
                  file_type TEXT,
                  category TEXT,
                  confidence REAL,
                  extracted_text TEXT,
                  summary TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(uploaded_file, category, confidence, text, summary):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        current_time = datetime.datetime.now()
        c.execute('''INSERT INTO documents 
                     (upload_date, filename, file_blob, file_type, category, confidence, extracted_text, summary)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                  (current_time, uploaded_file.name, file_bytes, uploaded_file.type, category, confidence, text, summary))
        conn.commit()
        conn.close()
        return "✅ Document saved to Database!"
    except Exception as e:
        return f"❌ DB Error: {e}"

def get_db_history():
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT id, upload_date, filename, category, confidence, summary FROM documents ORDER BY upload_date DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def delete_db_entries(ids_to_delete):
    """Deletes rows from the database based on a list of IDs."""
    try:
        if not ids_to_delete: return False
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # Safe parameterized query
        query = f"DELETE FROM documents WHERE id IN ({','.join(['?']*len(ids_to_delete))})"
        c.execute(query, ids_to_delete)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Delete Error: {e}")
        return False

# --- TEXT METRIC FUNCTIONS (Restored) ---
def calculate_text_metrics(text):
    if not text: return None
    words = text.split()
    sentences = text.split('.')
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_len = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Avg Word Length": round(avg_word_len, 1),
        "Readability Score (ARI)": round(4.71 * (len(text)/word_count) + 0.5 * (word_count/sentence_count) - 21.43, 1) if word_count > 0 and sentence_count > 0 else 0
    }



