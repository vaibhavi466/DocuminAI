import os
import re
import pandas as pd
from datetime import datetime
import shutil

# Define where to keep the history log
HISTORY_FILE = os.path.join("data", "history_log.csv")
ARCHIVE_DIR = os.path.join("data", "archived_documents")

def save_and_log(file_buffer, filename, category, confidence):
    """
    1. Saves the file into a specific folder (e.g. data/archived_documents/invoice/)
    2. Appends the event to a CSV history log.
    """
    
    # --- 1. SAFE STORAGE ---
    # Create the specific category folder if it doesn't exist
    category_folder = os.path.join(ARCHIVE_DIR, category)
    os.makedirs(category_folder, exist_ok=True)
    
    # Save the file there
    save_path = os.path.join(category_folder, filename)
    with open(save_path, "wb") as f:
        f.write(file_buffer.getbuffer())
        
    # --- 2. LOGGING HISTORY ---
    # Create a simple record
    record = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Predicted_Category": category,
        "Confidence": f"{confidence:.2%}",
        "Saved_Path": save_path
    }
    
    # Convert to DataFrame
    df_new = pd.DataFrame([record])
    
    # Append to existing CSV (or create new one)
    if os.path.exists(HISTORY_FILE):
        df_new.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(HISTORY_FILE, index=False)
        
    return "✅ Document archived and logged successfully."

def get_history():
    """
    Reads the CSV log to display in the app.
    """
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()

def calculate_text_metrics(text):
    """
    Calculates basic text features:
    - Word Count
    - Sentence Count
    - Avg Word Length
    - Readability Score (Automated Readability Index - ARI)
    """
    if not text:
        return None
        
    words = text.split()
    sentences = re.split(r'[.\n•]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    # 1. Basic Counts
    word_count = len(words)
    sentence_count = len(sentences) if len(sentences) > 0 else 1
    char_count = sum(len(w) for w in words)
    
    # 2. Averages
    avg_word_length = char_count / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count
    
    # 3. Readability Score (Automated Readability Index)
    # Formula: 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
    # Higher score = Harder to read. (Score 1-14 corresponds roughly to grade level)
    ari_score = (4.71 * avg_word_length) + (0.5 * avg_sentence_length) - 21.43
    ari_score = max(1, round(ari_score, 1)) # formatting
    
    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Avg Word Length": f"{avg_word_length:.1f} chars",
        "Readability Score (ARI)": ari_score
    }


def delete_history_entries(indices_to_delete):
    """
    Deletes specific rows from the history log CSV based on their indices.
    """
    if os.path.exists(HISTORY_FILE):
        # Read the current file
        df = pd.read_csv(HISTORY_FILE)
        
        # Drop the selected rows
        # 'errors="ignore"' prevents crashing if an index is missing
        df = df.drop(indices_to_delete, errors="ignore")
        
        # Save back to CSV
        df.to_csv(HISTORY_FILE, index=False)
        return True
    return False