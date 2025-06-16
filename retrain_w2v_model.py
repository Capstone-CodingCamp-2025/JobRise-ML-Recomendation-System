from dotenv import load_dotenv
import psycopg2
import pandas as pd
import os
from gensim.models import Word2Vec

# Setup path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "app", "models")  

# Load .env
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))
DATABASE_URL = os.getenv("DATABASE_URL")

# DB connection
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# Load job titles from DB
def fetch_job_titles():
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        SELECT title
        FROM jobs
        WHERE is_active = 'active'
    """
    cur.execute(query)
    rows = cur.fetchall()
    titles = [row[0] for row in rows]

    cur.close()
    conn.close()
    return titles

# Tokenize
def preprocess_titles(titles):
    return [title.lower().split() for title in titles if title]

# Train new W2V model
def retrain_w2v_model(sentences):
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=30
    )
    save_path = os.path.join(MODEL_DIR, "w2v_skill.model")  # updated
    model.save(save_path)
    print(f"[SUCCESS] Word2Vec model saved to {save_path}")

if __name__ == "__main__":
    print("[INFO] Fetching job titles from DB...")
    titles = fetch_job_titles()
    print(f"[INFO] Total titles fetched: {len(titles)}")

    print("[INFO] Preprocessing titles...")
    sentences = preprocess_titles(titles)

    print("[INFO] Training new Word2Vec model...")
    retrain_w2v_model(sentences)