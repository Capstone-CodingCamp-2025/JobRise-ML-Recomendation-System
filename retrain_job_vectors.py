from dotenv import load_dotenv
import psycopg2
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec

# Setup path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "app", "models")

# Load .env
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# Load W2V model
w2v_model = Word2Vec.load(os.path.join(MODEL_DIR, "w2v_skill.model"))

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def embed_text(words):
    vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

def retrain_job_vectors():
    conn = get_db_connection()
    cur = conn.cursor()

    # Pake title aja (karena skills ga ada di DB skrg)
    query = """
        SELECT id, title
        FROM jobs
        WHERE is_active = 'active'
    """
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    df_jobs = pd.DataFrame(rows, columns=columns)

    print(f"[INFO] Total active jobs fetched: {len(df_jobs)}")

    job_vectors = []
    job_ids = []

    if df_jobs.empty:
        print("[WARNING] No active jobs found. Exiting retrain.")
    else:
        for _, row in df_jobs.iterrows():
            # Combine title aja (karena skills udah ga ada)
            text = row['title']
            words = text.lower().split()

            vector = embed_text(words)

            job_vectors.append(vector)
            job_ids.append(row['id'])

        job_vectors = np.array(job_vectors)
        job_ids = np.array(job_ids)

        np.save(os.path.join(MODEL_DIR, "job_vectors.npy"), job_vectors)
        np.save(os.path.join(MODEL_DIR, "job_ids.npy"), job_ids)

        print(f"[SUCCESS] Saved {len(job_ids)} job vectors to /app/models/")

    cur.close()
    conn.close()

if __name__ == "__main__":
    retrain_job_vectors()