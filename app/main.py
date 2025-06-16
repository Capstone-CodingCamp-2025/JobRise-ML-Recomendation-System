from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors
import psycopg2
import os

# Base path to root project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env from root
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# Path ke folder model   
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Load model dan data
w2v_model = KeyedVectors.load(os.path.join(MODEL_DIR, "w2v_vectors.kv"))
job_vectors = np.load(os.path.join(MODEL_DIR, "job_vectors.npy"))
job_ids = np.load(os.path.join(MODEL_DIR, "job_ids.npy")).tolist()

# Koneksi ke DB
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# Inisialisasi FastAPI
app = FastAPI()

# Schema input dari client (daftar skill user) → KALO MAU POST /predict
class SkillInput(BaseModel):
    skills: List[str]
    top_k: int = 5

# Fungsi untuk embed skill
def embed_text(words: List[str]) -> np.ndarray:
    vectors = []
    for w in words:
        tokens = w.lower().split()
        for token in tokens:
            if token in w2v_model:
                vectors.append(w2v_model[token])
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

# Fungsi cosine similarity
def cosine_sim(user_vec, job_vecs):
    user_tensor = tf.math.l2_normalize(tf.convert_to_tensor(user_vec, dtype=tf.float32), axis=-1)
    job_tensor = tf.math.l2_normalize(tf.convert_to_tensor(job_vecs, dtype=tf.float32), axis=-1)
    return tf.reduce_sum(tf.multiply(user_tensor, job_tensor), axis=-1).numpy()

# Endpoint health check
@app.get("/")
def read_root():
    return {"message": "Welcome to the Job System Recommendation API"}

# Endpoint GET jobs
@app.get("/jobs")
def get_jobs():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        query = """
            SELECT id, title, description, skills
            FROM jobs
            WHERE is_active = 'active'
        """
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

        df_jobs = pd.DataFrame(rows, columns=columns)

        cur.close()
        conn.close()

        return df_jobs.to_dict(orient="records")

    except Exception as e:
        return {
            "jobs": [],
            "message": f"Exception during DB query: {str(e)}"
        }

# Endpoint GET /predict
@app.get("/predict")
def recommend_jobs(user_id: int, top_k: int = 12):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        query_profile = """
            SELECT id, full_name
            FROM profiles
            WHERE user_id = %s
        """
        cur.execute(query_profile, (user_id,))
        profile_row = cur.fetchone()

        if profile_row is None:
            cur.close()
            conn.close()
            return {
                "user_id": user_id,
                "full_name": "",
                "user_skills": [],
                "message": f"User_id {user_id} tidak punya profile."
            }

        profile_id, full_name = profile_row

        query_skills = """
            SELECT name
            FROM skills
            WHERE "profileId" = %s
        """
        cur.execute(query_skills, (profile_id,))
        skills_rows = cur.fetchall()
        user_skills = [row[0] for row in skills_rows] if skills_rows else []

        if not user_skills:
            cur.close()
            conn.close()
            return {
                "user_id": user_id,
                "full_name": full_name,
                "user_skills": [],
                "message": "User tidak punya skill di database."
            }

        user_vec = embed_text(user_skills)
        
        if np.all(user_vec == 0):
            similarities = np.zeros(len(job_vectors))
        else:
            similarities = cosine_sim(user_vec, job_vectors)

        top_k_idx = similarities.argsort()[::-1][:top_k]

        query_jobs = """
            SELECT id, title, company_name, company_logo, salary_min, salary_max, job_type, is_active
            FROM jobs
            WHERE is_active = 'active'
        """
        cur.execute(query_jobs)
        rows_jobs = cur.fetchall()
        columns_jobs = [desc[0] for desc in cur.description]

        df_jobs = pd.DataFrame(rows_jobs, columns=columns_jobs)

        # Build recommendations
        recommendations = []
        for idx in top_k_idx:
            job_id = job_ids[idx]
            job_row = df_jobs[df_jobs['id'] == job_id]

            if job_row.empty:
                continue  

            job_row = job_row.iloc[0]
            score = similarities[idx]

            recommendations.append({
                "id": int(job_row["id"]),
                "title": job_row["title"],
                "company_name": job_row["company_name"],
                "company_logo": job_row["company_logo"],
                "salary_min": job_row["salary_min"],
                "salary_max": job_row["salary_max"],
                "job_type": job_row["job_type"],
                "is_active": job_row["is_active"],
                "score": round(float(score), 3)
            })

        cur.close()
        conn.close()

        # Return JSON
        return {
            "user_id": user_id,
            "full_name": full_name,
            "user_skills": user_skills,
            "top_k": top_k,
            "recommendations": recommendations,
            "note": "Only jobs with is_active = 'active' are recommended. Skills always fetched realtime from DB."
        }

    except Exception as e:
        return {
            "user_id": user_id,
            "full_name": "",
            "user_skills": [],
            "recommendations": [],
            "message": f"Exception during DB query: {str(e)}"
        }



# For testing purposes
# New Endpoint for Skill-Based Prediction
@app.post("/predict_by_skills")
def recommend_jobs_by_skills(skill_input: SkillInput):
    try:
        # Extract skills and top_k from the request
        user_skills = skill_input.skills
        top_k = skill_input.top_k

        if not user_skills:
            return {
                "message": "No skills provided."
            }

        # Embed the user's skills to get the skill vector
        user_vec = embed_text(user_skills)
        
        if np.all(user_vec == 0):
            similarities = np.zeros(len(job_vectors))
        else:
            similarities = cosine_sim(user_vec, job_vectors)

        # Get top-k most similar jobs
        top_k_idx = similarities.argsort()[::-1][:top_k]

        # Query active jobs from the database
        conn = get_db_connection()
        cur = conn.cursor()

        query_jobs = """
            SELECT id, title, company_name, company_logo, salary_min, salary_max, job_type, is_active
            FROM jobs
            WHERE is_active = 'active'
        """
        cur.execute(query_jobs)
        rows_jobs = cur.fetchall()
        columns_jobs = [desc[0] for desc in cur.description]

        df_jobs = pd.DataFrame(rows_jobs, columns=columns_jobs)

        # Build recommendations
        recommendations = []
        for idx in top_k_idx:
            job_id = job_ids[idx]
            job_row = df_jobs[df_jobs['id'] == job_id]

            if job_row.empty:
                continue  

            job_row = job_row.iloc[0]
            score = similarities[idx]

            recommendations.append({
                "id": int(job_row["id"]),
                "title": job_row["title"],
                "company_name": job_row["company_name"],
                "company_logo": job_row["company_logo"],
                "salary_min": job_row["salary_min"],
                "salary_max": job_row["salary_max"],
                "job_type": job_row["job_type"],
                "is_active": job_row["is_active"],
                "score": round(float(score), 3)
            })

        cur.close()
        conn.close()

        # Return JSON response
        return {
            "skills": user_skills,
            "top_k": top_k,
            "recommendations": recommendations,
            "note": "Only jobs with is_active = 'active' are recommended."
        }

    except Exception as e:
        return {
            "message": f"Error during prediction: {str(e)}"
        }
