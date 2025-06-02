# from dotenv import load_dotenv
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from gensim.models import Word2Vec
# import requests
# import os

# # Base path to root project
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Load .env from root
# load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# # Path ke folder model
# MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# # Load model dan data
# w2v_model = Word2Vec.load(os.path.join(MODEL_DIR, "w2v_skill.model"))
# job_vectors = np.load(os.path.join(MODEL_DIR, "job_vectors.npy"))
# job_ids = np.load(os.path.join(MODEL_DIR, "job_ids.npy")).tolist()


# # Inisialisasi FastAPI
# app = FastAPI()

# # Schema input dari client (daftar skill user) → KALO MAU POST /predict
# class SkillInput(BaseModel):
#     skills: List[str]
#     top_k: int = 5

# # Fungsi untuk embed skill
# def embed_text(words: List[str]) -> np.ndarray:
#     vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
#     if not vectors:
#         return np.zeros(w2v_model.vector_size)
#     return np.mean(vectors, axis=0)

# # Fungsi cosine similarity
# def cosine_sim(user_vec, job_vecs):
#     user_tensor = tf.math.l2_normalize(tf.convert_to_tensor(user_vec, dtype=tf.float32), axis=-1)
#     job_tensor = tf.math.l2_normalize(tf.convert_to_tensor(job_vecs, dtype=tf.float32), axis=-1)
#     return tf.reduce_sum(tf.multiply(user_tensor, job_tensor), axis=-1).numpy()

# # Endpoint health check
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Job System Recommendation API"}

# # Endpoint GET jobs
# @app.get("/jobs")
# def get_jobs():
#     SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
#     SUPABASE_URL = os.getenv("SUPABASE_URL")

#     headers = {
#         "apikey": SUPABASE_API_KEY,
#         "Authorization": f"Bearer {SUPABASE_API_KEY}"
#     }

#     url = f"{SUPABASE_URL}/rest/v1/jobs?is_active=eq.active&select=id,title,description,skills"

#     try:
#         response = requests.get(url, headers=headers, timeout=5)

#         if response.status_code != 200:
#             return {
#                 "jobs": [],
#                 "message": f"Failed to fetch jobs from Supabase. Status: {response.status_code}, Error: {response.text}"
#             }

#         df_jobs = pd.DataFrame(response.json())
#         return df_jobs.to_dict(orient="records")

#     except Exception as e:
#         return {
#             "jobs": [],
#             "message": f"Exception during Supabase API call: {str(e)}"
#         }

# # Endpoint GET /predict
# @app.get("/predict")
# def recommend_jobs(user_id: int, top_k: int = 12):
    
#     SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
#     SUPABASE_URL = os.getenv("SUPABASE_URL")

#     headers = {
#         "apikey": SUPABASE_API_KEY,
#         "Authorization": f"Bearer {SUPABASE_API_KEY}"
#     }

#     # Ambil profileId & full_name dari profiles
#     url_profile = f"{SUPABASE_URL}/rest/v1/profiles?user_id=eq.{user_id}&select=id,full_name"

#     response_profile = requests.get(url_profile, headers=headers, timeout=5)

#     if response_profile.status_code != 200:
#         return {
#             "user_id": user_id,
#             "full_name": "",
#             "user_skills": [],
#             "message": f"Failed to fetch profileId. Status: {response_profile.status_code}, Error: {response_profile.text}"
#         }

#     df_profile = pd.DataFrame(response_profile.json())

#     if df_profile.empty:
#         return {
#             "user_id": user_id,
#             "full_name": "",
#             "user_skills": [],
#             "message": f"User_id {user_id} tidak punya profile."
#         }

#     # Ambil profileId & full_name
#     profile_id = df_profile.iloc[0]['id']
#     full_name = df_profile.iloc[0]['full_name']

#     # Ambil skills berdasarkan profileId
#     url_skills = f"{SUPABASE_URL}/rest/v1/skills?profileId=eq.{profile_id}&select=name"

#     response_skills = requests.get(url_skills, headers=headers, timeout=5)

#     if response_skills.status_code != 200:
#         return {
#             "user_id": user_id,
#             "full_name": full_name,
#             "user_skills": [],
#             "message": f"Failed to fetch skills. Status: {response_skills.status_code}, Error: {response_skills.text}"
#         }

#     df_user_skills = pd.DataFrame(response_skills.json())
#     user_skills = df_user_skills['name'].tolist() if not df_user_skills.empty else []

#     # Cek kalau user gak punya skill
#     if not user_skills:
#         return {
#             "user_id": user_id,
#             "full_name": full_name,
#             "user_skills": [],
#             "message": "User tidak punya skill di database."
#         }

#     # Embed user skill
#     user_vec = embed_text(user_skills)

#     if not user_vec.any():
#         return {
#             "user_id": user_id,
#             "full_name": full_name,
#             "user_skills": user_skills,
#             "message": "Skill user tidak cocok dengan model."
#         }

#     # Hitung similarity
#     similarities = cosine_sim(user_vec, job_vectors)
#     top_k_idx = similarities.argsort()[::-1][:top_k]

#     # Ambil detail job dari table (hanya active)
#     url_jobs = f"{SUPABASE_URL}/rest/v1/jobs?is_active=eq.active&select=id,title,company_name,company_logo,salary_min,salary_max,job_type,is_active"

#     response_jobs = requests.get(url_jobs, headers=headers, timeout=5)

#     if response_jobs.status_code != 200:
#         return {
#             "user_id": user_id,
#             "full_name": full_name,
#             "user_skills": user_skills,
#             "message": f"Failed to fetch jobs. Status: {response_jobs.status_code}, Error: {response_jobs.text}"
#         }

#     df_jobs = pd.DataFrame(response_jobs.json())

#     # Build recommendations
#     recommendations = []
#     for idx in top_k_idx:
#         job_id = job_ids[idx]
#         job_row = df_jobs[df_jobs['id'] == job_id]

#         if job_row.empty:
#             continue  # Skip kalo job_id ga ada (misal job deactive)

#         job_row = job_row.iloc[0]
#         score = similarities[idx]

#         recommendations.append({
#             "id": int(job_row["id"]),
#             "title": job_row["title"],
#             "company_name": job_row["company_name"],
#             "company_logo": job_row["company_logo"],
#             "salary_min": job_row["salary_min"],
#             "salary_max": job_row["salary_max"],
#             "job_type": job_row["job_type"],
#             "is_active": job_row["is_active"],
#             "score": round(float(score), 3)
#         })

#     # Return JSON
#     return {
#         "user_id": user_id,
#         "full_name": full_name,
#         "user_skills": user_skills,
#         "top_k": top_k,
#         "recommendations": recommendations,
#         "note": "Only jobs with is_active = 'active' are recommended. Skills always fetched realtime from Supabase."
#     }

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
import psycopg2
import os

# Base path to root project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env from root
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# Path ke folder model
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Load model dan data
w2v_model = Word2Vec.load(os.path.join(MODEL_DIR, "w2v_skill.model"))
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
    vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
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
            WHERE profileId = %s
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

        if not user_vec.any():
            cur.close()
            conn.close()
            return {
                "user_id": user_id,
                "full_name": full_name,
                "user_skills": user_skills,
                "message": "Skill user tidak cocok dengan model."
            }

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
