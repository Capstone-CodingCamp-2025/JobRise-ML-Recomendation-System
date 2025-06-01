# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from gensim.models import Word2Vec
# import requests
# import os


# load_dotenv()
# # DATABASE_URL = os.getenv("DATABASE_URL")
# # engine = create_engine(DATABASE_URL)

# # Load model Word2Vec dan job vectors
# w2v_model = Word2Vec.load("w2v_skill.model")
# job_vectors = np.load("job_vectors.npy")
# job_ids = np.load("job_ids.npy").tolist()  # job_ids disimpan saat training

# # Inisialisasi FastAPI
# app = FastAPI()


# # Schema input dari client (daftar skill user)
# class SkillInput(BaseModel):
#     skills: List[str]
#     top_k: int = 5  # jumlah rekomendasi (opsional, default = 5)

# # Fungsi untuk embed daftar kata (skill) ke vektor
# def embed_text(words: List[str]) -> np.ndarray:
#     vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
#     if not vectors:
#         return np.zeros(w2v_model.vector_size)
#     return np.mean(vectors, axis=0)

# # Fungsi cosine similarity menggunakan TensorFlow
# def cosine_sim(user_vec, job_vecs):
#     user_tensor = tf.math.l2_normalize(tf.convert_to_tensor(user_vec, dtype=tf.float32), axis=-1)
#     job_tensor = tf.math.l2_normalize(tf.convert_to_tensor(job_vecs, dtype=tf.float32), axis=-1)
#     return tf.reduce_sum(tf.multiply(user_tensor, job_tensor), axis=-1).numpy()

# # Endpoint prediksi rekomendasi
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Job System Recommendation API"}

# # API buat GET jobs
# @app.get("/jobs")
# def get_jobs():
#     SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
#     SUPABASE_URL = os.getenv("SUPABASE_URL")

#     headers = {
#         "apikey": SUPABASE_API_KEY,
#         "Authorization": f"Bearer {SUPABASE_API_KEY}"
#     }

#     url = f"{SUPABASE_URL}/rest/v1/jobs?is_active=eq.active&select=id,title,description,skills"
#     response = requests.get(url, headers=headers, timeout=5)

#     if response.status_code != 200:
#         return {
#             "jobs": [],
#             "message": f"Failed to fetch jobs from Supabase. Status: {response.status_code}, Error: {response.text}"
#         }

#     df_jobs = pd.DataFrame(response.json())
#     return df_jobs.to_dict(orient="records")

# # @app.get("/jobs")
# # def get_jobs():
# #     df_jobs = pd.read_sql("""
# #         SELECT id, title, description, skills
# #         FROM jobs
# #         WHERE is_active = 'active'
# #     """, con=engine)
# #     return df_jobs.to_dict(orient="records")

# @app.get("/predict")
# def recommend_jobs(user_id: int, top_k: int = 5):
#     # Ambil skill user dari table skills
#     # df_user_skills = pd.read_sql(f"""
#     #     SELECT name
#     #     FROM skills
#     #     WHERE "profileId" = {user_id}
#     # """, con=engine)

#     # user_skills = df_user_skills['name'].tolist()

#     # Example REST API URL → table skills
    
#     SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
#     SUPABASE_URL = os.getenv("SUPABASE_URL")

#     headers = {
#         "apikey": SUPABASE_API_KEY,
#         "Authorization": f"Bearer {SUPABASE_API_KEY}"
#     }
    
#     url = f"{SUPABASE_URL}/rest/v1/skills?profileId=eq.{user_id}&select=name"

#     response = requests.get(url, headers=headers)

#     if response.status_code != 200:
#         return {
#             "user_id": user_id,
#             "user_skills": [],
#             "message": f"Failed to fetch skills from Supabase. Status: {response.status_code}, Error: {response.text}"
#         }

#     # Convert to DataFrame
#     df_user_skills = pd.DataFrame(response.json())

#     user_skills = df_user_skills['name'].tolist() if not df_user_skills.empty else []

#     # Cek kalau user gak punya skill
#     if not user_skills:
#         return {
#             "user_id": user_id,
#             "user_skills": [],
#             "message": "User tidak punya skill di database."
#         }

#     # Embed user skill
#     user_vec = embed_text(user_skills)

#     if not user_vec.any():
#         return {
#             "user_id": user_id,
#             "user_skills": user_skills,
#             "message": "Skill user tidak cocok dengan model."
#         }

#     # Hitung similarity
#     similarities = cosine_sim(user_vec, job_vectors)
#     top_k_idx = similarities.argsort()[::-1][:top_k]

#     # Ambil detail job dari table
#     # df_jobs = pd.read_sql("""
#     #     SELECT id, title, company_name, company_logo, salary_min, salary_max, job_type, is_active
#     #     FROM jobs
#     #     WHERE is_active = 'active'
#     # """, con=engine)

#     # Build recommendations 
#     recommendations = []
#     for idx in top_k_idx:
#         job_id = job_ids[idx]
#         job_row = df_jobs[df_jobs['id'] == job_id].iloc[0]
#         score = similarities[idx]

#         recommendations.append({
#             "job_id": int(job_row["id"]),
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
#         "user_skills": user_skills,
#         "top_k": top_k,
#         "recommendations": recommendations
#     }



# # @app.post("/predict")
# # def recommend_jobs(input: SkillInput):
# #     user_vec = embed_text(input.skills)
    
# #     if not user_vec.any():
# #         return {"message": "Skill tidak cocok dengan model."}

# #     similarities = cosine_sim(user_vec, job_vectors)
# #     top_k_idx = similarities.argsort()[::-1][:input.top_k]

# #     results = [{"job_id": job_ids[i], "score": float(similarities[i])} for i in top_k_idx]
# #     return {"recommendations": results}

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
import requests
import os

# Load .env
load_dotenv()

# Load model Word2Vec dan job vectors
w2v_model = Word2Vec.load("w2v_skill.model")
job_vectors = np.load("job_vectors.npy")
job_ids = np.load("job_ids.npy").tolist()

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
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")

    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}"
    }

    url = f"{SUPABASE_URL}/rest/v1/jobs?is_active=eq.active&select=id,title,description,skills"

    try:
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            return {
                "jobs": [],
                "message": f"Failed to fetch jobs from Supabase. Status: {response.status_code}, Error: {response.text}"
            }

        df_jobs = pd.DataFrame(response.json())
        return df_jobs.to_dict(orient="records")

    except Exception as e:
        return {
            "jobs": [],
            "message": f"Exception during Supabase API call: {str(e)}"
        }

# Endpoint GET /predict
@app.get("/predict")
def recommend_jobs(user_id: int, top_k: int = 5):
    
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")

    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}"
    }

    # Ambil profileId & full_name dari profiles
    url_profile = f"{SUPABASE_URL}/rest/v1/profiles?user_id=eq.{user_id}&select=id,full_name"

    response_profile = requests.get(url_profile, headers=headers, timeout=5)

    if response_profile.status_code != 200:
        return {
            "user_id": user_id,
            "full_name": "",
            "user_skills": [],
            "message": f"Failed to fetch profileId. Status: {response_profile.status_code}, Error: {response_profile.text}"
        }

    df_profile = pd.DataFrame(response_profile.json())

    if df_profile.empty:
        return {
            "user_id": user_id,
            "full_name": "",
            "user_skills": [],
            "message": f"User_id {user_id} tidak punya profile."
        }

    # Ambil profileId & full_name
    profile_id = df_profile.iloc[0]['id']
    full_name = df_profile.iloc[0]['full_name']

    # Ambil skills berdasarkan profileId
    url_skills = f"{SUPABASE_URL}/rest/v1/skills?profileId=eq.{profile_id}&select=name"

    response_skills = requests.get(url_skills, headers=headers, timeout=5)

    if response_skills.status_code != 200:
        return {
            "user_id": user_id,
            "full_name": full_name,
            "user_skills": [],
            "message": f"Failed to fetch skills. Status: {response_skills.status_code}, Error: {response_skills.text}"
        }

    df_user_skills = pd.DataFrame(response_skills.json())
    user_skills = df_user_skills['name'].tolist() if not df_user_skills.empty else []

    # Cek kalau user gak punya skill
    if not user_skills:
        return {
            "user_id": user_id,
            "full_name": full_name,
            "user_skills": [],
            "message": "User tidak punya skill di database."
        }

    # Embed user skill
    user_vec = embed_text(user_skills)

    if not user_vec.any():
        return {
            "user_id": user_id,
            "full_name": full_name,
            "user_skills": user_skills,
            "message": "Skill user tidak cocok dengan model."
        }

    # Hitung similarity
    similarities = cosine_sim(user_vec, job_vectors)
    top_k_idx = similarities.argsort()[::-1][:top_k]

    # Ambil detail job dari table (hanya active)
    url_jobs = f"{SUPABASE_URL}/rest/v1/jobs?is_active=eq.active&select=id,title,company_name,company_logo,salary_min,salary_max,job_type,is_active"

    response_jobs = requests.get(url_jobs, headers=headers, timeout=5)

    if response_jobs.status_code != 200:
        return {
            "user_id": user_id,
            "full_name": full_name,
            "user_skills": user_skills,
            "message": f"Failed to fetch jobs. Status: {response_jobs.status_code}, Error: {response_jobs.text}"
        }

    df_jobs = pd.DataFrame(response_jobs.json())

    # Build recommendations
    recommendations = []
    for idx in top_k_idx:
        job_id = job_ids[idx]
        job_row = df_jobs[df_jobs['id'] == job_id]

        if job_row.empty:
            continue  # Skip kalo job_id ga ada (misal job deactive)

        job_row = job_row.iloc[0]
        score = similarities[idx]

        recommendations.append({
            "job_id": int(job_row["id"]),
            "title": job_row["title"],
            "company_name": job_row["company_name"],
            "company_logo": job_row["company_logo"],
            "salary_min": job_row["salary_min"],
            "salary_max": job_row["salary_max"],
            "job_type": job_row["job_type"],
            "is_active": job_row["is_active"],
            "score": round(float(score), 3)
        })

    # Return JSON
    return {
        "user_id": user_id,
        "full_name": full_name,
        "user_skills": user_skills,
        "top_k": top_k,
        "recommendations": recommendations,
        "note": "Only jobs with is_active = 'active' are recommended. Skills always fetched realtime from Supabase."
    }

# @app.get("/predict")
# def recommend_jobs(user_id: int, top_k: int = 5):
#     SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
#     SUPABASE_URL = os.getenv("SUPABASE_URL")

#     headers = {
#         "apikey": SUPABASE_API_KEY,
#         "Authorization": f"Bearer {SUPABASE_API_KEY}"
#     }

#     # Ambil skills user
#     url_skills = f"{SUPABASE_URL}/rest/v1/skills?profileId=eq.{user_id}&select=name"

#     try:
#         response_skills = requests.get(url_skills, headers=headers, timeout=5)

#         if response_skills.status_code != 200:
#             return {
#                 "user_id": user_id,
#                 "user_skills": [],
#                 "message": f"Failed to fetch skills from Supabase. Status: {response_skills.status_code}, Error: {response_skills.text}"
#             }

#         df_user_skills = pd.DataFrame(response_skills.json())
#         user_skills = df_user_skills['name'].tolist() if not df_user_skills.empty else []

#     except Exception as e:
#         return {
#             "user_id": user_id,
#             "user_skills": [],
#             "message": f"Exception during Supabase API call (skills): {str(e)}"
#         }

#     # Cek kalau user gak punya skill
#     if not user_skills:
#         return {
#             "user_id": user_id,
#             "user_skills": [],
#             "message": "User tidak punya skill di database."
#         }

#     # Embed skill
#     user_vec = embed_text(user_skills)

#     if not user_vec.any():
#         return {
#             "user_id": user_id,
#             "user_skills": user_skills,
#             "message": "Skill user tidak cocok dengan model."
#         }

#     # Hitung similarity
#     similarities = cosine_sim(user_vec, job_vectors)
#     top_k_idx = similarities.argsort()[::-1][:top_k]

#     # Ambil jobs (via REST API)
#     url_jobs = f"{SUPABASE_URL}/rest/v1/jobs?is_active=eq.active&select=id,title,company_name,company_logo,salary_min,salary_max,job_type,is_active"

#     try:
#         response_jobs = requests.get(url_jobs, headers=headers, timeout=5)

#         if response_jobs.status_code != 200:
#             return {
#                 "user_id": user_id,
#                 "user_skills": user_skills,
#                 "recommendations": [],
#                 "message": f"Failed to fetch jobs from Supabase. Status: {response_jobs.status_code}, Error: {response_jobs.text}"
#             }

#         df_jobs = pd.DataFrame(response_jobs.json())

#     except Exception as e:
#         return {
#             "user_id": user_id,
#             "user_skills": user_skills,
#             "recommendations": [],
#             "message": f"Exception during Supabase API call (jobs): {str(e)}"
#         }

#     # Build recommendations
#     recommendations = []
#     for idx in top_k_idx:
#         job_id = job_ids[idx]
#         job_row = df_jobs[df_jobs['id'] == job_id]

#         if job_row.empty:
#             continue

#         job_row = job_row.iloc[0]
#         score = similarities[idx]

#         recommendations.append({
#             "job_id": int(job_row["id"]),
#             "title": job_row["title"],
#             "company_name": job_row["company_name"],
#             "company_logo": job_row["company_logo"],
#             "salary_min": job_row["salary_min"],
#             "salary_max": job_row["salary_max"],
#             "job_type": job_row["job_type"],
#             "is_active": job_row["is_active"],
#             "score": round(float(score), 3)
#         })

#     return {
#         "user_id": user_id,
#         "user_skills": user_skills,
#         "top_k": top_k,
#         "recommendations": recommendations
#     }
