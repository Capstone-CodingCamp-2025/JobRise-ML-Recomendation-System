from dotenv import load_dotenv
from sqlalchemy import create_engine
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
import os


load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Load model Word2Vec dan job vectors
w2v_model = Word2Vec.load("w2v_skill.model")
job_vectors = np.load("job_vectors.npy")
job_ids = np.load("job_ids.npy").tolist()  # job_ids disimpan saat training

# Inisialisasi FastAPI
app = FastAPI()


# Schema input dari client (daftar skill user)
class SkillInput(BaseModel):
    skills: List[str]
    top_k: int = 5  # jumlah rekomendasi (opsional, default = 5)

# Fungsi untuk embed daftar kata (skill) ke vektor
def embed_text(words: List[str]) -> np.ndarray:
    vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

# Fungsi cosine similarity menggunakan TensorFlow
def cosine_sim(user_vec, job_vecs):
    user_tensor = tf.math.l2_normalize(tf.convert_to_tensor(user_vec, dtype=tf.float32), axis=-1)
    job_tensor = tf.math.l2_normalize(tf.convert_to_tensor(job_vecs, dtype=tf.float32), axis=-1)
    return tf.reduce_sum(tf.multiply(user_tensor, job_tensor), axis=-1).numpy()

# Endpoint prediksi rekomendasi
@app.get("/")
def read_root():
    return {"message": "Welcome to the Job System Recommendation API"}

# API buat GET jobs
@app.get("/jobs")
def get_jobs():
    df_jobs = pd.read_sql("""
        SELECT id, title, description, skills
        FROM jobs
        WHERE is_active = 'active'
    """, con=engine)
    return df_jobs.to_dict(orient="records")

@app.get("/predict")
@app.get("/predict")
def recommend_jobs(user_id: int, top_k: int = 5):
    # Ambil skill user dari table skills
    df_user_skills = pd.read_sql(f"""
        SELECT name
        FROM skills
        WHERE "profileId" = {user_id}
    """, con=engine)

    user_skills = df_user_skills['name'].tolist()

    # Cek kalau user gak punya skill
    if not user_skills:
        return {
            "user_id": user_id,
            "user_skills": [],
            "message": "User tidak punya skill di database."
        }

    # Embed user skill
    user_vec = embed_text(user_skills)

    if not user_vec.any():
        return {
            "user_id": user_id,
            "user_skills": user_skills,
            "message": "Skill user tidak cocok dengan model."
        }

    # Hitung similarity
    similarities = cosine_sim(user_vec, job_vectors)
    top_k_idx = similarities.argsort()[::-1][:top_k]

    # Ambil detail job dari table
    df_jobs = pd.read_sql("""
        SELECT id, title, company_name, company_logo, salary_min, salary_max, job_type, is_active
        FROM jobs
        WHERE is_active = 'active'
    """, con=engine)

    # Build recommendations 
    recommendations = []
    for idx in top_k_idx:
        job_id = job_ids[idx]
        job_row = df_jobs[df_jobs['id'] == job_id].iloc[0]
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
        "user_skills": user_skills,
        "top_k": top_k,
        "recommendations": recommendations
    }



# @app.post("/predict")
# def recommend_jobs(input: SkillInput):
#     user_vec = embed_text(input.skills)
    
#     if not user_vec.any():
#         return {"message": "Skill tidak cocok dengan model."}

#     similarities = cosine_sim(user_vec, job_vectors)
#     top_k_idx = similarities.argsort()[::-1][:input.top_k]

#     results = [{"job_id": job_ids[i], "score": float(similarities[i])} for i in top_k_idx]
#     return {"recommendations": results}
