from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec

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
@app.post("/predict")
def recommend_jobs(input: SkillInput):
    user_vec = embed_text(input.skills)
    
    if not user_vec.any():
        return {"message": "Skill tidak cocok dengan model."}

    similarities = cosine_sim(user_vec, job_vectors)
    top_k_idx = similarities.argsort()[::-1][:input.top_k]

    results = [{"job_id": job_ids[i], "score": float(similarities[i])} for i in top_k_idx]
    return {"recommendations": results}
