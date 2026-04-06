from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
import io

app = FastAPI(title="Movie Recommender API")

# Variables globales
model = None
user_item_matrix = None


#chargement au démarrage
@app.on_event("startup")
def load_artifacts():
    global model, user_item_matrix

    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("model.pkl", "rb") as f:
            user_item_matrix = pickle.load(f)

        print("Modèle chargé avec succès")

    except Exception as e:
        print(f" Erreur chargement modèle : {e}")
        model = None


#recommandations
@app.get("/recommendations")
def get_recommendations(user_id: int, k: int = 10):
    global model, user_item_matrix

    if model is None or user_item_matrix is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    if user_id >= user_item_matrix.shape[0]:
        raise HTTPException(status_code=404, detail="Utilisateur inconnu")

    try:
        recommendations = model.recommend(
            user_id,
            user_item_matrix[user_id],
            N=k
        )

        results = [
            {"movie_id": int(item), "score": float(score)}
            for item, score in recommendations
        ]

        return {
            "user_id": user_id,
            "recommendations": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")
def health():
    global model, user_item_matrix

    status = {
        "model_loaded": model is not None,
        "matrix_loaded": user_item_matrix is not None,
        "status": "ok" if model and user_item_matrix is not None else "error"
    }

    return status

# stockage simple
request_times = []
timestamps = []

@app.middleware("http")
async def track_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    request_times.append(duration)
    timestamps.append(time.time())

    return response


@app.get("/metrics/plot")
def plot_metrics():
    if len(request_times) == 0:
        raise HTTPException(status_code=404, detail="Pas de données")

    plt.figure()
    plt.plot(request_times)
    plt.title("Temps de réponse des requêtes")
    plt.xlabel("Requête")
    plt.ylabel("Temps (s)")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")