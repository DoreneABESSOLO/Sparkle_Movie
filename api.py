from fastapi import FastAPI
from pyspark.sql import SparkSession
from enregistrement import load_model

app = FastAPI(title="Movie recommandation")

spark = SparkSession.builder.appName("RecoService").getOrCreate()
model = load_model(spark)

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    recs = model.recommendForAllUsers(n)
    user_recs = recs.filter(recs.userId == user_id).collect()

    if not user_recs:
        return {"message": "Utilisateur inconnu"}

    return {
        "user_id": user_id,
        "recommendations": user_recs[0]["recommendations"]
    }