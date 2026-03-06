from fastapi import FastAPI
from pyspark.sql import SparkSession
from enregistrement import load_model
import os

# Désactivation complète de NativeIO Windows
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-17"  # adapte si besoin
os.environ["HADOOP_OPTIONAL_TOOLS"] = ""
os.environ["HADOOP_ROOT_LOGGER"] = "ERROR"
os.environ["WINUTILS"] = "false"

app = FastAPI(title="Movie recommandation")

spark = (
    SparkSession.builder
    .appName("RecoService")
    .config("spark.hadoop.io.nativeio.enabled", "false")
    .config("spark.hadoop.fs.file.impl.disable.cache", "true")
    .config("spark.driver.extraJavaOptions", "-Djava.library.path=C:/hadoop/bin")
    .config("spark.executor.extraJavaOptions", "-Djava.library.path=C:/hadoop/bin")
    .getOrCreate()
)

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