from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, avg, count

def create_spark_session(app_name="MovieRecommendation"):
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.python.worker.timeout", "120s") \
        .getOrCreate()


def load_data(spark, data_path="ml-latest-small"):
    movies = spark.read.csv(f"{data_path}/movies.csv", header=True, inferSchema=True)
    ratings = spark.read.csv(f"{data_path}/ratings.csv", header=True, inferSchema=True)
    return movies, ratings


def preprocess_data(movies, ratings):
    movies = movies.na.drop()
    ratings = ratings.na.drop()

    movies_r = movies.join(ratings, on="movieId", how="inner")
    return movies_r


def get_ratings_count(ratings):
    return ratings.groupBy("movieId").agg(count("rating").alias("ratings_count"))
