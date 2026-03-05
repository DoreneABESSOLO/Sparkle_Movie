from __future__ import annotations

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F


MOVIELENS_URL_SMALL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens(dest_dir: str | Path, url: str = MOVIELENS_URL_SMALL) -> Path:
    """
    Télécharge et extrait MovieLens dans dest_dir.
    Retourne le chemin du dossier extrait 
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "ml-latest-small.zip"
    extract_path = dest_dir / "ml-latest-small"

    if extract_path.exists() and (extract_path / "ratings.csv").exists():
        return extract_path

    if not zip_path.exists():
        print(f"[data_loader] Downloading MovieLens from {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)

    print(f"[data_loader] Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

    if not (extract_path / "ratings.csv").exists():
        raise FileNotFoundError(f"ratings.csv not found in {extract_path}")

    return extract_path


def load_dataframes(
    spark: SparkSession,
    data_dir: str | Path,
) -> Tuple[DataFrame, DataFrame]:
    """
    Charge ratings.csv et movies.csv depuis data_dir 
    Nettoyage minimal + schema inféré
    """
    data_dir = Path(data_dir)
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing: {ratings_path}")
    if not movies_path.exists():
        raise FileNotFoundError(f"Missing: {movies_path}")

    ratings_df = spark.read.csv(str(ratings_path), header=True, inferSchema=True)
    movies_df = spark.read.csv(str(movies_path), header=True, inferSchema=True)

    # Typage/clean safe
    ratings_df = (
        ratings_df
        .select(
            F.col("userId").cast("int").alias("userId"),
            F.col("movieId").cast("int").alias("movieId"),
            F.col("rating").cast("double").alias("rating"),
            F.col("timestamp").cast("long").alias("timestamp"),
        )
        .dropna(subset=["userId", "movieId", "rating"])
        .filter((F.col("rating") >= 0.5) & (F.col("rating") <= 5.0))
    )

    movies_df = (
        movies_df
        .select(
            F.col("movieId").cast("int").alias("movieId"),
            F.col("title").cast("string").alias("title"),
            F.col("genres").cast("string").alias("genres"),
        )
        .dropna(subset=["movieId", "title"])
        .dropDuplicates(["movieId"])
    )

    return ratings_df, movies_df


def split_ratings(
    ratings_df: DataFrame,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Split random train/val/test via randomSplit.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    train_df, val_df, test_df = ratings_df.randomSplit(
        [train_ratio, val_ratio, test_ratio],
        seed=seed
    )

    # Cache pour stabilité perf
    return train_df.cache(), val_df.cache(), test_df.cache()


def make_spark(app_name: str, master: str = "local[*]", shuffle_partitions: int = 8) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


if __name__ == "__main__":
    spark = make_spark("data-loader-test")
    data_root = Path(os.environ.get("DATA_ROOT", "data"))

    ds_path = download_movielens(data_root)
    ratings_df, movies_df = load_dataframes(spark, ds_path)
    train_df, val_df, test_df = split_ratings(ratings_df)

    print("ratings:", ratings_df.count(), "movies:", movies_df.count())
    print("train/val/test:", train_df.count(), val_df.count(), test_df.count())
    spark.stop()