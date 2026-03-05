from __future__ import annotations

import os
import json
import itertools
import shutil
import platform
import getpass
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

from config import load_config, resolve_paths
from data_loader import download_movielens, load_dataframes


# -------------------------
# Utils logs / runs
# -------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def make_run_dir(models_root: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = models_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# -------------------------
# Spark
# -------------------------
def make_spark(app_name: str, master: str, shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


# -------------------------
# Split
# -------------------------
def split_ratings(ratings_df, seed: int, train_ratio: float, val_ratio: float):
    """
    Split stable: train / val / test with 2 randomSplit.
    """
    train_df, temp_df = ratings_df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    val_weight = val_ratio / (1 - train_ratio)
    val_df, test_df = temp_df.randomSplit([val_weight, 1 - val_weight], seed=seed)
    return train_df.cache(), val_df.cache(), test_df.cache()


# -------------------------
# TF-IDF pipeline
# -------------------------
def build_tfidf_pipeline(num_features: int, stopwords: str = "english") -> Pipeline:
    tokenizer = Tokenizer(inputCol="doc", outputCol="tokens")
    if stopwords == "english":
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
        hashing_input = "filtered"
        stages = [tokenizer, remover]
    else:
        hashing_input = "tokens"
        stages = [tokenizer]

    hashing = HashingTF(inputCol=hashing_input, outputCol="tf", numFeatures=int(num_features))
    idf = IDF(inputCol="tf", outputCol="features")
    stages += [hashing, idf]
    return Pipeline(stages=stages)


# -------------------------
# ALS train/eval
# -------------------------
def train_eval_als(train_df, val_df, seed: int, als_base_cfg: dict, rank: int, regParam: float, maxIter: int,
                   rmse_eval: RegressionEvaluator, mae_eval: RegressionEvaluator):
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy=als_base_cfg.get("cold_start_strategy", "drop"),
        nonnegative=bool(als_base_cfg.get("nonnegative", True)),
        rank=int(rank),
        regParam=float(regParam),
        maxIter=int(maxIter),
        seed=int(seed),
    )
    model = als.fit(train_df)
    preds = model.transform(val_df)
    rmse = rmse_eval.evaluate(preds)
    mae = mae_eval.evaluate(preds)
    return model, rmse, mae


def main():
    # =========================
    # 1) Config
    # =========================
    cfg = resolve_paths(load_config())

    data_root = Path(cfg["paths"]["data_root"]).resolve()
    models_root = Path(cfg["paths"]["models_dir"]).resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["dataset"]["seed"])
    train_ratio = float(cfg["dataset"]["split"]["train_ratio"])
    val_ratio = float(cfg["dataset"]["split"]["val_ratio"])

    als_cfg = cfg["als"]
    tfidf_cfg = cfg["tfidf"]

    # Run logs dir
    run_dir = make_run_dir(models_root)
    started_at = utc_now_iso()

    save_json(run_dir / "train_config.json", {
        "started_at_utc": started_at,
        "config_effective": cfg,
        "env": {
            "user": getpass.getuser(),
            "python": os.sys.version,
            "platform": platform.platform(),
        }
    })
    print(f"[train] Run dir: {run_dir}")

    # =========================
    # 2) Spark
    # =========================
    spark = make_spark(
        app_name=cfg["spark"]["app_name_train"],
        master=cfg["spark"]["master"],
        shuffle_partitions=int(cfg["spark"]["shuffle_partitions"]),
    )

    try:
        # =========================
        # 3) Download + load data
        # =========================
        ds_path = download_movielens(data_root, url=cfg["dataset"]["movielens_url"])
        ratings_df, movies_df = load_dataframes(spark, ds_path)

        # quick clean safety
        ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating"])
        movies_df = movies_df.dropna(subset=["movieId", "title", "genres"])

        # =========================
        # 4) Split
        # =========================
        train_df, val_df, test_df = split_ratings(ratings_df, seed, train_ratio, val_ratio)

        print("[train] counts:",
              "total=", ratings_df.count(),
              "train=", train_df.count(),
              "val=", val_df.count(),
              "test=", test_df.count())

        # =========================
        # 5) Evaluators (RMSE + MAE)
        # =========================
        rmse_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        mae_eval = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

        # =========================
        # 6) ALS Grid Search (from config)
        # =========================
        grid = als_cfg["grid"]
        ranks = grid["rank"]
        reg_params = grid["regParam"]
        max_iters = grid["maxIter"]

        best_rmse = float("inf")
        best_model = None
        best_params = None
        results = []

        for r, reg, it in itertools.product(ranks, reg_params, max_iters):
            model, rmse, mae = train_eval_als(
                train_df=train_df,
                val_df=val_df,
                seed=seed,
                als_base_cfg=als_cfg,
                rank=r,
                regParam=reg,
                maxIter=it,
                rmse_eval=rmse_eval,
                mae_eval=mae_eval
            )
            results.append({
                "rank": int(r),
                "regParam": float(reg),
                "maxIter": int(it),
                "rmse_val": float(rmse),
                "mae_val": float(mae),
            })
            print(f"[ALS] rank={r}, regParam={reg}, maxIter={it} -> RMSE(val)={rmse:.4f} | MAE(val)={mae:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = (int(r), float(reg), int(it))

        print(f"\n[train] ✅ Best ALS params: {best_params} | RMSE(val)={best_rmse:.4f}")

        # =========================
        # 7) Refit on train+val (optional)
        # =========================
        if bool(als_cfg.get("final_refit_on_trainval", True)):
            trainval_df = train_df.unionByName(val_df)
            r, reg, it = best_params
            als_final = ALS(
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                coldStartStrategy=als_cfg.get("cold_start_strategy", "drop"),
                nonnegative=bool(als_cfg.get("nonnegative", True)),
                rank=int(r),
                regParam=float(reg),
                maxIter=int(it),
                seed=int(seed),
            )
            best_model = als_final.fit(trainval_df)

        # =========================
        # 8) Test metrics
        # =========================
        test_preds = best_model.transform(test_df)
        rmse_test = float(rmse_eval.evaluate(test_preds))
        mae_test = float(mae_eval.evaluate(test_preds))

        print(f"[train] 📌 RMSE(test) = {rmse_test:.6f}")
        print(f"[train] 📌 MAE(test)  = {mae_test:.6f}")

        # =========================
        # 9) TF-IDF training (from config)
        # =========================
        num_features = int(tfidf_cfg["num_features"])    # <--- param via YAML/ENV
        stopwords = tfidf_cfg.get("stopwords", "english")

        movies_text = movies_df.withColumn(
            "doc",
            F.concat_ws(" ", F.col("title"), F.regexp_replace(F.col("genres"), r"\|", " "))
        )

        tfidf_pipeline = build_tfidf_pipeline(num_features=num_features, stopwords=stopwords)
        tfidf_model = tfidf_pipeline.fit(movies_text)
        movies_features = tfidf_model.transform(movies_text).select("movieId", "features")

        # =========================
        # 10) Save artifacts (simple deployable format)
        # =========================
        import pickle

        save_dir = models_root / "models_simple"
        save_dir.mkdir(parents=True, exist_ok=True)

        # ALS factors -> pandas -> pickle
        user_factors_pd = best_model.userFactors.toPandas()
        item_factors_pd = best_model.itemFactors.toPandas()

        with open(save_dir / "als_user_factors.pkl", "wb") as f:
            pickle.dump(user_factors_pd, f)

        with open(save_dir / "als_item_factors.pkl", "wb") as f:
            pickle.dump(item_factors_pd, f)

        # TF-IDF features -> pandas -> pickle
        features_pd = movies_features.toPandas()
        with open(save_dir / "tfidf_features.pkl", "wb") as f:
            pickle.dump(features_pd, f)

        # metrics file (portable)
        metrics_payload = {
            "best_params": {"rank": best_params[0], "regParam": best_params[1], "maxIter": best_params[2]},
            "rmse_val_best": float(best_rmse),
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "grid_results": results,
            "saved_paths": {
                "als_user_factors": str(save_dir / "als_user_factors.pkl"),
                "als_item_factors": str(save_dir / "als_item_factors.pkl"),
                "tfidf_features": str(save_dir / "tfidf_features.pkl"),
            }
        }
        save_json(save_dir / "metrics.json", metrics_payload)

        # =========================
        # 11) Run logs (requirement #3)
        # =========================
        ended_at = utc_now_iso()

        save_json(run_dir / "summary.json", {
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "hyperparameters_best": metrics_payload["best_params"],
            "metrics": {
                "rmse_val_best": float(best_rmse),
                "rmse_test": rmse_test,
                "mae_test": mae_test,
            },
            "model_paths": metrics_payload["saved_paths"],
            "models_dir": str(save_dir),
        })

        save_json(run_dir / "grid_results.json", {"results": results})

        print(f"[train] ✅ Saved deployable artifacts to: {save_dir}")
        print(f"[train] ✅ Logged run to: {run_dir}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()