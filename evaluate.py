from __future__ import annotations

import os
import json
import platform
import getpass
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

from config import load_config, resolve_paths
from data_loader import download_movielens, load_dataframes


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

def make_spark(app_name: str, master: str, shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )

def split_ratings(ratings_df, seed: int, train_ratio: float, val_ratio: float):
    train_df, temp_df = ratings_df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    val_weight = val_ratio / (1 - train_ratio)
    val_df, test_df = temp_df.randomSplit([val_weight, 1 - val_weight], seed=seed)
    return train_df.cache(), val_df.cache(), test_df.cache()


def main():
    cfg = resolve_paths(load_config())

    data_root = Path(cfg["paths"]["data_root"]).resolve()
    models_root = Path(cfg["paths"]["models_dir"]).resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["dataset"]["seed"])
    train_ratio = float(cfg["dataset"]["split"]["train_ratio"])
    val_ratio = float(cfg["dataset"]["split"]["val_ratio"])

    # load best params from metrics.json
    save_dir = models_root / "models_simple"
    metrics_path = save_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run train.py first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_params = metrics["best_params"]

    # log dir
    run_dir = make_run_dir(models_root)
    started_at = utc_now_iso()

    save_json(run_dir / "evaluate_config.json", {
        "started_at_utc": started_at,
        "config_effective": cfg,
        "best_params_loaded": best_params,
        "env": {
            "user": getpass.getuser(),
            "python": os.sys.version,
            "platform": platform.platform(),
        }
    })
    print(f"[evaluate] Run dir: {run_dir}")

    spark = make_spark(
        app_name=cfg["spark"]["app_name_eval"],
        master=cfg["spark"]["master"],
        shuffle_partitions=int(cfg["spark"]["shuffle_partitions"]),
    )

    try:
        ds_path = download_movielens(data_root, url=cfg["dataset"]["movielens_url"])
        ratings_df, _movies_df = load_dataframes(spark, ds_path)
        ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating"])

        train_df, val_df, test_df = split_ratings(ratings_df, seed, train_ratio, val_ratio)
        trainval_df = train_df.unionByName(val_df)

        rmse_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        mae_eval = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

        als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy=cfg["als"].get("cold_start_strategy", "drop"),
            nonnegative=bool(cfg["als"].get("nonnegative", True)),
            rank=int(best_params["rank"]),
            regParam=float(best_params["regParam"]),
            maxIter=int(best_params["maxIter"]),
            seed=int(seed),
        )

        model = als.fit(trainval_df)
        preds = model.transform(test_df)

        rmse_test = float(rmse_eval.evaluate(preds))
        mae_test = float(mae_eval.evaluate(preds))

        ended_at = utc_now_iso()

        out = {
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "best_params": best_params,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "models_dir": str(save_dir),
        }

        # write logs
        save_json(run_dir / "evaluate.json", out)
        save_json(save_dir / "evaluate.json", out)

        print(f"[evaluate] ✅ RMSE(test)={rmse_test:.6f} | MAE(test)={mae_test:.6f}")
        print(f"[evaluate] ✅ Wrote: {run_dir / 'evaluate.json'}")
        print(f"[evaluate] ✅ Wrote: {save_dir / 'evaluate.json'}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()